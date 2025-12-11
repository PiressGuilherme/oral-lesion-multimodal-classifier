#!/usr/bin/env python3
"""
Script de Treinamento do Modelo Multimodal para Classificação de Lesões Orais.

Este script orquestra todo o pipeline de treinamento, validação e teste.

Características principais:
1. Entradas Multimodais:
   - Imagem da Cavidade Oral (Contexto).
   - Imagem da Lesão/ROI (Foco), quando disponível.
   - Dados Tabulares (Idade, Sexo, Hábitos de risco).

2. Flexibilidade de Arquitetura (Backbone):
   - Permite trocar a rede neural de visão (backbone) via argumento de linha de comando
     (ex: 'convnext_tiny', 'resnet50', 'efficientnet_b2'), utilizando a biblioteca `timm`.

3. Estratégias de Balanceamento de Classes:
   - WeightedRandomSampler (via DataLoader): Mostra classes raras com mais frequência.
   - Loss Ponderada: Penaliza mais os erros nas classes minoritárias.
   - Focal Loss (Opcional): Foca o aprendizado em exemplos "difíceis" de classificar.

4. Estratégias de Fine-Tuning (Congelamento/Descongelamento):
   - Começa com o backbone congelado (apenas treina o classificador).
   - Após N épocas, pode descongelar:
     - Tudo ('all').
     - Apenas as camadas finais ('partial').
     - Nada ('none').

Saídas:
   - Salva o melhor modelo ('best_model.pt') baseado no F1-Score da validação.
   - Salva um CSV com o histórico de métricas por época ('training_history.csv').
   - Imprime a matriz de confusão final no conjunto de teste.
"""

import os
import sys
import argparse
import random
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from dataset_multimodal import create_dataloaders, TABULAR_COLUMNS
from model_multimodal import MultiModalMobileNetV3Large


# =====================================================================
# Métricas Auxiliares (Implementação manual para evitar dep. do sklearn)
# =====================================================================

def confusion_matrix_mc(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Calcula a Matriz de Confusão para problemas multiclasse.
    Linhas: Verdadeiro (Ground Truth).
    Colunas: Predito pelo modelo.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def accuracy_mc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula a Acurácia global (Total de acertos / Total de amostras)."""
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).sum()) / float(len(y_true))


def macro_f1_mc(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """
    Calcula o Macro F1-Score.
    1. Calcula o F1 para cada classe individualmente.
    2. Tira a média aritmética dos F1s (tratando todas as classes com igual importância).
    """
    cm = confusion_matrix_mc(y_true, y_pred, num_classes)
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        if denom == 0:
            f1 = 0.0
        else:
            f1 = (2.0 * tp) / denom
        f1s.append(f1)
    return float(np.mean(f1s))


def mcc_multiclass(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """
    Calcula o Coeficiente de Correlação de Matthews (MCC) para multiclasse.
    É uma métrica robusta para datasets desbalanceados, variando de -1 a +1.
    """
    cm = confusion_matrix_mc(y_true, y_pred, num_classes)
    t_k = cm.sum(axis=1)
    p_k = cm.sum(axis=0)
    c = np.trace(cm)
    s = cm.sum()
    if s == 0:
        return 0.0
    sum_tk_pk = float(np.sum(t_k * p_k))
    num = c * s - sum_tk_pk
    denom_left = s ** 2 - float(np.sum(p_k ** 2))
    denom_right = s ** 2 - float(np.sum(t_k ** 2))
    denom = np.sqrt(denom_left * denom_right)
    if denom == 0:
        return 0.0
    return float(num / denom)


# =====================================================================
# Função de Perda Personalizada: Focal Loss
# =====================================================================

class FocalLoss(nn.Module):
    """
    Implementação da Focal Loss para classificação multiclasse.
    Útil quando há desbalanceamento severo ou exemplos "difíceis".

    Fórmula: Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Tensor com pesos por classe (para lidar com desbalanceamento).
        gamma: Fator de foco. Quanto maior, mais o modelo ignora exemplos fáceis
               e foca nos difíceis. Padrão comum é 2.0.
    """

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calcula a Cross Entropy padrão sem redução (por amostra)
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        # Calcula p_t (probabilidade da classe correta)
        pt = torch.exp(-ce)
        # Aplica o termo focal (1 - pt)^gamma
        loss = ((1 - pt) ** self.gamma * ce).mean()
        return loss


# =====================================================================
# Loop de Execução de Uma Época (Treino ou Validação)
# =====================================================================

def run_one_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, Any]:
    """
    Executa uma passagem completa pelo dataset (uma época).
    Se 'optimizer' for fornecido, executa o passo de treino (backward).
    Caso contrário, executa apenas avaliação (modo validação).

    Retorna: Dicionário com a média da loss e das métricas (Acc, F1, MCC).
    """
    is_train = optimizer is not None
    model.train(is_train)  # Ativa/Desativa Dropout e BatchNorm

    total_loss = 0.0
    total_samples = 0

    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    for batch in loader:
        # Move os dados para GPU/CPU
        oral = batch["oral_image"].to(device)
        lesion = batch["lesion_image"].to(device)
        tab = batch["tabular"].to(device)
        has_lesion = batch["has_lesion"].to(device)
        labels = batch["label"].to(device)

        # Forward Pass
        logits = model(
            oral_image=oral,
            lesion_image=lesion,
            tabular=tab,
            has_lesion=has_lesion,
        )

        # Cálculo da Perda
        loss = loss_fn(logits, labels)

        # Backward Pass (apenas no treino)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Acumula estatísticas
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = logits.argmax(dim=1)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    if total_samples == 0:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "macro_f1": 0.0,
            "mcc": 0.0,
        }

    # Concatena todas as predições da época para calcular métricas globais
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    avg_loss = total_loss / total_samples
    acc = accuracy_mc(y_true, y_pred)
    mf1 = macro_f1_mc(y_true, y_pred, num_classes=4)
    mcc = mcc_multiclass(y_true, y_pred, num_classes=4)

    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": mf1,
        "mcc": mcc,
    }


# =====================================================================
# Avaliação Final no Conjunto de Teste
# =====================================================================

def evaluate_on_test(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Similar ao loop de validação, mas focado em gerar resultados finais.
    Retorna as métricas e também a Matriz de Confusão.
    """
    model.eval()
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            oral = batch["oral_image"].to(device)
            lesion = batch["lesion_image"].to(device)
            tab = batch["tabular"].to(device)
            has_lesion = batch["has_lesion"].to(device)
            labels = batch["label"].to(device)

            logits = model(
                oral_image=oral,
                lesion_image=lesion,
                tabular=tab,
                has_lesion=has_lesion,
            )

            preds = logits.argmax(dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    if not all_labels:
        metrics = {"loss": 0.0, "acc": 0.0, "macro_f1": 0.0, "mcc": 0.0}
        cm = np.zeros((4, 4), dtype=int)
        return metrics, cm

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    acc = accuracy_mc(y_true, y_pred)
    mf1 = macro_f1_mc(y_true, y_pred, num_classes=4)
    mcc = mcc_multiclass(y_true, y_pred, num_classes=4)
    cm = confusion_matrix_mc(y_true, y_pred, num_classes=4)

    metrics = {
        "acc": acc,
        "macro_f1": mf1,
        "mcc": mcc,
    }
    return metrics, cm


# =====================================================================
# Função Principal (Main)
# =====================================================================

def main():
    # Configuração de argumentos via linha de comando
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Pasta raiz do dataset (contendo manifest.csv, pasta roi/, etc.).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Número total de épocas de treino.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Tamanho do lote (batch size).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Taxa de aprendizado inicial (Learning Rate).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Decaimento de peso (L2 Regularization) para o otimizador.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Dispositivo de processamento: 'cuda' (GPU) ou 'cpu'.",
    )
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=5,
        help="Quantas épocas o backbone ficará congelado no início do treino.",
    )
    parser.add_argument(
        "--backbone_unfreeze_mode",
        type=str,
        default="all",
        choices=["all", "partial", "none"],
        help=(
            "Como descongelar o backbone após o período inicial: "
            "'all' = descongela tudo, "
            "'partial' = descongela apenas a fração final (camadas profundas), "
            "'none' = mantém congelado até o fim."
        ),
    )
    parser.add_argument(
        "--backbone_unfreeze_fraction",
        type=float,
        default=0.3,
        help=(
            "Se usar modo 'partial', define a fração de parâmetros a descongelar. "
            "Ex.: 0.3 significa que os últimos 30%% da rede serão treináveis."
        ),
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="Tipo de função de perda: 'ce' (CrossEntropy) ou 'focal' (FocalLoss).",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Valor de Gamma para a FocalLoss (se selecionada).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="convnext_tiny",
        help=(
            "Nome do backbone de visão a ser baixado do timm. "
            "Ex: 'convnext_tiny', 'mobilenetv3_large_100', 'resnet18'."
        ),
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        default=0.3,
        help="Probabilidade de Dropout nas camadas lineares (MLP).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Dimensão para redimensionamento das imagens (H e W).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Número de subprocessos para carregar dados (0 = main thread).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória para reprodutibilidade.",
    )
    parser.add_argument(
        "--class_weights",
        type=str,
        default=None,
        help=(
            "Pesos manuais para as classes (string 'w0,w1,w2,w3'). "
            "Ordem: [0=Healthy, 1=Benign, 2=OPMD, 3=OCA]. "
            "Se omitido, calcula automaticamente baseado na frequência do treino."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Configuração de Hardware (Device)
    # -----------------------------------------------------------------
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("[AVISO] CUDA solicitado mas não disponível. Usando CPU.", file=sys.stderr)
        device = torch.device("cpu")

    # -----------------------------------------------------------------
    # Reprodutibilidade (Seeds)
    # -----------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Log dos parâmetros escolhidos
    print(f"[INFO] Usando device: {device}")
    print(f"[INFO] Usando backbone: {args.backbone}")
    print(f"[INFO] Usando dropout_p: {args.dropout_p}")
    print(f"[INFO] Épocas de treino: {args.epochs}")
    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"[INFO] Learning rate: {args.lr}")
    print(f"[INFO] Weight decay: {args.weight_decay}")
    print(f"[INFO] Loss type: {args.loss_type}")
    print(f"[INFO] Focal gamma: {args.focal_gamma}")
    print(f"[INFO] freeze_backbone_epochs: {args.freeze_backbone_epochs}")
    print(f"[INFO] backbone_unfreeze_mode: {args.backbone_unfreeze_mode}")
    print(f"[INFO] backbone_unfreeze_fraction: {args.backbone_unfreeze_fraction}")
    if args.class_weights is not None:
        print(f"[INFO] Pesos manuais definidos: {args.class_weights}")

    # -----------------------------------------------------------------
    # Criação dos DataLoaders
    # -----------------------------------------------------------------
    print("[INFO] Criando DataLoaders...")
    train_loader, val_loader, test_loader, class_weights_dl = create_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    # -----------------------------------------------------------------
    # Tratamento dos Pesos de Classe (Automático vs Manual)
    # -----------------------------------------------------------------
    # Converte os pesos vindos do DataLoader para tensor
    if isinstance(class_weights_dl, dict):
        keys = sorted(class_weights_dl.keys())
        weights_list = [float(class_weights_dl[k]) for k in keys]
        class_weights = torch.tensor(
            weights_list, dtype=torch.float32, device=device
        )
    elif isinstance(class_weights_dl, torch.Tensor):
        class_weights = class_weights_dl.to(device=device, dtype=torch.float32)
    else:
        class_weights = torch.tensor(
            class_weights_dl, dtype=torch.float32, device=device
        )

    # Se o usuário passou pesos manuais via argumento, sobrescreve os automáticos
    if args.class_weights is not None:
        parts = [p.strip() for p in args.class_weights.split(",") if p.strip() != ""]
        if len(parts) != 4:
            raise ValueError(
                f"--class_weights deve ter exatamente 4 valores. Recebido: {args.class_weights}"
            )
        manual = torch.tensor([float(p) for p in parts], dtype=torch.float32, device=device)
        class_weights = manual
        print("[INFO] Usando PESOS MANUAIS para a loss.")
    else:
        print("[INFO] Usando PESOS AUTOMÁTICOS (baseados na frequência do treino).")

    for c, w in enumerate(class_weights.detach().cpu().numpy()):
        print(f"  Classe {c}: peso = {w:.4f}")

    # -----------------------------------------------------------------
    # Instanciação do Modelo
    # -----------------------------------------------------------------
    num_tab_features = len(TABULAR_COLUMNS)
    num_classes = 4

    model = MultiModalMobileNetV3Large(
        num_tab_features=num_tab_features,
        num_classes=num_classes,
        dropout_p=args.dropout_p,
        tab_hidden_dim=64,
        fusion_hidden_dim=256,
        backbone_name=args.backbone,
        pretrained=True,
    ).to(device)

    # Congela o backbone inicialmente para estabilizar o treino do classificador
    model.freeze_backbone()
    print(f"[INFO] Congelando backbone nas primeiras {args.freeze_backbone_epochs} épocas.")

    # -----------------------------------------------------------------
    # Definição da Loss e Otimizador
    # -----------------------------------------------------------------
    if args.loss_type == "ce":
        loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)
        print("[INFO] Usando CrossEntropyLoss ponderada.")
    else:
        loss_fn = FocalLoss(alpha=class_weights, gamma=args.focal_gamma).to(device)
        print("[INFO] Usando FocalLoss ponderada.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # -----------------------------------------------------------------
    # Loop de Treinamento
    # -----------------------------------------------------------------
    best_val_macro_f1 = -1.0
    best_model_path = os.path.join(args.dataset_dir, "best_model.pt")
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n========== Época {epoch}/{args.epochs} ==========")

        # Lógica de descongelamento do backbone após N épocas
        if epoch == args.freeze_backbone_epochs + 1:
            if args.backbone_unfreeze_mode == "all":
                model.unfreeze_backbone()
                print("[INFO] Descongelando backbone COMPLETO para fine-tuning.")
            elif args.backbone_unfreeze_mode == "partial":
                model.unfreeze_backbone_last_fraction(args.backbone_unfreeze_fraction)
                print(
                    "[INFO] Descongelando PARCIALMENTE o backbone "
                    f"(fração = {args.backbone_unfreeze_fraction:.2f})."
                )
            elif args.backbone_unfreeze_mode == "none":
                print("[INFO] Modo 'none': backbone permanecerá congelado durante todo o treino.")

        # 1. Passo de Treino
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        print(
            f"[TRAIN] loss={train_metrics['loss']:.4f}  "
            f"acc={train_metrics['acc']:.4f}  "
            f"macro_f1={train_metrics['macro_f1']:.4f}  "
            f"mcc={train_metrics['mcc']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "split": "train",
            "loss": train_metrics["loss"],
            "acc": train_metrics["acc"],
            "macro_f1": train_metrics["macro_f1"],
            "mcc": train_metrics["mcc"],
        })

        # 2. Passo de Validação
        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=None,
        )
        print(
            f"[VAL]   loss={val_metrics['loss']:.4f}  "
            f"acc={val_metrics['acc']:.4f}  "
            f"macro_f1={val_metrics['macro_f1']:.4f}  "
            f"mcc={val_metrics['mcc']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "split": "val",
            "loss": val_metrics["loss"],
            "acc": val_metrics["acc"],
            "macro_f1": val_metrics["macro_f1"],
            "mcc": val_metrics["mcc"],
        })

        # 3. Checkpoint: Salva se o F1 da validação melhorou
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_model_path)
            print(
                f"[INFO] Novo melhor modelo salvo em {best_model_path} "
                f"(val_macro_f1={best_val_macro_f1:.4f})"
            )

    # -----------------------------------------------------------------
    # Finalização
    # -----------------------------------------------------------------
    hist_path = os.path.join(args.dataset_dir, "training_history.csv")
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(hist_path, index=False)
    print(f"\n[OK] Histórico de treino salvo em: {hist_path}")

    # -----------------------------------------------------------------
    # Avaliação no Conjunto de Teste (usando o melhor modelo)
    # -----------------------------------------------------------------
    print("\n[INFO] Carregando melhor modelo para avaliação no TEST...")
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    test_metrics, cm = evaluate_on_test(
        model=model,
        loader=test_loader,
        device=device,
    )

    print("\n========== RESULTADOS FINAIS (TEST) ==========")
    print(
        f"[TEST] acc={test_metrics['acc']:.4f}  "
        f"macro_f1={test_metrics['macro_f1']:.4f}  "
        f"mcc={test_metrics['mcc']:.4f}"
    )
    print("\nMatriz de confusão (linhas = verdade, colunas = predito):")
    print(cm)


if __name__ == "__main__":
    main()