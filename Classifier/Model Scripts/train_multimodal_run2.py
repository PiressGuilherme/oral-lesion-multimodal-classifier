#!/usr/bin/env python3
"""
Script de Treinamento Avançado (Run 2) para Classificação Multimodal.

Este script expande o treinamento anterior adicionando controles precisos sobre
o congelamento do backbone e a taxa de aprendizado (LR).

Recursos Adicionais desta Versão:
1. Controle de Freeze por Janela Explícita:
   - Define um intervalo [E_ini, E_fim] onde o backbone é forçado a ficar CONGELADO.
   - Fora dessa janela, aplica-se o modo de descongelamento configurado (all/partial).
   - Útil para estratégias de fine-tuning não lineares.

2. Modulação de LR em Duas Fases:
   - Divide o treino em Fase 1 e Fase 2, separadas pela época de fronteira (boundary).
   - Permite definir LRs distintos para o início e o fim do treinamento.

Entradas e Arquitetura permanecem as mesmas:
   - Imagem Oral + Lesão + Dados Tabulares.
   - Backbone substituível via timm.
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
# Métricas Auxiliares (Idêntico à versão anterior)
# =====================================================================

def confusion_matrix_mc(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Calcula a Matriz de Confusão para multiclasse."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def accuracy_mc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula a Acurácia Global."""
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).sum()) / float(len(y_true))


def macro_f1_mc(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Calcula o Macro F1-Score."""
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
    """Calcula o Coeficiente de Correlação de Matthews (MCC)."""
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
# Focal Loss (Idêntico à versão anterior)
# =====================================================================

class FocalLoss(nn.Module):
    """FocalLoss multiclasse com pesos integrados."""

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma * ce).mean()
        return loss


# =====================================================================
# Loop de Execução de Época (Train/Val)
# =====================================================================

def run_one_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, Any]:
    """
    Executa uma época completa.
    Nota: O ajuste de LR e o congelamento de backbone agora são feitos
    fora desta função (no loop principal), antes de chamar esta função.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0

    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

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

        loss = loss_fn(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
# Avaliação Final (Test)
# =====================================================================

def evaluate_on_test(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray]:
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
# NOVO: Helpers para controle do estado do backbone por época
# =====================================================================

def _apply_backbone_unfreeze_mode(model: nn.Module, args) -> str:
    """
    Auxiliar: Aplica o modo de descongelamento 'ativo' (quando NÃO estamos
    dentro de uma janela de congelamento forçado).
    Retorna uma string descrevendo o estado para logging.
    """
    mode = args.backbone_unfreeze_mode
    if mode == "all":
        model.unfreeze_backbone()
        return "descongelado_total"
    elif mode == "partial":
        model.unfreeze_backbone_last_fraction(args.backbone_unfreeze_fraction)
        return f"descongelado_parcial(frac={args.backbone_unfreeze_fraction:.2f})"
    elif mode == "none":
        model.freeze_backbone()
        return "congelado_total (modo=none)"
    else:
        model.freeze_backbone()
        return "congelado_total (modo=desconhecido)"


def set_backbone_mode_for_epoch(model: nn.Module, epoch: int, args) -> str:
    """
    Lógica Central de Controle do Backbone.
    Decide se o backbone deve estar congelado ou treinar nesta época específica.

    Prioridade de decisão:
    1. Janela Explícita (--freeze_epoch_start e end):
       Se a época atual estiver dentro desse intervalo, FORÇA o congelamento.

    2. Modo Legado (--freeze_backbone_epochs):
       Se não houver janela explícita, usa a lógica antiga: congela nas primeiras N épocas.

    Caso contrário, aplica o modo de descongelamento configurado (all/partial).
    """
    # Lógica da Janela Explícita
    if args.freeze_epoch_start > 0 and args.freeze_epoch_end > 0:
        if args.freeze_epoch_start <= epoch <= args.freeze_epoch_end:
            model.freeze_backbone()
            return "congelado (janela_explicita)"
        else:
            return _apply_backbone_unfreeze_mode(model, args)
    else:
        # Lógica "Antiga": congela início, depois solta
        if epoch <= args.freeze_backbone_epochs:
            model.freeze_backbone()
            return "congelado_inicial"
        else:
            return _apply_backbone_unfreeze_mode(model, args)


# =====================================================================
# Função Principal (Main)
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Pasta do dataset.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Número total de épocas.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Tamanho do batch.",
    )

    # ----------------------------------------------------------------
    # Parâmetros de Learning Rate (Modulação em 2 Fases)
    # ----------------------------------------------------------------
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="LR base (usado como fallback se as fases não forem definidas).",
    )
    parser.add_argument(
        "--phase_boundary_epoch",
        type=int,
        default=None,
        help=(
            "Época de corte entre Fase 1 e Fase 2. "
            "Ex: Se 10, épocas 1-10 usam lr_phase1, épocas 11+ usam lr_phase2."
        ),
    )
    parser.add_argument(
        "--lr_phase1",
        type=float,
        default=None,
        help="LR da Fase 1 (Início). Se None, usa --lr.",
    )
    parser.add_argument(
        "--lr_phase2",
        type=float,
        default=None,
        help="LR da Fase 2 (Final). Se None, usa --lr.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Decaimento de pesos (L2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="'cuda' ou 'cpu'.",
    )

    # ----------------------------------------------------------------
    # Parâmetros de Congelamento (Legado e Novo)
    # ----------------------------------------------------------------
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=0,
        help="Modo Antigo: Número de épocas iniciais congeladas.",
    )
    parser.add_argument(
        "--freeze_epoch_start",
        type=int,
        default=0,
        help="Modo Novo: Época INICIAL da janela de congelamento forçado.",
    )
    parser.add_argument(
        "--freeze_epoch_end",
        type=int,
        default=0,
        help="Modo Novo: Época FINAL da janela de congelamento forçado.",
    )

    # Comportamento fora do congelamento
    parser.add_argument(
        "--backbone_unfreeze_mode",
        type=str,
        default="all",
        choices=["all", "partial", "none"],
        help="Modo do backbone quando NÃO está congelado (all/partial/none).",
    )
    parser.add_argument(
        "--backbone_unfreeze_fraction",
        type=float,
        default=0.3,
        help="Fração de descongelamento para o modo 'partial'. Ex: 0.3",
    )

    # Parâmetros gerais de modelo/loss
    parser.add_argument(
        "--loss_type",
        type=str,
        default="ce",
        choices=["ce", "focal"],
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="convnext_tiny",
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--class_weights",
        type=str,
        default=None,
        help="Pesos manuais 'w0,w1,w2,w3'.",
    )

    args = parser.parse_args()

    # Configuração de Hardware
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("[AVISO] CUDA indisponível, usando CPU.", file=sys.stderr)
        device = torch.device("cpu")

    # Sementes
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # -----------------------------------------------------------------
    # Configuração das Fases de LR e Janelas
    # -----------------------------------------------------------------
    # Define o limite da fase. Se inválido, a Fase 1 dura o treino todo.
    if args.phase_boundary_epoch is None or args.phase_boundary_epoch <= 0:
        phase_boundary_epoch = args.epochs
    else:
        phase_boundary_epoch = min(args.phase_boundary_epoch, args.epochs)

    lr_phase1 = args.lr_phase1 if args.lr_phase1 is not None else args.lr
    lr_phase2 = args.lr_phase2 if args.lr_phase2 is not None else args.lr

    # Valida janela de freeze
    if args.freeze_epoch_start > 0:
        args.freeze_epoch_start = max(1, args.freeze_epoch_start)
    if args.freeze_epoch_end > 0:
        args.freeze_epoch_end = min(args.freeze_epoch_end, args.epochs)

    # Logs iniciais
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Backbone: {args.backbone}")
    print(f"[INFO] Épocas: {args.epochs}")
    print(
        f"[INFO] Controle de LR: "
        f"Fase 1 (1..{phase_boundary_epoch}) lr={lr_phase1:.2e} | "
        f"Fase 2 ({phase_boundary_epoch+1}..{args.epochs}) lr={lr_phase2:.2e}"
    )
    print(
        f"[INFO] Janela Explícita de Freeze: "
        f"[{args.freeze_epoch_start}, {args.freeze_epoch_end}] "
        "(0 indica desativado)"
    )
    print(f"[INFO] Modo de Descongelamento: {args.backbone_unfreeze_mode} (frac={args.backbone_unfreeze_fraction})")

    # -----------------------------------------------------------------
    # Carga de Dados e Modelo
    # -----------------------------------------------------------------
    print("[INFO] Criando DataLoaders...")
    train_loader, val_loader, test_loader, class_weights_dl = create_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    # Processamento de pesos de classe
    if isinstance(class_weights_dl, dict):
        keys = sorted(class_weights_dl.keys())
        weights_list = [float(class_weights_dl[k]) for k in keys]
        class_weights = torch.tensor(weights_list, dtype=torch.float32, device=device)
    elif isinstance(class_weights_dl, torch.Tensor):
        class_weights = class_weights_dl.to(device=device, dtype=torch.float32)
    else:
        class_weights = torch.tensor(class_weights_dl, dtype=torch.float32, device=device)

    if args.class_weights is not None:
        parts = [p.strip() for p in args.class_weights.split(",") if p.strip() != ""]
        manual = torch.tensor([float(p) for p in parts], dtype=torch.float32, device=device)
        class_weights = manual
        print("[INFO] Usando PESOS MANUAIS.")
    else:
        print("[INFO] Usando PESOS AUTOMÁTICOS.")

    # Instanciação do Modelo
    num_tab_features = len(TABULAR_COLUMNS)
    num_classes = 4

    model = MultiModalMobileNetV3Large(
        num_tab_features=num_tab_features,
        num_classes=num_classes,
        dropout_p=args.dropout_p,
        backbone_name=args.backbone,
        pretrained=True,
    ).to(device)

    # Configuração da Loss
    if args.loss_type == "ce":
        loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)
    else:
        loss_fn = FocalLoss(alpha=class_weights, gamma=args.focal_gamma).to(device)

    # Otimizador (Inicia com LR da Fase 1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_phase1,
        weight_decay=args.weight_decay,
    )

    # -----------------------------------------------------------------
    # Loop de Treino
    # -----------------------------------------------------------------
    best_val_macro_f1 = -1.0
    best_model_path = os.path.join(args.dataset_dir, "best_model.pt")
    history: List[Dict[str, Any]] = []

    current_lr = None
    backbone_state = None

    for epoch in range(1, args.epochs + 1):
        print(f"\n========== Época {epoch}/{args.epochs} ==========")

        # 1) Aplica a LR correta para a fase atual
        if epoch <= phase_boundary_epoch:
            target_lr = lr_phase1
        else:
            target_lr = lr_phase2

        # Atualiza o otimizador se a LR mudou
        if current_lr is None or abs(current_lr - target_lr) > 1e-12:
            for g in optimizer.param_groups:
                g["lr"] = target_lr
            current_lr = target_lr
            print(f"[INFO] LR ajustado para {current_lr:.2e}")

        # 2) Gerencia o estado do backbone (Congela/Descongela)
        new_backbone_state = set_backbone_mode_for_epoch(model, epoch, args)
        if new_backbone_state != backbone_state:
            print(f"[INFO] Estado do Backbone alterado para: {new_backbone_state}")
            backbone_state = new_backbone_state

        # 3) Passo de Treino
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
            f"f1={train_metrics['macro_f1']:.4f}  "
        )

        # Registra no histórico
        history.append({
            "epoch": epoch,
            "split": "train",
            "loss": train_metrics["loss"],
            "acc": train_metrics["acc"],
            "macro_f1": train_metrics["macro_f1"],
            "mcc": train_metrics["mcc"],
            "lr": current_lr,              # NOVO: Salva a LR usada
            "backbone_state": backbone_state # NOVO: Salva o estado
        })

        # 4) Passo de Validação
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
            f"f1={val_metrics['macro_f1']:.4f}  "
        )

        history.append({
            "epoch": epoch,
            "split": "val",
            "loss": val_metrics["loss"],
            "acc": val_metrics["acc"],
            "macro_f1": val_metrics["macro_f1"],
            "mcc": val_metrics["mcc"],
            "lr": current_lr,
            "backbone_state": backbone_state
        })

        # 5) Checkpoint
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] Modelo salvo (F1: {best_val_macro_f1:.4f})")

    # Salva histórico
    hist_path = os.path.join(args.dataset_dir, "training_history.csv")
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(hist_path, index=False)
    print(f"\n[OK] Histórico salvo em: {hist_path}")

    # Avaliação Final (Teste)
    print("\n[INFO] Avaliando no TESTE com melhor modelo...")
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    test_metrics, cm = evaluate_on_test(model=model, loader=test_loader, device=device)
    print(f"[TEST] Acc={test_metrics['acc']:.4f} F1={test_metrics['macro_f1']:.4f}")
    print(cm)


if __name__ == "__main__":
    main()