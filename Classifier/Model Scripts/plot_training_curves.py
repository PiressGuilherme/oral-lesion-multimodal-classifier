#!/usr/bin/env python3
"""
Script de Visualização e Avaliação Final do Modelo.

Este script executa duas tarefas principais:
1. Análise do Histórico de Treino:
   - Lê o arquivo 'training_history.csv'.
   - Gera gráficos de linha comparando Treino vs Validação para:
     Loss, Acurácia, Macro F1 e MCC.

2. Avaliação Profunda no Conjunto de Teste:
   - Carrega o peso do melhor modelo salvo ('best_model.pt').
   - Executa inferência no conjunto de teste.
   - Gera visualizações avançadas:
     - Matriz de Confusão (Heatmap).
     - Gráfico de Barras de F1-Score por classe.
     - Curvas Precision-Recall (PR) para classes críticas (OCA e OPMD).
     - Curvas ROC (Receiver Operating Characteristic) para classes críticas.

Pré-requisitos:
   - É necessário ter a biblioteca 'scikit-learn' instalada para curvas ROC/PR.
   - O modelo deve ser instanciado com os MESMOS parâmetros usados no treino
     (mesmo backbone, mesmo dropout, etc.).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# Tenta importar scikit-learn para métricas avançadas.
# Se não estiver instalado, o script avisa e encerra.
try:
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
except ImportError as e:
    print(
        "[ERRO] A biblioteca scikit-learn (sklearn) é necessária para gerar curvas PR/ROC.\n"
        "Instale com: pip install scikit-learn",
        file=sys.stderr,
    )
    raise

from dataset_multimodal import create_dataloaders, TABULAR_COLUMNS
from model_multimodal import MultiModalMobileNetV3Large


# -------------------------------------------------------------------
# Bloco 1: Funções para Plotar Curvas de Treino vs Validação
# -------------------------------------------------------------------

def plot_metric(df: pd.DataFrame, metric: str, out_dir: str) -> None:
    """
    Gera e salva um gráfico de linha comparando a evolução de uma métrica
    (ex: loss, acc) entre os conjuntos de treino e validação ao longo das épocas.
    """
    if metric not in df.columns:
        print(f"[AVISO] Métrica '{metric}' não encontrada no CSV. Pulando gráfico.")
        return

    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "val"]

    if df_train.empty or df_val.empty:
        print(f"[AVISO] Split de treino ou validação vazio para '{metric}'. Pulando.")
        return

    epochs = df_train["epoch"].values

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df_train[metric].values, marker="o", label="Treino")
    plt.plot(epochs, df_val[metric].values, marker="o", label="Validação")

    plt.xlabel("Época")
    plt.ylabel(metric)
    plt.title(f"Evolução de {metric} (Treino vs Validação)")
    plt.grid(True, linestyle="--", alpha=0.4)

    # Ajuste dos limites do eixo Y para melhor visualização
    if metric == "loss":
        # Loss não tem teto definido, mas o chão é 0
        plt.ylim(bottom=0)
    elif metric == "mcc":
        # Correlação de Matthews vai de -1 a +1
        plt.ylim(-1.0, 1.0)
    else:
        # Acurácia, F1, etc., variam de 0 a 1
        plt.ylim(0.0, 1.0)

    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"curve_{metric}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[OK] Gráfico salvo: {out_path}")


def generate_training_curves(history_path: str, out_dir: str) -> None:
    """
    Lê o arquivo CSV de histórico, valida sua estrutura e chama a função de plotagem
    para cada métrica disponível. Também imprime um resumo das melhores épocas.
    """
    if not os.path.isfile(history_path):
        print(f("[ERRO] Arquivo não encontrado: {history_path}"), file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Lendo histórico de treino: {history_path}")
    df = pd.read_csv(history_path)

    required_cols = {"epoch", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERRO] Colunas obrigatórias faltando no CSV: {missing}", file=sys.stderr)
        sys.exit(1)

    df["epoch"] = df["epoch"].astype(int)
    df["split"] = df["split"].astype(str)
    df = df.sort_values(["epoch", "split"]).reset_index(drop=True)

    metrics = ["loss", "acc", "macro_f1", "mcc"]
    print("[INFO] Gerando gráficos para as métricas:", ", ".join(metrics))
    for metric in metrics:
        plot_metric(df, metric, out_dir)

    # Identifica e imprime as melhores épocas baseadas na validação
    df_val = df[df["split"] == "val"].copy()
    if not df_val.empty:
        best_loss_row = df_val.loc[df_val["loss"].idxmin()]
        best_f1_row = df_val.loc[df_val["macro_f1"].idxmax()]
        best_mcc_row = df_val.loc[df_val["mcc"].idxmax()]

        print("\n[RESUMO DAS MELHORES ÉPOCAS (Validação)]:")
        print(f"  Menor Loss:      Época {int(best_loss_row['epoch'])} | Valor: {best_loss_row['loss']:.4f}")
        print(f"  Maior Macro F1:  Época {int(best_f1_row['epoch'])} | Valor: {best_f1_row['macro_f1']:.4f}")
        print(f"  Maior MCC:       Época {int(best_mcc_row['epoch'])} | Valor: {best_mcc_row['mcc']:.4f}")


# -------------------------------------------------------------------
# Bloco 2: Avaliação no Conjunto de Teste
# -------------------------------------------------------------------

def evaluate_test_and_collect(
    model_path,
    dataset_dir,
    device,
    img_size,
    batch_size,
    backbone,
    dropout_p,
):
    """
    Recarrega o modelo treinado (best_model.pt) e executa a inferência em todo o
    conjunto de TESTE.

    Retorna um dicionário contendo:
        - Rótulos verdadeiros (y_true)
        - Rótulos preditos (y_pred)
        - Probabilidades brutas (y_proba) para curvas ROC/PR
        - Matriz de confusão calculada
        - F1-Score por classe
    """
    # Se o caminho do modelo não for passado explicitamente, assume o padrão
    if model_path is None:
        model_path = os.path.join(dataset_dir, "best_model.pt")

    if not os.path.isfile(model_path):
        print(f"[AVISO] Modelo não encontrado em {model_path}. "
              "Pulando etapa de avaliação no teste.", file=sys.stderr)
        return None

    print(f"\n[INFO] Carregando modelo de: {model_path}")
    print("[INFO] Criando DataLoaders (foco no split de TESTE)...")

    # Cria dataloaders (train/val serão ignorados aqui, usaremos apenas test_loader)
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=0,
        img_size=img_size,
    )

    # Recria a arquitetura do modelo
    # IMPORTANTE: Os parâmetros aqui (backbone, dropout) DEVEM ser os mesmos do treino
    num_tab_features = len(TABULAR_COLUMNS)
    num_classes = 4

    model = MultiModalMobileNetV3Large(
        num_tab_features=num_tab_features,
        num_classes=num_classes,
        dropout_p=dropout_p,
        tab_hidden_dim=64,
        fusion_hidden_dim=256,
        backbone_name=backbone,
        pretrained=False,  # False pois vamos carregar os pesos do arquivo .pt
    ).to(device)

    # Carrega os pesos treinados
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    # Loop de inferência
    with torch.no_grad():
        for batch in test_loader:
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
            # Aplica Softmax para obter probabilidades (0 a 1)
            probs = F.softmax(logits, dim=1)
            # A classe com maior probabilidade é a predição final
            preds = torch.argmax(probs, dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    if not all_labels:
        print("[AVISO] Nenhuma amostra encontrada no split de TESTE.", file=sys.stderr)
        return None

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_proba = np.concatenate(all_probs)

    # Cálculo manual da Matriz de Confusão
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    # Cálculo manual do F1 por classe
    f1_per_class = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = (2 * tp + fp + fn)
        f1 = 0.0 if denom == 0 else (2.0 * tp) / denom
        f1_per_class.append(f1)

    print("\n[INFO] Avaliação no TESTE concluída.")
    print("      F1 por classe [Healthy, Benign, OPMD, OCA]:")
    print("      ", [round(x, 4) for x in f1_per_class])

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "cm": cm,
        "f1_per_class": f1_per_class,
    }


# -------------------------------------------------------------------
# Bloco 3: Funções de Plotagem (Matriz, F1, PR, ROC)
# -------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, class_names, out_dir: str) -> None:
    """Gera e salva o heatmap da Matriz de Confusão."""
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Matriz de Confusão - Teste")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Adiciona os números dentro de cada célula do heatmap
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("Classe Verdadeira")
    plt.xlabel("Classe Predita")
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, "confusion_matrix_test.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Heatmap da matriz de confusão salvo em: {out_path}")


def plot_f1_per_class(f1s, class_names, out_dir: str) -> None:
    """Gera e salva um gráfico de barras com o F1-Score de cada classe."""
    plt.figure(figsize=(6, 4))
    x = np.arange(len(class_names))
    plt.bar(x, f1s)
    plt.xticks(x, class_names)
    plt.ylim(0, 1.0) 
    plt.ylabel("F1-score")
    plt.title("F1-score por classe (Teste)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, "f1_per_class_test.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Gráfico de F1 por classe salvo em: {out_path}")


def plot_pr_curves(y_true: np.ndarray, y_proba: np.ndarray, out_dir: str) -> None:
    """
    Gera Curvas Precision-Recall (PR) comparando:
    1. OCA vs Todo o resto
    2. OPMD vs Todo o resto
    Útil para avaliar a qualidade da detecção de classes críticas.
    """
    # Binariza o problema: Classe de interesse = 1, Resto = 0
    y_pos_oca = (y_true == 3).astype(int)
    scores_oca = y_proba[:, 3] # Probabilidade da coluna 3 (OCA)

    y_pos_opmd = (y_true == 2).astype(int)
    scores_opmd = y_proba[:, 2] # Probabilidade da coluna 2 (OPMD)

    plt.figure(figsize=(7, 5))
    any_curve = False

    # Plot OCA
    if y_pos_oca.sum() > 0 and y_pos_oca.sum() < len(y_pos_oca):
        precision_oca, recall_oca, _ = precision_recall_curve(y_pos_oca, scores_oca)
        ap_oca = auc(recall_oca, precision_oca)
        plt.plot(recall_oca, precision_oca, label=f"OCA vs resto (AP≈{ap_oca:.3f})")
        any_curve = True
    else:
        print("[AVISO] Impossível calcular PR para OCA (classe ausente ou única no teste).")

    # Plot OPMD
    if y_pos_opmd.sum() > 0 and y_pos_opmd.sum() < len(y_pos_opmd):
        precision_opmd, recall_opmd, _ = precision_recall_curve(y_pos_opmd, scores_opmd)
        ap_opmd = auc(recall_opmd, precision_opmd)
        plt.plot(recall_opmd, precision_opmd, label=f"OPMD vs resto (AP≈{ap_opmd:.3f})")
        any_curve = True
    else:
        print("[AVISO] Impossível calcular PR para OPMD (classe ausente ou única no teste).")

    if not any_curve:
        plt.close()
        print("[AVISO] Nenhuma curva PR foi gerada.")
        return

    plt.xlabel("Recall (Sensibilidade)")
    plt.ylabel("Precision (Precisão)")
    plt.title("Curvas Precision-Recall (Teste)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "pr_curves_oca_opmd.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Curvas Precision-Recall salvas em: {out_path}")


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, out_dir: str) -> None:
    """
    Gera Curvas ROC (Receiver Operating Characteristic) comparando:
    1. OCA vs Resto
    2. OPMD vs Resto
    """
    y_pos_oca = (y_true == 3).astype(int)
    scores_oca = y_proba[:, 3]

    y_pos_opmd = (y_true == 2).astype(int)
    scores_opmd = y_proba[:, 2]

    plt.figure(figsize=(7, 5))
    any_curve = False

    if y_pos_oca.sum() > 0 and y_pos_oca.sum() < len(y_pos_oca):
        fpr_oca, tpr_oca, _ = roc_curve(y_pos_oca, scores_oca)
        auc_oca = auc(fpr_oca, tpr_oca)
        plt.plot(fpr_oca, tpr_oca, label=f"OCA vs resto (AUC={auc_oca:.3f})")
        any_curve = True
    else:
        print("[AVISO] Impossível calcular ROC para OCA.")

    if y_pos_opmd.sum() > 0 and y_pos_opmd.sum() < len(y_pos_opmd):
        fpr_opmd, tpr_opmd, _ = roc_curve(y_pos_opmd, scores_opmd)
        auc_opmd = auc(fpr_opmd, tpr_opmd)
        plt.plot(fpr_opmd, tpr_opmd, label=f"OPMD vs resto (AUC={auc_opmd:.3f})")
        any_curve = True
    else:
        print("[AVISO] Impossível calcular ROC para OPMD.")

    if not any_curve:
        plt.close()
        print("[AVISO] Nenhuma curva ROC foi gerada.")
        return

    plt.plot([0, 1], [0, 1], "k--", label="Aleatório")
    plt.xlabel("Taxa de Falso Positivo (FPR)")
    plt.ylabel("Taxa de Verdadeiro Positivo (TPR)")
    plt.title("Curvas ROC (Teste)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "roc_curves_oca_opmd.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Curvas ROC salvas em: {out_path}")


# -------------------------------------------------------------------
# Função Principal (Main)
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--history_path",
        required=True,
        help="Caminho para o arquivo CSV gerado durante o treino (training_history.csv).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="plots",
        help="Pasta onde os gráficos gerados serão salvos.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Pasta raiz do dataset. Se não for passada, o script tentará inferir a partir do caminho do histórico.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Dispositivo usado para a inferência no teste ('cuda' ou 'cpu').",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Tamanho da imagem esperado pelo modelo (deve ser o mesmo do treino).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Tamanho do batch para a avaliação (pode ser maior que no treino pois não consome memória de gradientes).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="convnext_tiny",
        help=(
        "Nome do backbone (timm) usado no modelo. "
        "IMPORTANTE: Deve ser idêntico ao usado no treino (ex: 'mobilenetv3_large_100')."
        ),
    )
    parser.add_argument(
    	"--dropout_p",
    	type=float,
    	default=0.3,
    	help="Taxa de Dropout usada no modelo (deve ser igual ao treino).",
    )
    parser.add_argument(
    	"--model_path",
    	type=str,
    	default=None,
    	help="Caminho específico para o arquivo de pesos (.pt). Se vazio, procura 'best_model.pt' na pasta do dataset.",
    )
    args = parser.parse_args()
    
    print(f"[INFO] Configuração: Backbone='{args.backbone}', Dropout={args.dropout_p}")

    history_path = args.history_path
    out_dir = args.out_dir

    # 1) Passo 1: Gerar gráficos baseados no CSV (Histórico)
    generate_training_curves(history_path, out_dir)

    # 2) Passo 2: Avaliação profunda usando o modelo treinado
    if args.dataset_dir is not None:
        dataset_dir = args.dataset_dir
    else:
        # Tenta adivinhar o diretório do dataset baseado na localização do CSV
        dataset_dir = os.path.dirname(os.path.abspath(history_path))

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("[AVISO] CUDA solicitado mas não disponível. Usando CPU.", file=sys.stderr)
        device = torch.device("cpu")

    print(f"\n[INFO] Iniciando avaliação no TESTE (Device: {device})...")

    # Define onde buscar o arquivo de modelo
    model_path = args.model_path
    if model_path is None:
    	model_path = os.path.join(dataset_dir, "best_model.pt")

    eval_result = evaluate_test_and_collect(
    	model_path=model_path,
    	dataset_dir=dataset_dir,
    	device=device,
    	img_size=args.img_size,
    	batch_size=args.batch_size,
    	backbone=args.backbone,
    	dropout_p=args.dropout_p,
    )

    if eval_result is None:
        print("\n[AVISO] Não foi possível completar a avaliação (modelo não encontrado ou teste vazio).")
        return

    # Desempacota resultados
    y_true = eval_result["y_true"]
    y_pred = eval_result["y_pred"]
    y_proba = eval_result["y_proba"]
    cm = eval_result["cm"]
    f1_per_class = eval_result["f1_per_class"]

    class_names = ["Healthy", "Benign", "OPMD", "OCA"]

    # Gera os gráficos finais
    plot_confusion_matrix(cm, class_names, out_dir)
    plot_f1_per_class(f1_per_class, class_names, out_dir)
    plot_pr_curves(y_true, y_proba, out_dir)
    plot_roc_curves(y_true, y_proba, out_dir)

    print("\n[SUCESSO] Todos os gráficos foram gerados e salvos em:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()