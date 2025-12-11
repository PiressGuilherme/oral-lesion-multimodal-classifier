#!/usr/bin/env python3
"""
Dataset e DataLoaders Multimodais para o projeto de Oncologia Oral.

Este script define a classe de Dataset personalizada e as funções para criação dos DataLoaders.
Ele é responsável por alimentar o modelo com dados multimodais:
1. Imagem da Cavidade Oral (Recorte/ROI).
2. Imagem da Lesão (Recorte/ROI) - se não existir, usa um tensor de zeros.
3. Dados Tabulares (Idade normalizada + One-hot de sexo e hábitos).

Funcionalidades principais:
- Leitura do arquivo 'roi_manifest.csv' (gerado no passo anterior).
- Processamento de imagens (Redimensionamento, Normalização e Data Augmentation no treino).
- Tratamento de dados ausentes (ex: imagens de lesão inexistentes).
- Estratégia de balanceamento: Utiliza 'WeightedRandomSampler' no conjunto de treino
  para lidar com o desequilíbrio entre as classes.

Saída para cada amostra (dicionário):
    - 'oral_image': Tensor [3, H, W]
    - 'lesion_image': Tensor [3, H, W] (Preto/Zeros se has_lesion=0)
    - 'tabular': Tensor unidimensional com as features clínicas.
    - 'label': Inteiro representando a classe (0-3).
    - 'has_lesion': Inteiro (0 ou 1).
    - 'image_name', 'patient_id': Metadados auxiliares.

Uso via linha de comando (apenas para inspeção/teste):
    python dataset_multimodal.py --dataset_dir "/caminho/do/dataset"
"""

import argparse
import os
import sys

from collections import Counter

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# --------------------------------------------------
# Definição das Colunas Tabulares
# --------------------------------------------------
# Lista fixa das colunas do CSV que serão convertidas em tensores
# para alimentar o ramo MLP (Multi-Layer Perceptron) do modelo.
TABULAR_COLUMNS = [
    "age_norm",
    "gender_F",
    "gender_M",
    "smoking_No",
    "smoking_Yes",
    "alcohol_No",
    "alcohol_Yes",
    "betel_No",
    "betel_Yes",
]


class OralLesionMultimodalDataset(Dataset):
    """
    Classe de Dataset PyTorch personalizada para lidar com dados multimodais.
    Carrega imagens e dados tabulares com base no split (train/val/test) especificado.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_dir: str,
        split: str,
        img_size: int = 224,
        train: bool = True,
    ):
        """
        Inicializa o dataset.

        Args:
            df (pd.DataFrame): DataFrame contendo o manifesto (roi_manifest.csv).
            dataset_dir (str): Caminho raiz do dataset.
            split (str): Qual divisão carregar ('train', 'val' ou 'test').
            img_size (int): Tamanho para redimensionamento das imagens (ex: 224x224).
            train (bool): Se True, aplica data augmentation (rotação, flip, etc).
        """
        # --------------------------------------------------
        # Bloco 1: Validação e Filtragem
        # --------------------------------------------------
        # Verifica se as colunas necessárias existem e filtra o DataFrame
        # para manter apenas as linhas correspondentes ao 'split' solicitado.
        if "split" not in df.columns:
            raise KeyError("O DataFrame deve ter uma coluna 'split' (train/val/test).")
        if "oral_roi_path" not in df.columns:
            raise KeyError("O DataFrame deve ter uma coluna 'oral_roi_path'.")

        self.dataset_dir = dataset_dir
        self.split = split.lower()
        self.img_size = img_size

        # Filtra pelo split desejado e garante que o caminho da ROI oral não seja nulo/vazio
        mask = df["split"].astype(str).str.lower().eq(self.split)
        mask &= df["oral_roi_path"].notna()
        mask &= df["oral_roi_path"].astype(str).str.len() > 0
        self.df = df[mask].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(
                f"Dataset para split='{self.split}' ficou vazio. "
                "Verifique roi_manifest.csv e o campo 'split'."
            )

        # Checa se todas as colunas tabulares definidas globalmente existem no CSV
        for col in TABULAR_COLUMNS:
            if col not in self.df.columns:
                raise KeyError(f"Coluna tabular ausente: {col}")

        # --------------------------------------------------
        # Bloco 2: Transformações de Imagem
        # --------------------------------------------------
        # Define as transformações (pré-processamento) para as imagens.
        # - Treino: Inclui aumentação de dados leve (Flip horizontal, rotação leve).
        # - Val/Teste: Apenas redimensionamento e normalização.
        # A normalização usa as médias e desvios padrão do ImageNet.
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def __len__(self):
        """Retorna o número total de amostras no dataset."""
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Recupera uma amostra (imagens + dados tabulares + label) pelo índice.
        """
        row = self.df.iloc[idx]

        # --------------------------------------------------
        # Bloco 3: Carregamento da Imagem da Cavidade Oral
        # --------------------------------------------------
        # Esta imagem é obrigatória. Carrega, converte para RGB e aplica transforms.
        oral_rel = row["oral_roi_path"]
        oral_path = os.path.join(self.dataset_dir, oral_rel)
        if not os.path.isfile(oral_path):
            raise FileNotFoundError(f"Imagem ROI oral não encontrada: {oral_path}")
        oral_img = Image.open(oral_path).convert("RGB")
        oral_tensor = self.transform(oral_img)

        # --------------------------------------------------
        # Bloco 4: Carregamento da Imagem da Lesão
        # --------------------------------------------------
        # Esta imagem é opcional (nem todos os pacientes têm lesão visível/anotada).
        # Se a lesão existir, carrega e transforma.
        # Se NÃO existir, cria um tensor de zeros com o mesmo formato da imagem oral.
        lesion_tensor = None
        has_lesion = int(row.get("has_lesion", 0))
        lesion_rel = row.get("lesion_roi_path", "")

        if has_lesion and isinstance(lesion_rel, str) and len(lesion_rel) > 0:
            lesion_path = os.path.join(self.dataset_dir, lesion_rel)
            if os.path.isfile(lesion_path):
                lesion_img = Image.open(lesion_path).convert("RGB")
                lesion_tensor = self.transform(lesion_img)

        # Fallback: tensor preto (zeros) se não houver lesão
        if lesion_tensor is None:
            lesion_tensor = torch.zeros_like(oral_tensor)

        # --------------------------------------------------
        # Bloco 5: Processamento Tabular e Labels
        # --------------------------------------------------
        # Converte as colunas tabulares selecionadas para um tensor de float.
        # Converte o label da classe para um tensor Long (inteiro).
        tab_values = row[TABULAR_COLUMNS].astype(float).values
        tab_tensor = torch.tensor(tab_values, dtype=torch.float32)

        label = int(row["y"])
        label_tensor = torch.tensor(label, dtype=torch.long)

        has_lesion_tensor = torch.tensor(has_lesion, dtype=torch.float32)

        # Metadados auxiliares (úteis para debug ou log)
        image_name = row.get("Image_Name", "")
        patient_id = row.get("patient_id", "")

        sample = {
            "oral_image": oral_tensor,
            "lesion_image": lesion_tensor,
            "tabular": tab_tensor,
            "label": label_tensor,
            "has_lesion": has_lesion_tensor,
            "image_name": image_name,
            "patient_id": patient_id,
        }
        return sample


def make_class_weights(df: pd.DataFrame, label_col: str = "y"):
    """
    Calcula pesos inversamente proporcionais à frequência de cada classe.
    Útil para lidar com classes desbalanceadas durante o treinamento.
    Retorna um dicionário {classe: peso}.
    """
    counts = df[label_col].value_counts().to_dict()
    total = sum(counts.values())
    # O peso é (Total / Frequência). Classes raras terão pesos maiores.
    class_weights = {cls: total / cnt for cls, cnt in counts.items()}
    return class_weights


def create_dataloaders(
    dataset_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: int = 224,
):
    """
    Função principal para instanciar os DataLoaders de Treino, Validação e Teste.

    Aplica uma estratégia de amostragem ponderada (WeightedRandomSampler) no
    DataLoader de treino para garantir que classes minoritárias sejam vistas
    com mais frequência.

    Retorna:
        train_loader, val_loader, test_loader, class_weights
    """
    roi_manifest_path = os.path.join(dataset_dir, "roi_manifest.csv")
    if not os.path.isfile(roi_manifest_path):
        raise FileNotFoundError(f"roi_manifest.csv não encontrado em {roi_manifest_path}")

    df = pd.read_csv(roi_manifest_path)

    # --------------------------------------------------
    # Bloco 1: Instanciação dos Datasets
    # --------------------------------------------------
    train_dataset = OralLesionMultimodalDataset(
        df, dataset_dir, split="train", img_size=img_size, train=True
    )
    val_dataset = OralLesionMultimodalDataset(
        df, dataset_dir, split="val", img_size=img_size, train=False
    )
    test_dataset = OralLesionMultimodalDataset(
        df, dataset_dir, split="test", img_size=img_size, train=False
    )

    # --------------------------------------------------
    # Bloco 2: Configuração do Sampler (Balanceamento)
    # --------------------------------------------------
    # Calcula pesos para cada amostra do treino baseados na sua classe.
    train_df = train_dataset.df
    class_weights = make_class_weights(train_df, label_col="y")

    # Mapeia o peso da classe para cada linha (amostra) do dataframe
    weights_per_sample = train_df["y"].map(class_weights).values
    weights_tensor = torch.tensor(weights_per_sample, dtype=torch.float32)

    # O Sampler irá escolher amostras baseando-se nesses pesos
    sampler = WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(weights_tensor),
        replacement=True,
    )

    # --------------------------------------------------
    # Bloco 3: Criação dos DataLoaders
    # --------------------------------------------------
    # - pin_memory=True acelera a transferência de dados para a GPU.
    # - No treino, usamos o 'sampler' (então shuffle deve ser False/None no construtor).
    # - Na validação e teste, usamos ordem sequencial (shuffle=False).
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights


# -------------------------------------------------------------------
# Funções para Teste e Inspeção via Linha de Comando
# -------------------------------------------------------------------


def class_dist_from_df(df: pd.DataFrame, label_col: str = "y"):
    """Auxiliar: Retorna dicionário com contagem de classes."""
    return dict(sorted(Counter(df[label_col].tolist()).items()))


def main_cli():
    """
    Função executada quando o script é chamado diretamente.
    Serve para validar se o Dataset e os Loaders estão funcionando corretamente.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Pasta contendo roi_manifest.csv e a pasta roi/.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Tamanho do batch para o teste de carregamento.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Número de workers do DataLoader.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Tamanho (altura=largura) para o resize das imagens.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    roi_manifest_path = os.path.join(dataset_dir, "roi_manifest.csv")
    if not os.path.isfile(roi_manifest_path):
        print(f"[ERRO] roi_manifest.csv não encontrado em {roi_manifest_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(roi_manifest_path)

    # Imprime a distribuição de classes original em cada split
    for split in ("train", "val", "test"):
        sub = df[df["split"].astype(str).str.lower() == split]
        dist = class_dist_from_df(sub, label_col="y") if len(sub) > 0 else {}
        print(f"[INFO] Split '{split}': {len(sub)} amostras, dist. classes = {dist}")

    # Cria DataLoaders e imprime infos de um batch de treino
    print("\n[INFO] Criando DataLoaders para teste...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    print("[INFO] Pesos de classe calculados (para oversampling):")
    print("       (classe: peso)")
    for cls, w in sorted(class_weights.items()):
        print(f"       {cls}: {w:.4f}")

    # Pega um batch real de treino para checar formatos (shapes) dos tensores
    batch = next(iter(train_loader))
    oral = batch["oral_image"]
    lesion = batch["lesion_image"]
    tab = batch["tabular"]
    labels = batch["label"]
    print("\n[INFO] Exemplo de batch de treino carregado com sucesso:")
    print(f"  oral_image shape:   {tuple(oral.shape)}   (B, C, H, W) - Esperado: ({args.batch_size}, 3, {args.img_size}, {args.img_size})")
    print(f"  lesion_image shape: {tuple(lesion.shape)} - (Zeros se não houver lesão)")
    print(f"  tabular shape:      {tuple(tab.shape)}    (B, num_features) - num_features={len(TABULAR_COLUMNS)}")
    print(f"  labels shape:       {tuple(labels.shape)}")

    print("\n[OK] Dataset multimodal e DataLoaders prontos para uso.")


if __name__ == "__main__":
    main_cli()