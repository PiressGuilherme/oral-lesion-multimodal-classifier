#!/usr/bin/env python3
"""
python build_manifest_and_split.py --dataset_dir "$(pwd)" --bias_train_to_many_images

Script para construir um MANIFESTO (csv) único para o conjunto de dados de Oncologia Oral
e realizar a divisão (split) por paciente.

Objetivos:
- Manifesto: Criar uma tabela com uma linha por imagem, vinculando caminho da imagem, ID do paciente,
  rótulo (label) e metadados.
- Divisão por paciente: Separar em treino/validação/teste (train/val/test) garantindo que todas as
  imagens de um mesmo paciente fiquem no mesmo grupo (sem vazamento de dados/data leakage).
- Opcional: Favorecer pacientes com mais imagens no conjunto de treino.
- Garantia: Assegura um número mínimo de pacientes com câncer (OCA) em cada divisão.

Estrutura esperada do conjunto de dados (dataset_dir):
    Dataset/
      images/               # todas as imagens (ex: C-3-7-3.jpg, ...)
      Imagewise_Data.csv    # ou .xlsx
      Patientwise_Data.csv  # ou .xlsx
      Annotation.json       # não utilizado nesta etapa

Saídas (gravadas no dataset_dir):
    manifest.csv
    train_manifest.csv
    val_manifest.csv
    test_manifest.csv
    by_patient.csv
    split_stats.txt

Exemplos de execução:

    # Caso típico:
    python build_manifest_and_split.py --dataset_dir "/Downloads/Dataset"

    # Favorecer pacientes com muitas imagens no treino:
    python build_manifest_and_split.py --dataset_dir "/Downloads/Dataset" --bias_train_to_many_images

    # Com proporções personalizadas (ex.: 70% treino, 10% validação, 20% teste):
    python build_manifest_and_split.py --dataset_dir "/Downloads/Dataset" --train_ratio 0.7 --val_ratio 0.1 --test_ratio 0.2
"""

import argparse
import os
import sys
import glob
import math
import random
from collections import Counter

import pandas as pd


# --------------------------------------------------
# Funções Auxiliares (Helpers)
# --------------------------------------------------
# Este bloco contém funções utilitárias para leitura de arquivos, manipulação de strings,
# normalização de colunas e geração de estatísticas básicas.


def read_csv_or_xlsx(path_csv, path_xlsx=None):
    """
    Tenta ler um arquivo CSV (testando várias codificações) ou um arquivo Excel (XLSX).
    Retorna um DataFrame do pandas ou levanta um erro se nenhum arquivo for encontrado.
    """
    path_csv = path_csv if path_csv and os.path.exists(path_csv) else None
    path_xlsx = path_xlsx if path_xlsx and os.path.exists(path_xlsx) else None

    last_err = None
    if path_csv:
        for enc in ("utf-8", "latin1", "utf-16"):
            try:
                return pd.read_csv(path_csv, encoding=enc)
            except Exception as e:
                last_err = e
    if path_xlsx:
        try:
            return pd.read_excel(path_xlsx)
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    raise FileNotFoundError("Nenhum arquivo CSV ou XLSX foi encontrado nos caminhos especificados.")


def patient_id_from_name(name):
    """
    Extrai o ID do paciente a partir do nome do arquivo.
    A lógica assume que o ID são os dois primeiros tokens separados por hífen.
    Exemplo: 'C-41-7-2.jpg' -> Retorna 'C-41'.
    """
    base = os.path.splitext(str(name))[0]
    parts = base.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return base


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza os nomes das colunas do DataFrame: remove espaços nas extremidades
    e substitui espaços internos por underscores ('_').
    """
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def build_name_to_path_index(images_dir):
    """
    Cria um dicionário mapeando o nome base da imagem (sem extensão) para seu caminho completo no disco.
    Varre o diretório em busca de extensões comuns de imagem (.jpg, .png, etc.).
    """
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    mapping = {}
    for ext in exts:
        for p in glob.glob(os.path.join(images_dir, f"*{ext}")):
            base = os.path.splitext(os.path.basename(p))[0]
            mapping[base] = p
    return mapping


def class_distribution(df, label_col="y"):
    """
    Calcula a distribuição das classes em um DataFrame e retorna um dicionário
    ordenado no formato {classe: quantidade}.
    """
    return dict(sorted(Counter(df[label_col].tolist()).items()))


def write_stats(path, **kwargs):
    """
    Escreve estatísticas em um arquivo de texto simples.
    Recebe argumentos nomeados e grava no formato 'chave: valor'.
    """
    with open(path, "w", encoding="utf-8") as f:
        for k, v in kwargs.items():
            f.write(f"{k}: {v}\n")


# --------------------------------------------------
# Função Principal (Main)
# --------------------------------------------------

def main():
    # Configuração dos argumentos da linha de comando.
    # Define diretório do dataset, proporções de divisão e semente aleatória.
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument(
        "--dataset_dir",
        required=True,
        help="Pasta contendo a subpasta images/ e os arquivos CSV/XLSX/JSON.",
    )
    ap.add_argument(
        "--bias_train_to_many_images",
        action="store_true",
        help="Se ativado, prefere colocar pacientes com mais imagens no conjunto de treino.",
    )
    ap.add_argument("--train_ratio", type=float, default=0.6)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória (seed) usada para divisão dos pacientes.",
    )
    args = ap.parse_args()

    random.seed(args.seed)

    # Verificação de sanidade das proporções: a soma deve ser próxima de 1.0.
    # Caso contrário, emite um aviso.
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-3):
        print(
            f"[AVISO] train_ratio + val_ratio + test_ratio = {ratio_sum:.3f}, não é exatamente 1.0.",
            file=sys.stderr,
        )

    # Verifica se o diretório de imagens existe.
    images_dir = os.path.join(args.dataset_dir, "images")
    if not os.path.isdir(images_dir):
        print(f"[ERRO] Diretório de imagens não encontrado: {images_dir}", file=sys.stderr)
        sys.exit(1)

    # --------------------------------------------------
    # Bloco 1: Carregamento das Tabelas
    # --------------------------------------------------
    # Lê os arquivos de dados por imagem (Imagewise) e por paciente (Patientwise).
    # Normaliza os nomes das colunas e renomeia colunas essenciais para um padrão
    # (Image_Name, Category, Patient_ID).
    img = read_csv_or_xlsx(
        os.path.join(args.dataset_dir, "Imagewise_Data.csv"),
        os.path.join(args.dataset_dir, "Imagewise_Data.xlsx"),
    )
    pt = read_csv_or_xlsx(os.path.join(args.dataset_dir, "Patientwise_Data.csv"))
    img = normalize_column_names(img)
    pt = normalize_column_names(pt)

    if "Image_Name" not in img.columns:
        for c in img.columns:
            if c.lower().replace(" ", "_") == "image_name":
                img = img.rename(columns={c: "Image_Name"})
                break
    if "Image_Name" not in img.columns:
        raise KeyError("Coluna 'Image_Name' não encontrada em Imagewise_Data.")

    if "Category" not in img.columns:
        raise KeyError("Coluna 'Category' não encontrada em Imagewise_Data.")

    rename_map = {}
    for c in pt.columns:
        if c.lower().startswith("chewing_betel_quid"):
            rename_map[c] = "Chewing_Betel_Quid"
    if rename_map:
        pt = pt.rename(columns=rename_map)

    # --------------------------------------------------
    # Bloco 2: Identificação do Paciente na Tabela de Imagens
    # --------------------------------------------------
    # Cria uma nova coluna 'patient_id' na tabela de imagens extraindo a informação
    # diretamente do nome do arquivo da imagem.
    img["patient_id"] = img["Image_Name"].apply(patient_id_from_name)

    # --------------------------------------------------
    # Bloco 3: Junção dos Metadados
    # --------------------------------------------------
    # Prepara a tabela de pacientes para o join, garantindo a coluna 'Patient_ID'.
    # Realiza o merge (left join) para que cada imagem tenha os dados clínicos do seu paciente.
    if "Patient_ID" not in pt.columns:
        for c in pt.columns:
            if c.lower().replace(" ", "_") == "patient_id":
                pt = pt.rename(columns={c: "Patient_ID"})
                break
    if "Patient_ID" not in pt.columns:
        raise KeyError("Coluna 'Patient_ID' não encontrada em Patientwise_Data.")

    pt = pt.set_index("Patient_ID")
    df = img.join(pt, on="patient_id", how="left")

    # --------------------------------------------------
    # Bloco 4: Mapeamento de Classes
    # --------------------------------------------------
    # Converte as categorias textuais (Healthy, Benign, OPMD, OCA) para IDs numéricos (0, 1, 2, 3).
    # Verifica se existem categorias desconhecidas.
    cls2id = {"Healthy": 0, "Benign": 1, "OPMD": 2, "OCA": 3}
    df["y"] = df["Category"].map(cls2id)
    if df["y"].isna().any():
        missing = df[df["y"].isna()]["Category"].unique().tolist()
        raise ValueError(f"Categorias desconhecidas encontradas: {missing}")

    # --------------------------------------------------
    # Bloco 5: Engenharia de Atributos (Feature Engineering)
    # --------------------------------------------------
    # 1. Normalização da idade (Z-score).
    # 2. Codificação One-Hot para Gênero.
    # 3. Binarização (Yes/No -> One-Hot) para Fumo, Álcool e Betel.
    if "Age" in df.columns and df["Age"].notna().any():
        df["age_norm"] = (df["Age"] - df["Age"].mean()) / df["Age"].std(ddof=0)
    else:
        df["age_norm"] = 0.0

    gender = (
        df["Gender"]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"MALE": "M", "FEMALE": "F"})
    )
    df = pd.concat(
        [df, pd.get_dummies(gender, prefix="gender", dtype=int)],
        axis=1,
    )
    for col in ("gender_F", "gender_M"):
        if col not in df.columns:
            df[col] = 0

    def binarize_yes_no(series: pd.Series, prefix: str) -> pd.DataFrame:
        s = series.astype(str).str.strip().str.title().replace({"Y": "Yes", "N": "No"})
        d = pd.get_dummies(s, prefix=prefix, dtype=int)
        for need in (f"{prefix}_No", f"{prefix}_Yes"):
            if need not in d.columns:
                d[need] = 0
        return d

    df = pd.concat(
        [
            df,
            binarize_yes_no(df["Smoking"], "smoking"),
            binarize_yes_no(df["Alcohol"], "alcohol"),
            binarize_yes_no(df["Chewing_Betel_Quid"], "betel"),
        ],
        axis=1,
    )

    # --------------------------------------------------
    # Bloco 6: Resolução dos Caminhos das Imagens
    # --------------------------------------------------
    # Mapeia cada entrada do CSV para o arquivo físico no disco.
    # Lança um erro se alguma imagem listada no CSV não for encontrada na pasta.
    name2path = build_name_to_path_index(images_dir)
    df["image_base"] = df["Image_Name"].apply(
        lambda s: os.path.splitext(str(s))[0]
    )
    df["image_path"] = df["image_base"].map(name2path)
    if df["image_path"].isna().any():
        missing = df[df["image_path"].isna()]["Image_Name"].tolist()[:20]
        raise FileNotFoundError(
            f"Algumas imagens não foram encontradas em {images_dir}. Primeiros exemplos: {missing}"
        )

    # --------------------------------------------------
    # Bloco 7: Seleção de Colunas do Manifesto
    # --------------------------------------------------
    # Cria um DataFrame limpo contendo apenas as colunas relevantes para o treinamento
    # e análise posterior (metadados brutos e processados).
    keep_cols = [
        "Image_Name",
        "image_path",
        "patient_id",
        "Category",
        "y",
        "Age",
        "Gender",
        "Smoking",
        "Alcohol",
        "Chewing_Betel_Quid",
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
    manifest = df[keep_cols].copy()

    # --------------------------------------------------
    # Bloco 8: Resumo por Paciente
    # --------------------------------------------------
    # Agrupa os dados por paciente para contar quantas imagens cada um possui.
    # Isso é crucial para a estratégia de divisão (split) baseada em pacientes.
    by_patient = (
        manifest.groupby(["patient_id", "Category"])["Image_Name"]
        .count()
        .unstack(fill_value=0)
    )
    by_patient["Total_Images"] = by_patient.sum(axis=1)
    by_patient = by_patient.reset_index()

    patient_images = by_patient.set_index("patient_id")["Total_Images"]

    # --------------------------------------------------
    # Bloco 9: Divisão dos Pacientes (Train/Val/Test)
    # --------------------------------------------------
    # Realiza a separação dos dados respeitando a integridade do paciente (todas as imagens
    # de um paciente vão para o mesmo conjunto).
    #
    # Lógica:
    # 1. Calcula o alvo de imagens para treino.
    # 2. Ordena pacientes por quantidade de imagens.
    # 3. Se 'bias_train_to_many_images' for True, preenche o treino com pacientes que têm mais imagens.
    #    Caso contrário, embaralha e preenche aleatoriamente.
    # 4. O restante é dividido entre Validação e Teste conforme as proporções.
    total_images = int(by_patient["Total_Images"].sum())
    target_train_images = total_images * args.train_ratio

    patients_sorted = (
        by_patient.sort_values("Total_Images", ascending=False)["patient_id"].tolist()
    )

    if args.bias_train_to_many_images:
        train_patients = []
        cum_train = 0
        for pid in patients_sorted:
            n = int(patient_images.loc[pid])
            if cum_train < target_train_images:
                train_patients.append(pid)
                cum_train += n
            else:
                break
        remaining = [p for p in patients_sorted if p not in set(train_patients)]
    else:
        pids = patients_sorted[:]
        random.shuffle(pids)
        train_patients = []
        cum_train = 0
        for pid in pids:
            n = int(patient_images.loc[pid])
            if cum_train + n <= target_train_images or len(train_patients) < 1:
                train_patients.append(pid)
                cum_train += n
        remaining = [p for p in pids if p not in set(train_patients)]

    if not remaining:
        print("[ERRO] Nenhum paciente restou para as divisões de validação/teste.", file=sys.stderr)
        sys.exit(1)

    remaining_shuffled = remaining[:]
    random.shuffle(remaining_shuffled)
    remaining_images = int(patient_images.loc[remaining_shuffled].sum())

    if args.val_ratio + args.test_ratio <= 0:
        val_fraction = 0.5
    else:
        val_fraction = args.val_ratio / (args.val_ratio + args.test_ratio)

    target_val_images = remaining_images * val_fraction

    val_patients_list = []
    cum_val = 0
    for pid in remaining_shuffled:
        n = int(patient_images.loc[pid])
        if (
            cum_val < target_val_images
            and len(remaining_shuffled) - len(val_patients_list) > 1
        ):
            val_patients_list.append(pid)
            cum_val += n

    if not val_patients_list:
        val_patients_list = [remaining_shuffled[0]]

    test_patients_list = [p for p in remaining_shuffled if p not in set(val_patients_list)]
    if not test_patients_list:
        moved = val_patients_list.pop()
        test_patients_list = [moved]

    train_patients = set(train_patients)
    val_patients = set(val_patients_list)
    test_patients = set(test_patients_list)

    # --------------------------------------------------
    # Bloco 9.1: Cobertura de Classes
    # --------------------------------------------------
    # Verifica se os conjuntos de Validação e Teste possuem pelo menos um exemplo de cada classe.
    # Se faltar alguma classe, tenta mover um paciente do outro conjunto (Teste <-> Validação)
    # que possua a classe faltante.
    def present_classes(pids):
        subset = manifest[manifest["patient_id"].isin(pids)]
        return set(subset["y"].unique().tolist())

    all_classes = set(range(4))
    for target_set_name, pset, otherset in [
        ("val", val_patients, test_patients),
        ("test", test_patients, val_patients),
    ]:
        missing = all_classes - present_classes(pset)
        if missing:
            for cls in list(missing):
                cand = None
                for pid in list(otherset):
                    ys = set(
                        manifest[manifest["patient_id"] == pid]["y"]
                        .unique()
                        .tolist()
                    )
                    if cls in ys:
                        cand = pid
                        break
                if cand is not None:
                    otherset.remove(cand)
                    pset.add(cand)

    # --------------------------------------------------
    # Bloco 9.2: Garantia de Mínimo de Pacientes OCA por Divisão
    # --------------------------------------------------
    # Define um número mínimo de pacientes da classe OCA para cada conjunto (treino, validação, teste).
    # O algoritmo tenta mover pacientes entre os conjuntos para satisfazer essa restrição,
    # iterando até encontrar uma distribuição válida ou atingir o limite de tentativas.
    MIN_OCA_PATIENTS_BY_SPLIT = {"train": 1, "val": 1, "test": 1}

    if "OCA" in by_patient.columns:
        oca_patients = set(
            by_patient.loc[by_patient["OCA"] > 0, "patient_id"].tolist()
        )
    else:
        oca_patients = set()

    split2pids = {
        "train": set(train_patients),
        "val": set(val_patients),
        "test": set(test_patients),
    }

    if oca_patients and any(v > 0 for v in MIN_OCA_PATIENTS_BY_SPLIT.values()):
        max_iters = 100
        for _ in range(max_iters):
            deficits = {}
            for split_name, pids in split2pids.items():
                want = MIN_OCA_PATIENTS_BY_SPLIT.get(split_name, 0)
                if want <= 0:
                    continue
                have = len(oca_patients & pids)
                if have < want:
                    deficits[split_name] = want - have

            if not deficits:
                break

            surplus = {}
            for split_name, pids in split2pids.items():
                want = MIN_OCA_PATIENTS_BY_SPLIT.get(split_name, 0)
                have = len(oca_patients & pids)
                extra = have - want
                if extra > 0:
                    surplus[split_name] = extra

            if not surplus:
                print(
                    "[AVISO] Não foi possível satisfazer totalmente as restrições de mínimo de pacientes OCA.",
                    file=sys.stderr,
                )
                break

            needy_split = max(deficits, key=deficits.get)
            donor_split = max(surplus, key=surplus.get)

            candidate = None
            for pid in split2pids[donor_split]:
                if pid in oca_patients:
                    candidate = pid
                    break

            if candidate is None:
                del surplus[donor_split]
                if not surplus:
                    print(
                        "[AVISO] Não foi possível mover pacientes OCA suficientes para atingir os mínimos.",
                        file=sys.stderr,
                    )
                    break
                continue

            split2pids[donor_split].remove(candidate)
            split2pids[needy_split].add(candidate)

        train_patients = split2pids["train"]
        val_patients = split2pids["val"]
        test_patients = split2pids["test"]

    # --------------------------------------------------
    # Bloco 10: Geração e Salvamento dos Arquivos
    # --------------------------------------------------
    # Atribui o rótulo do split (train/val/test) para cada paciente no manifesto.
    # Salva o manifesto completo e os manifestos individuais para cada split.
    # Salva também a tabela de resumo por paciente.
    pid2split = {}
    for p in train_patients:
        pid2split[p] = "train"
    for p in val_patients:
        pid2split[p] = "val"
    for p in test_patients:
        pid2split[p] = "test"

    manifest["split"] = manifest["patient_id"].map(pid2split)

    out_manifest = os.path.join(args.dataset_dir, "manifest.csv")
    manifest.to_csv(out_manifest, index=False)

    manifest[manifest["split"] == "train"].to_csv(
        os.path.join(args.dataset_dir, "train_manifest.csv"), index=False
    )
    manifest[manifest["split"] == "val"].to_csv(
        os.path.join(args.dataset_dir, "val_manifest.csv"), index=False
    )
    manifest[manifest["split"] == "test"].to_csv(
        os.path.join(args.dataset_dir, "test_manifest.csv"), index=False
    )

    by_patient.to_csv(os.path.join(args.dataset_dir, "by_patient.csv"), index=False)

    # --------------------------------------------------
    # Bloco 11: Cálculo e Gravação de Estatísticas
    # --------------------------------------------------
    # Calcula métricas finais sobre a divisão realizada (contagem de imagens, pacientes e
    # distribuição de classes) e grava em um arquivo de texto (split_stats.txt).
    def count_oca_patients_in(pids):
        if "OCA" not in by_patient.columns:
            return 0
        subset = by_patient[by_patient["patient_id"].isin(pids)]
        return int((subset["OCA"] > 0).sum())

    stats_path = os.path.join(args.dataset_dir, "split_stats.txt")
    stats = {
        "total_images": total_images,
        "train_images": int(
            manifest[manifest["split"] == "train"].shape[0]
        ),
        "val_images": int(
            manifest[manifest["split"] == "val"].shape[0]
        ),
        "test_images": int(
            manifest[manifest["split"] == "test"].shape[0]
        ),
        "train_patients": len(set(train_patients)),
        "val_patients": len(set(val_patients)),
        "test_patients": len(set(test_patients)),
        "train_class_dist": class_distribution(
            manifest[manifest["split"] == "train"]
        ),
        "val_class_dist": class_distribution(
            manifest[manifest["split"] == "val"]
        ),
        "test_class_dist": class_distribution(
            manifest[manifest["split"] == "test"]
        ),
        "train_oca_patients": count_oca_patients_in(train_patients),
        "val_oca_patients": count_oca_patients_in(val_patients),
        "test_oca_patients": count_oca_patients_in(test_patients),
    }
    write_stats(stats_path, **stats)

    print(f"[OK] Gravado: {out_manifest}")
    print(
        f"[OK] Manifestos de Train/Val/Test + by_patient.csv + split_stats.txt salvos em: {args.dataset_dir}"
    )
    print("Mapeamento de classes:", {"Healthy": 0, "Benign": 1, "OPMD": 2, "OCA": 3})
    print(
        "Colunas One-hot:",
        [
            "age_norm",
            "gender_F",
            "gender_M",
            "smoking_No",
            "smoking_Yes",
            "alcohol_No",
            "alcohol_Yes",
            "betel_No",
            "betel_Yes",
        ],
    )


if __name__ == "__main__":
    main()