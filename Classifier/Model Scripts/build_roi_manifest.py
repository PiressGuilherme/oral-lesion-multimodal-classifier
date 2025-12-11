#!/usr/bin/env python3
"""
Gera recortes (crops) de 'Oral Cavity' (Cavidade Oral) e 'Lesion' (Lesão) a partir
do arquivo de anotações COCO (Annotation.json) e do manifesto existente (manifest.csv),
criando um novo manifesto de ROIs (roi_manifest.csv).

Fluxo de processamento para cada linha do manifesto:
1. Lê a imagem original usando o caminho salvo.
2. Consulta o Annotation.json para encontrar as coordenadas (bounding boxes):
   - Bbox da Cavidade Oral (Obrigatório).
   - Bboxes de Lesões (Pode haver zero ou várias).
3. Gera os recortes:
   - 1 recorte da Cavidade Oral (sempre).
   - 1 recorte de Lesão (se houver lesões; caso haja múltiplas, cria um único recorte
     que engloba todas elas).
4. Salva os recortes organizados nas pastas:
   Dataset/
     roi/
       train/
         <Nome_Imagem>_oral.jpg
         <Nome_Imagem>_lesion.jpg (apenas se houver lesão)
       val/
       test/

Saída principal:
  Arquivo: Dataset/roi_manifest.csv
  Colunas adicionadas:
    - oral_roi_path: Caminho relativo para o crop da cavidade oral.
    - lesion_roi_path: Caminho relativo para o crop da lesão (ou vazio).
    - has_lesion: Flag binária (0 ou 1) indicando presença de lesão.

Exemplos de execução:

    # Execução padrão (padding de 5% ao redor da área anotada):
    python build_roi_manifest.py --dataset_dir "/Downloads/Dataset"

    # Execução com padding personalizado (ex: 10%):
    python build_roi_manifest.py --dataset_dir "/Downloads/Dataset" --padding 0.10
"""

import argparse
import os
import sys
import json
import math

import pandas as pd
from PIL import Image


# --------------------------------------------------
# Funções Auxiliares de Leitura e Processamento
# --------------------------------------------------

def load_coco(coco_path):
    """
    Carrega o conteúdo do arquivo JSON de anotações (formato COCO) para a memória.
    Retorna um dicionário Python representando a estrutura do JSON.
    """
    if not os.path.isfile(coco_path):
        raise FileNotFoundError(f"Annotation.json não encontrado em: {coco_path}")
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    return coco


def build_indices_from_coco(coco):
    """
    Cria índices e dicionários de busca rápida a partir dos dados brutos do COCO.
    Isso otimiza o processo de encontrar anotações para cada imagem sem precisar
    iterar pela lista inteira repetidamente.

    Retorna:
    - name_to_catid: Mapeia nome da categoria (str) -> ID da categoria (int).
    - base_to_image_id: Mapeia nome do arquivo sem extensão -> ID da imagem.
    - image_info_by_id: Mapeia ID da imagem -> metadados (largura, altura, nome).
    - anns_by_image_id: Mapeia ID da imagem -> lista de anotações associadas.
    """
    # Mapeamento de categorias
    name_to_catid = {}
    for cat in coco.get("categories", []):
        name = str(cat.get("name", "")).strip().lower()
        cid = cat.get("id")
        if name:
            name_to_catid[name] = cid

    # Mapeamento de imagens
    base_to_image_id = {}
    image_info_by_id = {}
    for img in coco.get("images", []):
        img_id = img["id"]
        file_name = img["file_name"]
        base = os.path.splitext(os.path.basename(file_name))[0]
        base_to_image_id[base] = img_id
        image_info_by_id[img_id] = {
            "file_name": file_name,
            "width": img.get("width", None),
            "height": img.get("height", None),
        }

    # Mapeamento de anotações
    anns_by_image_id = {}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        anns_by_image_id.setdefault(img_id, []).append(ann)

    return name_to_catid, base_to_image_id, image_info_by_id, anns_by_image_id


def union_bboxes(bboxes):
    """
    Calcula a união de múltiplas bounding boxes.
    Entrada: Lista de listas no formato COCO [x, y, w, h].
    Saída: Uma tupla (x_min, y_min, x_max, y_max) que representa o retângulo
    mínimo capaz de conter todas as caixas fornecidas.
    """
    if not bboxes:
        return None
    x_mins = []
    y_mins = []
    x_maxs = []
    y_maxs = []
    for b in bboxes:
        x, y, w, h = b
        x_mins.append(x)
        y_mins.append(y)
        x_maxs.append(x + w)
        y_maxs.append(y + h)
    return min(x_mins), min(y_mins), max(x_maxs), max(y_maxs)


def expand_and_clip_bbox(x_min, y_min, x_max, y_max, img_w, img_h, padding_frac=0.05):
    """
    Expande as coordenadas de uma bounding box adicionando uma margem (padding)
    proporcional ao maior lado da caixa.
    Além disso, garante (clip) que as novas coordenadas não ultrapassem os limites
    físicos da imagem (0, 0, largura, altura).
    """
    bw = x_max - x_min
    bh = y_max - y_min
    if bw <= 0 or bh <= 0:
        # Retorna a imagem inteira se o bbox for inválido/degenerado
        return 0, 0, img_w, img_h

    pad = padding_frac * max(bw, bh)

    x_min_p = max(0, int(math.floor(x_min - pad)))
    y_min_p = max(0, int(math.floor(y_min - pad)))
    x_max_p = min(img_w, int(math.ceil(x_max + pad)))
    y_max_p = min(img_h, int(math.ceil(y_max + pad)))

    # Verificação de segurança para garantir que o bbox resultante seja válido
    if x_max_p <= x_min_p:
        x_min_p, x_max_p = 0, img_w
    if y_max_p <= y_min_p:
        y_min_p, y_max_p = 0, img_h

    return x_min_p, y_min_p, x_max_p, y_max_p


# --------------------------------------------------
# Função Principal (Main)
# --------------------------------------------------

def main():
    # Configuração dos argumentos da linha de comando
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Pasta contendo manifest.csv, Annotation.json e a pasta images/.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Padding fracionário em torno dos bboxes (ex.: 0.05 = 5%).",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    manifest_path = os.path.join(dataset_dir, "manifest.csv")
    coco_path = os.path.join(dataset_dir, "Annotation.json")

    if not os.path.isfile(manifest_path):
        print(f"[ERRO] manifest.csv não encontrado em {manifest_path}", file=sys.stderr)
        sys.exit(1)

    # --------------------------------------------------
    # Bloco 1: Carregamento do Manifesto
    # --------------------------------------------------
    # Carrega o arquivo CSV gerado na etapa anterior e verifica se as colunas
    # essenciais para o processamento estão presentes.
    print(f"[INFO] Carregando manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)

    required_cols = {"Image_Name", "image_path", "split", "y", "patient_id"}
    missing_cols = required_cols - set(manifest.columns)
    if missing_cols:
        raise KeyError(f"Colunas obrigatórias ausentes em manifest.csv: {missing_cols}")

    # --------------------------------------------------
    # Bloco 2: Carregamento e Indexação do COCO
    # --------------------------------------------------
    # Carrega o JSON de anotações e constrói os índices auxiliares para busca rápida.
    # Em seguida, tenta identificar automaticamente os IDs das categorias "Lesion" e "Oral Cavity"
    # baseando-se nos nomes das categorias presentes no JSON.
    print(f"[INFO] Carregando COCO: {coco_path}")
    coco = load_coco(coco_path)
    (
        name_to_catid,
        base_to_image_id,
        image_info_by_id,
        anns_by_image_id,
    ) = build_indices_from_coco(coco)

    # Lógica para descobrir qual ID corresponde a Lesão e qual a Cavidade Oral
    lesion_cat_id = None
    oral_cat_id = None

    for name, cid in name_to_catid.items():
        if "lesion" == name:
            lesion_cat_id = cid
        if name in ("oral cavity", "oral_cavity", "oral-cavity"):
            oral_cat_id = cid

    # Estratégias de fallback (caso os nomes não sejam exatos)
    if lesion_cat_id is None:
        if len(name_to_catid) == 2:
            for name, cid in name_to_catid.items():
                if "oral" not in name:
                    lesion_cat_id = cid
        if lesion_cat_id is None:
            raise ValueError(
                f"Não foi possível identificar category_id para 'Lesion'. Categorias encontradas: {name_to_catid}"
            )

    if oral_cat_id is None:
        for name, cid in name_to_catid.items():
            if "oral" in name:
                oral_cat_id = cid
                break
        if oral_cat_id is None:
            raise ValueError(
                f"Não foi possível identificar category_id para 'Oral Cavity'. Categorias encontradas: {name_to_catid}"
            )

    print(f"[INFO] lesion_cat_id = {lesion_cat_id}, oral_cat_id = {oral_cat_id}")

    # --------------------------------------------------
    # Bloco 3: Preparação do Ambiente de Saída
    # --------------------------------------------------
    # Cria a estrutura de pastas para salvar as imagens recortadas e inicializa
    # as novas colunas no DataFrame do manifesto.
    roi_root = os.path.join(dataset_dir, "roi")
    os.makedirs(roi_root, exist_ok=True)

    manifest["oral_roi_path"] = ""
    manifest["lesion_roi_path"] = ""
    manifest["has_lesion"] = 0

    # Contadores para estatísticas finais
    n_total = 0
    n_missing_coco = 0
    n_missing_oral = 0
    n_with_lesion = 0
    n_missing_lesion = 0

    # --------------------------------------------------
    # Bloco 4: Loop de Processamento das Imagens
    # --------------------------------------------------
    # Itera sobre cada imagem listada no manifesto.
    # Para cada imagem:
    # 1. Carrega a imagem do disco.
    # 2. Busca as anotações correspondentes no COCO.
    # 3. Processa e salva o crop da Cavidade Oral.
    # 4. Processa e salva o crop da Lesão (se houver).
    print("[INFO] Gerando ROIs (Oral Cavity e Lesion) para cada imagem...")
    for idx, row in manifest.iterrows():
        n_total += 1

        image_path = row["image_path"]
        image_name = row["Image_Name"]
        split = str(row["split"]).strip().lower()

        if split not in ("train", "val", "test"):
            split = "unsplit"

        base = os.path.splitext(os.path.basename(image_path))[0]

        # Verifica se a imagem existe no arquivo de anotações
        image_id = base_to_image_id.get(base, None)
        if image_id is None:
            n_missing_coco += 1
            print(
                f"[AVISO] Imagem {base} não encontrada em Annotation.json (COCO). Pulando.",
                file=sys.stderr,
            )
            continue

        # Obtém dimensões da imagem e carrega o arquivo
        img_info = image_info_by_id.get(image_id, {})
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(
                f"[AVISO] Falha ao abrir imagem {image_path}: {e}. Pulando.",
                file=sys.stderr,
            )
            continue

        img_w, img_h = img.size
        if not img_info.get("width") or not img_info.get("height"):
            img_info["width"] = img_w
            img_info["height"] = img_h

        # Filtra anotações por categoria
        anns = anns_by_image_id.get(image_id, [])
        oral_anns = [a for a in anns if a.get("category_id") == oral_cat_id]
        lesion_anns = [a for a in anns if a.get("category_id") == lesion_cat_id]

        # --------------------------------------------------
        # Bloco 4.1: Processamento da Cavidade Oral
        # --------------------------------------------------
        # O recorte da cavidade oral é obrigatório. Se não houver anotação, pula a imagem.
        if not oral_anns:
            n_missing_oral += 1
            print(
                f"[AVISO] Nenhuma anotação de Oral Cavity para imagem {base}. Pulando.",
                file=sys.stderr,
            )
            continue

        oral_bbox = oral_anns[0].get("bbox", None)
        if not oral_bbox:
            n_missing_oral += 1
            print(
                f"[AVISO] bbox de Oral Cavity ausente para imagem {base}. Pulando.",
                file=sys.stderr,
            )
            continue

        # Calcula coordenadas com padding
        x, y, w, h = oral_bbox
        oral_xmin, oral_ymin, oral_xmax, oral_ymax = expand_and_clip_bbox(
            x, y, x + w, y + h, img_w, img_h, padding_frac=args.padding
        )

        # --------------------------------------------------
        # Bloco 4.2: Processamento da Lesão
        # --------------------------------------------------
        # O recorte da lesão é opcional (nem todas imagens têm lesão).
        # Se houver múltiplas lesões, elas são unidas em um único retângulo.
        lesion_roi_exists = False
        lesion_path_rel = ""

        if lesion_anns:
            bboxes = [ann.get("bbox", None) for ann in lesion_anns if ann.get("bbox")]
            if bboxes:
                union = union_bboxes(bboxes)
                if union is not None:
                    lxmin, lymin, lxmax, lymax = expand_and_clip_bbox(
                        union[0],
                        union[1],
                        union[2],
                        union[3],
                        img_w,
                        img_h,
                        padding_frac=args.padding,
                    )
                    lesion_crop = img.crop((lxmin, lymin, lxmax, lymax))

                    split_dir = os.path.join(roi_root, split)
                    os.makedirs(split_dir, exist_ok=True)

                    lesion_filename = f"{base}_lesion.jpg"
                    lesion_full_path = os.path.join(split_dir, lesion_filename)
                    lesion_crop.save(lesion_full_path, format="JPEG")
                    lesion_path_rel = os.path.relpath(lesion_full_path, dataset_dir)
                    lesion_roi_exists = True
        else:
            n_missing_lesion += 1

        # --------------------------------------------------
        # Bloco 4.3: Salvamento da Cavidade Oral e Atualização
        # --------------------------------------------------
        # Salva o crop da cavidade oral e atualiza as informações na linha correspondente
        # do DataFrame.
        split_dir = os.path.join(roi_root, split)
        os.makedirs(split_dir, exist_ok=True)
        oral_filename = f"{base}_oral.jpg"
        oral_full_path = os.path.join(split_dir, oral_filename)
        oral_crop = img.crop((oral_xmin, oral_ymin, oral_xmax, oral_ymax))
        oral_crop.save(oral_full_path, format="JPEG")
        oral_path_rel = os.path.relpath(oral_full_path, dataset_dir)

        manifest.at[idx, "oral_roi_path"] = oral_path_rel
        manifest.at[idx, "lesion_roi_path"] = lesion_path_rel
        manifest.at[idx, "has_lesion"] = 1 if lesion_roi_exists else 0

        if lesion_roi_exists:
            n_with_lesion += 1

    # --------------------------------------------------
    # Bloco 5: Salvamento do Manifesto de ROIs
    # --------------------------------------------------
    # Salva o novo arquivo CSV contendo os caminhos para os recortes gerados
    # e imprime um resumo estatístico da operação.
    roi_manifest_path = os.path.join(dataset_dir, "roi_manifest.csv")
    manifest.to_csv(roi_manifest_path, index=False)

    print("\n[INFO] Finalizado.")
    print(f"  Total de linhas processadas no manifest: {n_total}")
    print(f"  Imagens sem entrada em COCO (puladas): {n_missing_coco}")
    print(f"  Imagens sem Oral Cavity válida (puladas): {n_missing_oral}")
    print(f"  Imagens com pelo menos uma Lesion: {n_with_lesion}")
    print(f"  Imagens sem Lesion anotada: {n_missing_lesion}")
    print(f"[OK] roi_manifest salvo em: {roi_manifest_path}")
    print("[OK] ROIs salvas em subpastas de:", roi_root)


if __name__ == "__main__":
    main()