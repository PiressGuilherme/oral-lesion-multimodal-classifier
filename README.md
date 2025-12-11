# Classificação Multimodal de Lesões Orais

Este repositório contém a implementação oficial do projeto **"Classificação Multimodal de Lesões Orais: Uma Abordagem Híbrida Integrando Redes Neurais Convolucionais e Dados Clínicos Estruturados"**.

[cite_start]O projeto propõe uma arquitetura de Deep Learning híbrida que funde características visuais (extraídas via CNNs) com dados clínicos tabulares para a classificação multiclasse de lesões orais, focando na distinção entre tecidos Saudáveis, Benignos, Desordens Potencialmente Malignas (OPMD) e Carcinoma Oral (OCA)[cite: 1011, 1020].

## 📋 Sobre o Projeto

[cite_start]A classificação automatizada de lesões orais enfrenta desafios como a alta similaridade visual entre classes e o severo desbalanceamento de dados[cite: 1014]. Este framework aborda esses problemas através de:

1.  [cite_start]**Fusão Multimodal:** Processamento simultâneo do contexto visual da cavidade oral, recorte focado na lesão (ROI) e metadados do paciente (idade, sexo, hábitos)[cite: 1016].
2.  [cite_start]**Prevenção de Vazamento de Dados (Data Leakage):** Estratégias rigorosas de particionamento baseadas no ID do paciente, garantindo que imagens do mesmo indivíduo não apareçam em conjuntos de treino e teste simultaneamente[cite: 1017, 1045].
3.  [cite_start]**Arquitetura Modular:** Uso de backbones modernos (como ConvNeXt Tiny) via biblioteca `timm`, permitindo fácil substituição dos extratores de características[cite: 1040, 1049].

## 🚀 Arquitetura do Modelo

[cite_start]O modelo utiliza uma estratégia de **Fusão Tardia (Late Fusion)** com três ramos de processamento[cite: 1035, 1135]:

* **Ramo Visual Global:** CNN processando a imagem da cavidade oral completa (com padding de contexto).
* **Ramo Visual ROI:** CNN processando o recorte focado na lesão (gerado via anotações COCO).
    * [cite_start]*Nota:* Inclui um mecanismo de "gating" para zerar features em pacientes saudáveis sem lesão[cite: 1163].
* [cite_start]**Ramo Tabular:** MLP (Multilayer Perceptron) processando dados clínicos normalizados e codificados (One-Hot)[cite: 1165].

Os vetores de características são concatenados e processados por um MLP de fusão antes da classificação final.

## 🛠️ Tecnologias Utilizadas

* [cite_start]**Linguagem:** Python 3.10 [cite: 1189]
* [cite_start]**Framework:** PyTorch 2.0 (CUDA 12.6) [cite: 1189, 906]
* [cite_start]**Visão Computacional:** `timm` (PyTorch Image Models), `torchvision`, `PIL` [cite: 1016, 1055]
* [cite_start]**Manipulação de Dados:** Pandas, Numpy [cite: 1054]
* [cite_start]**Métricas:** Scikit-learn [cite: 1056]

## 📂 Estrutura e Scripts Principais

O pipeline de engenharia de dados é automatizado pelos seguintes scripts identificados no estudo:

* `build_manifest_and_split.py`: Realiza o particionamento dos dados. [cite_start]Extrai IDs únicos de pacientes e gera os splits de Treino/Validação/Teste garantindo isolamento estrito de pacientes e balanceamento de classes críticas (como OCA)[cite: 1083, 1091].
* [cite_start]`build_roi_manifest.py`: Processa arquivos de anotação COCO (`Annotation.json`) para gerar recortes (crops) dinâmicos das lesões e do contexto oral[cite: 1100, 1101].
* [cite_start]`OralLesionMultimodalDataset`: Classe personalizada de Dataset que gerencia o carregamento das imagens e dados tabulares[cite: 1124].

## ⚙️ Configuração e Treinamento

### Pré-requisitos
Certifique-se de ter as bibliotecas instaladas (exemplo genérico baseado no texto):
```bash
pip install torch torchvision timm pandas scikit-learn numpy
