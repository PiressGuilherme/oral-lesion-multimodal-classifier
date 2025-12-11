# Classificação Multimodal de Lesões Orais

Este repositório contém a implementação oficial do projeto **"Classificação Multimodal de Lesões Orais: Uma Abordagem Híbrida Integrando Redes Neurais Convolucionais e Dados Clínicos Estruturados"**.

projeto propõe uma arquitetura de Deep Learning híbrida que funde características visuais (extraídas via CNNs) com dados clínicos tabulares para a classificação multiclasse de lesões orais, focando na distinção entre tecidos Saudáveis, Benignos, Desordens Potencialmente Malignas (OPMD) e Carcinoma Oral (OCA).

## 📋 Sobre o Projeto

A classificação automatizada de lesões orais enfrenta desafios como a alta similaridade visual entre classes e o severo desbalanceamento de dados. Este framework aborda esses problemas através de:

1.  **Fusão Multimodal:** Processamento simultâneo do contexto visual da cavidade oral, recorte focado na lesão (ROI) e metadados do paciente (idade, sexo, hábitos).
2.  **Prevenção de Vazamento de Dados (Data Leakage):** Estratégias rigorosas de particionamento baseadas no ID do paciente, garantindo que imagens do mesmo indivíduo não apareçam em conjuntos de treino e teste simultaneamente.
3.  **Arquitetura Modular:** Uso de backbones modernos (como ConvNeXt Tiny) via biblioteca `timm`, permitindo fácil substituição dos extratores de características.

## 🚀 Arquitetura do Modelo

O modelo utiliza uma estratégia de **Fusão Tardia (Late Fusion)** com três ramos de processamento:

* **Ramo Visual Global:** CNN processando a imagem da cavidade oral completa (com padding de contexto).
* **Ramo Visual ROI:** CNN processando o recorte focado na lesão (gerado via anotações COCO).
    * *Nota:* Inclui um mecanismo de "gating" para zerar features em pacientes saudáveis sem lesão.
* **Ramo Tabular:** MLP (Multilayer Perceptron) processando dados clínicos normalizados e codificados (One-Hot).

Os vetores de características são concatenados e processados por um MLP de fusão antes da classificação final.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10 
* **Framework:** PyTorch 2.0 (CUDA 12.6) 
* **Visão Computacional:** `timm` (PyTorch Image Models), `torchvision`, `PIL` 
* **Manipulação de Dados:** Pandas, Numpy
* **Métricas:** Scikit-learn

## 📂 Estrutura e Scripts Principais

O pipeline de engenharia de dados é automatizado pelos seguintes scripts identificados no estudo:

* `build_manifest_and_split.py`: Realiza o particionamento dos dados. Extrai IDs únicos de pacientes e gera os splits de Treino/Validação/Teste garantindo isolamento estrito de pacientes e balanceamento de classes críticas (como OCA).
* `build_roi_manifest.py`: Processa arquivos de anotação COCO (`Annotation.json`) para gerar recortes (crops) dinâmicos das lesões e do contexto oral.
* `OralLesionMultimodalDataset`: Classe personalizada de Dataset que gerencia o carregamento das imagens e dados tabulares.

## ⚙️ Configuração e Treinamento

### Pré-requisitos
Certifique-se de ter as bibliotecas instaladas (exemplo genérico baseado no texto):
```bash
pip install torch torchvision timm pandas scikit-learn numpy
