#!/usr/bin/env python3
"""
Modelos Multimodais para o Projeto de Detecção de Lesões Orais.

Este script define a arquitetura da rede neural que combina processamento de imagens
(Visão Computacional) e dados clínicos (Tabulares).

Arquitetura:
    1. Branch Visual (Oral): Extrai características da foto da cavidade oral.
    2. Branch Visual (Lesão): Extrai características do recorte da lesão (se existir).
    3. Branch Tabular: Processa dados como idade, sexo, fumo, álcool via MLP.
    4. Fusão: Concatena os vetores de características das três fontes.
    5. Classificador: MLP final que gera os logits para as 4 classes.

Backbone de Visão (via biblioteca `timm`):
    - O modelo é agnóstico ao backbone. O padrão é 'convnext_tiny'.
    - Pode ser trocado por 'mobilenetv3_large_100', 'efficientnet_b2', 'resnet50', etc.

Entradas do Modelo (Forward):
    - oral_image:   Tensor [Batch, 3, H, W]
    - lesion_image: Tensor [Batch, 3, H, W] (pode ser zeros se não houver lesão)
    - tabular:      Tensor [Batch, num_tab_features]
    - has_lesion:   Tensor [Batch] contendo 0 ou 1 (usado para mascarar a branch da lesão)

Estratégias de Fine-Tuning:
    - freeze_backbone(): Congela os pesos da parte visual (feature extractor).
    - unfreeze_backbone(): Descongela tudo para treino completo.
    - unfreeze_backbone_last_fraction(fraction): Descongela apenas as últimas camadas
      do backbone (ex: 30% finais), útil para ajustar representações de alto nível
      sem destruir detectores de bordas/texturas aprendidos no ImageNet.
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError(
        "Este modelo requer o pacote 'timm'. "
        "Instale com: pip install timm"
    ) from e


class _BackboneWrapper(nn.Module):
    """
    Classe utilitária (Wrapper) para encapsular um modelo da biblioteca `timm`.

    Funcionalidades:
    1. Instancia o modelo pré-treinado.
    2. Remove a camada de classificação original (num_classes=0).
    3. Aplica Global Average Pooling (global_pool='avg') para obter um vetor de características 1D.
    4. Infere automaticamente o tamanho do vetor de saída (embedding) rodando um dado falso.
    """

    def __init__(self, backbone_name: str = "convnext_tiny", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name

        # Cria o modelo sem a camada final (FC) e já com pooling
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,      # Remove cabeçalho de classificação
            global_pool="avg",  # Retorna vetor (B, Features) em vez de mapa (B, F, H, W)
        )

        # Descobre o tamanho do embedding (ex: 768, 1280, etc) dinamicamente
        self.out_dim = self._infer_out_dim()

    def _infer_out_dim(self, img_size: int = 224) -> int:
        """
        Executa uma passagem (forward) com uma imagem falsa de zeros para descobrir
        o tamanho real do vetor de saída do backbone escolhido.
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, img_size, img_size)
            feat = self.model(x)
        if feat.ndim != 2:
            raise ValueError(
                f"Backbone '{self.backbone_name}' retornou tensor com formato inesperado: {feat.shape}. "
                "Esperado: (Batch, Features)."
            )
        return int(feat.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiModalMobileNetV3Large(nn.Module):
    """
    Modelo Multimodal Principal.

    Nota sobre o nome: O nome da classe é mantido como 'MultiModalMobileNetV3Large' por
    motivos de compatibilidade com códigos anteriores, mas internamente ela é genérica
    e aceita qualquer backbone suportado pelo `timm` (ex: ConvNeXt, ResNet, EfficientNet).

    Componentes:
        - 2 Backbones CNN (compartilhados ou independentes, aqui instanciados separadamente).
        - 1 MLP para dados tabulares.
        - 1 MLP de Fusão e Classificação.
    """

    def __init__(
        self,
        num_tab_features: int,
        num_classes: int,
        dropout_p: float = 0.3,
        tab_hidden_dim: int = 64,
        fusion_hidden_dim: int = 256,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
    ):
        """
        Args:
            num_tab_features (int): Quantidade de colunas de dados clínicos (ex: 9).
            num_classes (int): Número de classes de saída (ex: 4).
            dropout_p (float): Probabilidade de Dropout na camada de fusão.
            tab_hidden_dim (int): Tamanho da camada oculta para dados tabulares.
            fusion_hidden_dim (int): Tamanho da camada oculta após a fusão.
            backbone_name (str): Nome do modelo `timm` a ser baixado.
            pretrained (bool): Se True, usa pesos da ImageNet.
        """
        super().__init__()

        self.num_tab_features = num_tab_features
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.tab_hidden_dim = tab_hidden_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.backbone_name = backbone_name
        self.pretrained = pretrained

        # --------------------------------------------------
        # Backbones de Visão
        # --------------------------------------------------
        # Instancia dois extratores de características independentes:
        # um para a foto geral da boca e outro focado na lesão.
        self.backbone_oral = _BackboneWrapper(backbone_name=backbone_name, pretrained=pretrained)
        self.backbone_lesion = _BackboneWrapper(backbone_name=backbone_name, pretrained=pretrained)

        backbone_dim = self.backbone_oral.out_dim

        # --------------------------------------------------
        # Branch Tabular (MLP)
        # --------------------------------------------------
        # Processa o vetor de metadados (idade, fumo, etc.)
        self.tab_mlp = nn.Sequential(
            nn.Linear(num_tab_features, tab_hidden_dim),
            nn.BatchNorm1d(tab_hidden_dim),
            nn.ReLU(inplace=True),
        )

        # --------------------------------------------------
        # Branch de Fusão
        # --------------------------------------------------
        # Dimensão de entrada: Soma das dimensões dos dois backbones + dimensão tabular
        fusion_in_dim = backbone_dim + backbone_dim + tab_hidden_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),

            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
        )

        # Classificador final
        self.classifier = nn.Linear(fusion_hidden_dim, num_classes)

    # ---------------------------------------------------------
    # Métodos de Controle de Congelamento (Freezing)
    # ---------------------------------------------------------

    def _set_backbone_requires_grad(self, requires_grad: bool) -> None:
        """Método auxiliar para iterar sobre parâmetros dos backbones."""
        for module in [self.backbone_oral, self.backbone_lesion]:
            for param in module.parameters():
                param.requires_grad = requires_grad

    def freeze_backbone(self) -> None:
        """
        Congela TOTALMENTE os backbones de visão (oral e lesão).
        Útil no início do treinamento para treinar apenas o classificador e evitar
        destruir os pesos pré-treinados com gradientes instáveis.
        """
        self._set_backbone_requires_grad(False)

    def unfreeze_backbone(self) -> None:
        """
        Descongela TOTALMENTE os backbones.
        Permite que toda a rede aprenda.
        """
        self._set_backbone_requires_grad(True)

    def unfreeze_backbone_last_fraction(self, fraction: float = 0.3) -> None:
        """
        Estratégia de Fine-Tuning Gradual:
        Descongela apenas uma fração final dos parâmetros dos backbones.

        A lógica é que as camadas iniciais aprendem traços genéricos (linhas, curvas),
        enquanto as finais aprendem conceitos semânticos específicos. Muitas vezes,
        queremos ajustar apenas a parte semântica.

        Args:
            fraction (float): Porcentagem (0.0 a 1.0) dos parâmetros finais a descongelar.
                              Ex: 0.3 descongela os últimos 30% da rede.
        """
        if fraction <= 0.0:
            # Nada a descongelar
            return

        fraction = min(max(fraction, 0.0), 1.0)

        for module in [self.backbone_oral.model, self.backbone_lesion.model]:
            params = list(module.parameters())
            if not params:
                continue

            # Passo 1: Garante que tudo está congelado
            for p in params:
                p.requires_grad = False

            # Passo 2: Descongela os últimos K parâmetros
            k = max(1, int(len(params) * fraction))
            for p in params[-k:]:
                p.requires_grad = True

    # ---------------------------------------------------------
    # Forward Pass (Passagem Direta)
    # ---------------------------------------------------------
    def forward(
        self,
        oral_image: torch.Tensor,
        lesion_image: torch.Tensor,
        tabular: torch.Tensor,
        has_lesion: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Executa a inferência do modelo.

        Args:
            oral_image:  Tensor de imagens da cavidade oral [B, 3, H, W].
            lesion_image:Tensor de imagens da lesão [B, 3, H, W].
            tabular:     Tensor de dados clínicos [B, num_tab_features].
            has_lesion:  Tensor auxiliar [B] indicando se a lesão existe (1) ou não (0).
                         Usado para zerar (mascarar) ruído vindo da imagem da lesão
                         quando ela é apenas um padding preto.

        Returns:
            logits: Tensor de saída não normalizado [B, num_classes].
        """
        # 1. Extração de características visuais
        oral_feat = self.backbone_oral(oral_image)        # Shape: [B, D]
        lesion_feat = self.backbone_lesion(lesion_image)  # Shape: [B, D]

        # 2. Mascaramento da Lesão
        # Se o paciente não tem lesão visível (has_lesion=0), multiplicamos o vetor
        # de características da lesão por 0. Isso impede que a rede aprenda com
        # o ruído de uma imagem preta.
        if has_lesion is not None:
            if has_lesion.dim() == 1:
                mask = has_lesion.view(-1, 1).float()
            else:
                mask = has_lesion[:, :1].float()
            lesion_feat = lesion_feat * mask

        # 3. Processamento Tabular
        tab_feat = self.tab_mlp(tabular)  # Shape: [B, tab_hidden_dim]

        # 4. Fusão (Concatenação)
        fused = torch.cat([oral_feat, lesion_feat, tab_feat], dim=1)  # Shape: [B, Total_Dim]

        # 5. Classificação
        x = self.fusion_mlp(fused)
        logits = self.classifier(x)
        return logits


# ---------------------------------------------------------
# Bloco de Teste Rápido (Executado apenas se rodar direto)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Iniciando teste de sanidade do modelo...")

    # Configurações fictícias
    BS = 2              # Batch Size
    NUM_CLASSES = 4
    NUM_TAB = 9         # Features tabulares
    IMG_SIZE = 224
    BACKBONE = "convnext_tiny" # Tente mudar para 'resnet18' ou 'mobilenetv3_large_100'

    try:
        # 1. Instanciação
        print(f"[INFO] Criando modelo com backbone: {BACKBONE}")
        model = MultiModalMobileNetV3Large(
            num_tab_features=NUM_TAB,
            num_classes=NUM_CLASSES,
            backbone_name=BACKBONE,
            pretrained=False # False apenas para ser mais rápido o teste
        )
        print("[OK] Modelo instanciado com sucesso.")

        # 2. Criação de tensores falsos (Dummy Data)
        oral_img = torch.randn(BS, 3, IMG_SIZE, IMG_SIZE)
        lesion_img = torch.randn(BS, 3, IMG_SIZE, IMG_SIZE)
        tabular_data = torch.randn(BS, NUM_TAB)
        has_lesion = torch.tensor([1.0, 0.0]) # Exemplo: 1º tem lesão, 2º não tem

        # 3. Teste do Forward Pass
        print("[INFO] Executando forward pass...")
        logits = model(oral_img, lesion_img, tabular_data, has_lesion)

        # 4. Verificação de saídas
        print(f"[INFO] Shape dos logits de saída: {logits.shape}")
        print(f"[INFO] Valor dos logits:\n{logits}")

        if logits.shape == (BS, NUM_CLASSES):
            print("[SUCESSO] O modelo processou as entradas e gerou a saída no formato correto.")
        else:
            print("[ERRO] O formato da saída está incorreto.")

        # 5. Teste de Freezing
        print("\n[INFO] Testando congelamento parcial...")
        model.unfreeze_backbone_last_fraction(0.5)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[INFO] Parâmetros treináveis: {trainable_params} de {total_params} ({trainable_params/total_params:.1%})")

    except ImportError as e:
        print(f"\n[ERRO CRÍTICO] {e}")
    except Exception as e:
        print(f"\n[ERRO] Ocorreu um erro durante o teste: {e}")