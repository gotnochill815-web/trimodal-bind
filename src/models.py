import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel


class ImageEncoder(nn.Module):
    """
    Image encoder using a pretrained ResNet18 backbone.
    Outputs normalized embeddings.
    """

    def __init__(self, embed_dim=256):
        super().__init__()

        self.backbone = timm.create_model(
            "resnet18",
            pretrained=True,
            num_classes=0
        )

        self.proj = nn.Linear(512, embed_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class AudioEncoder(nn.Module):
    """
    CNN-based encoder for mel spectrograms.
    """

    def __init__(self, embed_dim=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.proj = nn.Linear(64, embed_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(1)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class TextEncoder(nn.Module):
    """
    DistilBERT-based text encoder.
    """

    def __init__(self, embed_dim=256):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            "distilbert-base-uncased"
        )

        self.proj = nn.Linear(768, embed_dim)

    def forward(self, input_ids, attention_mask):
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = output.last_hidden_state[:, 0, :]
        x = self.proj(cls)

        return F.normalize(x, dim=-1)