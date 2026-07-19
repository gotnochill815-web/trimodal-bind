import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.models import ImageEncoder, AudioEncoder, TextEncoder
from src.dataset import TriModalDataset, collate_fn
from src.utils import tri_modal_loss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-4


def main():

    # Load metadata
    meta = pd.read_csv("ESC-50-master/meta/esc50.csv")

    meta = pd.read_csv("ESC-50-master/meta/esc50.csv")

    classes = sorted(meta["category"].unique())

    mini = meta[meta["category"].isin(classes)].copy()
    mini["text"] = mini["category"].str.replace("_", " ")

    print(f"Dataset size: {len(mini)}")

    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    dataset = TriModalDataset(mini)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn(tokenizer)
    )

    img_encoder = ImageEncoder().to(DEVICE)
    aud_encoder = AudioEncoder().to(DEVICE)
    txt_encoder = TextEncoder().to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(img_encoder.parameters()) +
        list(aud_encoder.parameters()) +
        list(txt_encoder.parameters()),
        lr=LR
    )

    print("Starting training...")

    for epoch in range(EPOCHS):

        img_encoder.train()
        aud_encoder.train()
        txt_encoder.train()

        total_loss = 0

        for imgs, auds, toks in loader:

            imgs = imgs.to(DEVICE)
            auds = auds.to(DEVICE)

            toks = {
                "input_ids": toks["input_ids"].to(DEVICE),
                "attention_mask": toks["attention_mask"].to(DEVICE)
            }

            optimizer.zero_grad()

            img_emb = img_encoder(imgs)
            aud_emb = aud_encoder(auds)
            txt_emb = txt_encoder(**toks)

            loss = tri_modal_loss(
                img_emb,
                aud_emb,
                txt_emb
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}"
        )

    torch.save(
        {
            "image_encoder": img_encoder.state_dict(),
            "audio_encoder": aud_encoder.state_dict(),
            "text_encoder": txt_encoder.state_dict()
        },
        "trimodal_bind.pt"
    )

    print("Training complete.")
    print("Model saved as trimodal_bind.pt")


if __name__ == "__main__":
    main()