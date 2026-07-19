
import os
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from models import ImageEncoder, TextEncoder
# -------------------------------------------------
# Configuration
# -------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "generated_images"

CHECKPOINT = hf_hub_download(
    repo_id="prakhya15/trimodal-bind-model",
    filename="trimodal_bind.pt"
)

class_names = [
    "dog","rooster","pig","cow","frog",
    "cat","hen","insects","sheep","crow",
    "rain","sea_waves","crackling_fire","crickets","chirping_birds",
    "water_drops","wind","pouring_water","toilet_flush","thunderstorm",
    "crying_baby","sneezing","clapping","breathing","coughing",
    "footsteps","laughing","brushing_teeth","snoring","drinking_sipping",
    "door_wood_knock","mouse_click","keyboard_typing","door_wood_creaks","can_opening",
    "washing_machine","vacuum_cleaner","clock_alarm","clock_tick","glass_breaking",
    "helicopter","chainsaw","siren","car_horn","engine",
    "train","church_bells","airplane","fireworks","hand_saw"
]

TOP_K = 5

# -------------------------------------------------
# Metadata
# -------------------------------------------------

mini = []
for cls in class_names:
    p = IMAGE_DIR / f"{cls}.png"
    if p.exists():
        mini.append({"category": cls, "text": cls.replace("_", " ")})

print(f"Found {len(mini)} images.")

# -------------------------------------------------
# Transform
# -------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------------------------
# Models
# -------------------------------------------------

img_encoder = ImageEncoder().to(DEVICE)
txt_encoder = TextEncoder().to(DEVICE)

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

img_encoder.load_state_dict(checkpoint["image_encoder"])
txt_encoder.load_state_dict(checkpoint["text_encoder"])

img_encoder.eval()
txt_encoder.eval()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# -------------------------------------------------
# Gallery
# -------------------------------------------------

gallery = []

print("Building gallery...")

with torch.no_grad():
    for row in mini:
        image_path = IMAGE_DIR / f'{row["category"]}.png'

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {image_path}: {e}")
            continue

        tensor = transform(image).unsqueeze(0).to(DEVICE)

        emb = img_encoder(tensor)[0].cpu()
        emb = emb / emb.norm()

        gallery.append({
            "label": row["text"],
            "path": str(image_path),
            "embedding": emb
        })

print(f"Gallery built with {len(gallery)} images.")

# -------------------------------------------------
# Retrieval
# -------------------------------------------------

def retrieve(query):
    tokens = tokenizer(
        [query],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    tokens = {
        "input_ids": tokens["input_ids"].to(DEVICE),
        "attention_mask": tokens["attention_mask"].to(DEVICE),
    }

    with torch.no_grad():
        q = txt_encoder(**tokens)[0].cpu()

    q = q / q.norm()

    scores = []
    for item in gallery:
        score = torch.dot(q, item["embedding"]).item()
        scores.append({
            "score": score,
            "label": item["label"],
            "path": item["path"],
        })

    scores.sort(key=lambda x: x["score"], reverse=True)

    gallery_output = []
    ranking = ""

    for i, item in enumerate(scores[:TOP_K], start=1):
        gallery_output.append((item["path"], f"{i}. {item['label']}"))
        ranking += f"{i}. {item['label']} (Similarity={item['score']:.3f})\n"

    return gallery_output, ranking

# -------------------------------------------------
# UI
# -------------------------------------------------

demo = gr.Interface(
    fn=retrieve,
    inputs=gr.Textbox(
        label="Text Query",
        placeholder="e.g. dog barking"
    ),
    outputs=[
        gr.Gallery(label="Top Retrieved Images", columns=5, height=300),
        gr.Textbox(label="Ranking")
    ],
    title="TriModal-Bind",
    description="Text-to-image retrieval using a shared multimodal embedding space.",
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
    )
