import os
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from src.models import ImageEncoder, TextEncoder

# -------------------------------------------------
# Configuration
# -------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = hf_hub_download(
    repo_id="prakhya15/trimodal-bind-model",
    filename="trimodal_bind.pt"
)

IMAGE_DIR = "generated_images"

class_names = [
    "dog", "rooster", "pig", "cow", "frog",
    "cat", "hen", "insects", "sheep", "crow",
    "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds",
    "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm",
    "crying_baby", "sneezing", "clapping", "breathing", "coughing",
    "footsteps", "laughing", "brushing_teeth", "snoring", "drinking_sipping",
    "door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creaks", "can_opening",
    "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick", "glass_breaking",
    "helicopter", "chainsaw", "siren", "car_horn", "engine",
    "train", "church_bells", "airplane", "fireworks", "hand_saw"
]

TOP_K = 5

# -------------------------------------------------
# Build metadata
# -------------------------------------------------

mini = []

for cls in class_names:
    image_path = os.path.join(IMAGE_DIR, f"{cls}.png")

    if os.path.exists(image_path):
        mini.append({
            "category": cls,
            "text": cls.replace("_", " ")
        })

print(f"Found {len(mini)} images.")

# -------------------------------------------------
# Image Transform
# -------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------------------------
# Load Models
# -------------------------------------------------

img_encoder = ImageEncoder().to(DEVICE)
txt_encoder = TextEncoder().to(DEVICE)

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

img_encoder.load_state_dict(checkpoint["image_encoder"])
txt_encoder.load_state_dict(checkpoint["text_encoder"])

img_encoder.eval()
txt_encoder.eval()

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

# -------------------------------------------------
# Build Gallery
# -------------------------------------------------

gallery = []

print("Building gallery...")

with torch.no_grad():

    for row in mini:

        category = row["category"]
        label = row["text"]

        image_path = os.path.join(
            IMAGE_DIR,
            f"{category}.png"
        )

        image = Image.open(image_path).convert("RGB")

        tensor = transform(image).unsqueeze(0).to(DEVICE)

        emb = img_encoder(tensor)[0].cpu()
        emb = emb / emb.norm()

        gallery.append({
            "label": label,
            "path": image_path,
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
        truncation=True
    )

    tokens = {
        "input_ids": tokens["input_ids"].to(DEVICE),
        "attention_mask": tokens["attention_mask"].to(DEVICE)
    }

    with torch.no_grad():
        q = txt_encoder(**tokens)[0].cpu()

    q = q / q.norm()

    scores = []

    for item in gallery:

        score = torch.dot(
            q,
            item["embedding"]
        ).item()

        scores.append({
            "score": score,
            "label": item["label"],
            "path": item["path"]
        })

    scores.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    gallery_output = []
    result_text = ""

    for rank, item in enumerate(scores[:TOP_K], start=1):

        gallery_output.append(
            (
                item["path"],
                f"{rank}. {item['label']}"
            )
        )

        result_text += (
            f"{rank}. {item['label']} "
            f"(Similarity = {item['score']:.3f})\n"
        )

    return gallery_output, result_text

# -------------------------------------------------
# Gradio UI
# -------------------------------------------------

demo = gr.Interface(
    fn=retrieve,
    inputs=gr.Textbox(
        label="Text Query",
        placeholder="e.g. dog barking"
    ),
    outputs=[
        gr.Gallery(
            label="Top Retrieved Images",
            columns=5,
            height=300
        ),
        gr.Textbox(label="Ranking")
    ],
    title="TriModal-Bind",
    description="""
Retrieve the most semantically similar images from natural language
using a shared multimodal embedding space learned through
contrastive learning.
"""
)

# -------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
