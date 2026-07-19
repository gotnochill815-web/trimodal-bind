import torch
import streamlit as st
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from models import ImageEncoder, TextEncoder

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="TriModal-Bind",
    page_icon="🔍",
    layout="wide"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Project Root
ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT_DIR / "generated_images"

# -------------------------------------------------
# Download Checkpoint
# -------------------------------------------------

CHECKPOINT = hf_hub_download(
    repo_id="prakhya15/trimodal-bind-model",
    filename="trimodal_bind.pt"
)

CLASS_NAMES = [
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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------------------------------
# Load Everything Once
# -------------------------------------------------

@st.cache_resource
def load_resources():

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

    gallery = []

    with torch.no_grad():

        for cls in CLASS_NAMES:

            path = IMAGE_DIR / f"{cls}.png"

            if not path.exists():
                continue

            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                continue

            tensor = transform(image).unsqueeze(0).to(DEVICE)

            emb = img_encoder(tensor)[0].cpu()
            emb = emb / emb.norm()

            gallery.append({
                "label": cls.replace("_", " "),
                "path": path,
                "embedding": emb
            })

    return tokenizer, txt_encoder, gallery


tokenizer, txt_encoder, gallery = load_resources()

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
        query_embedding = txt_encoder(**tokens)[0].cpu()

    query_embedding = query_embedding / query_embedding.norm()

    scores = []

    for item in gallery:

        similarity = torch.dot(
            query_embedding,
            item["embedding"]
        ).item()

        scores.append((similarity, item))

    scores.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return scores[:5]

# -------------------------------------------------
# UI
# -------------------------------------------------

st.title("🔍 TriModal-Bind")

st.write(
    "Retrieve the most semantically similar images using a shared multimodal embedding space."
)

query = st.text_input(
    "Enter a text query",
    placeholder="dog barking"
)

if st.button("Retrieve"):

    if not query.strip():
        st.warning("Please enter a query.")
        st.stop()

    results = retrieve(query)

    cols = st.columns(len(results))

    for col, (score, item) in zip(cols, results):

        with col:
            st.image(
                str(item["path"]),
                use_container_width=True
            )

            st.markdown(
                f"**{item['label']}**"
            )

            st.caption(
                f"Similarity: {score:.3f}"
            )
