#  Trimodal-Bind
### Learning a Shared Embedding Space for Image, Audio and Text Retrieval

Trimodal-Bind is a deep learning project that learns a **shared embedding space** for three different modalities:
-  Images
-  Audio
-  Text

The model is trained using **contrastive learning (InfoNCE Loss)** so that semantically similar image, audio, and text samples are mapped close together in the embedding space.

Once trained, the system can retrieve the most relevant image given an audio clip, making it a simple multimodal retrieval engine.

---

#  Features

- Joint Image, Audio and Text embeddings
- Contrastive (InfoNCE) training
- ESC-50 dataset support
- Stable Diffusion generated image gallery
- Gradio Web Interface
- Modular PyTorch implementation
- Cosine Similarity Retrieval
- Top-K Image Retrieval

---

#  Project Structure

```
trimodal-bind/
│
├── src/
│   ├── app.py
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
│
├── generated_images/
│
├── notebooks/
│   └── trimodal_bind_v1.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

#  Model Architecture

The system consists of three independent encoders.

```
                 Image
                   │
           ResNet18 Encoder
                   │
                   ▼
              Image Embedding
                   │
                   │
                   ▼
             Shared Embedding Space
                   ▲
                   │
                   │
          Audio Embedding
                   ▲
            CLAP Encoder
                   ▲
                 Audio


                 Text
                   │
             CLIP Text Encoder
                   │
                   ▼
              Text Embedding
```

All embeddings are projected into the same latent space and trained using **InfoNCE Loss**.

---

#  Dataset

**ESC-50**

- 50 environmental sound classes
- 2,000 labeled audio clips
- 5-second audio recordings

Examples include:

- Dog
- Cat
- Rain
- Helicopter
- Clock Alarm
- Chainsaw
- Rooster
- Fireworks
- Crying Baby
- Wind

Each class is represented by an AI-generated image using Stable Diffusion.

---

#  Training

The model is trained using:

- PyTorch
- Contrastive Learning
- InfoNCE Loss
- Adam Optimizer

During training, positive image-audio-text triplets are pulled closer while unrelated samples are pushed apart.

---

#  Retrieval Pipeline

Given an input audio:

1. Encode audio using CLAP
2. Compute embedding
3. Compare against image embeddings
4. Compute cosine similarity
5. Retrieve Top-K most similar images

---

#  Demo

The project includes a Gradio interface.

Run:

```bash
python src/app.py
```

Upload an audio clip and retrieve the most relevant images.

---

#  Installation

Clone the repository

```bash
git clone https://github.com/gotnochill815-web/trimodal-bind.git
```

Move into the project

```bash
cd trimodal-bind
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🛠 Technologies

- PyTorch
- Transformers
- OpenCLIP
- LAION CLAP
- Stable Diffusion
- Gradio
- TorchVision
- NumPy

---

#  Future Improvements

- Train on larger multimodal datasets
- Support text-to-image retrieval
- Cross-modal image captioning
- FAISS indexing for large-scale retrieval
- Hard negative mining
- Recall@K and mAP evaluation
- Replace ResNet18 with CLIP Vision Encoder
- Multi-image generation per class
- End-to-end training of all encoders

---

#  Author

**Prakhya Khandelwal**

AI/ML Researcher

- GitHub: https://github.com/gotnochill815-web
- LinkedIn: *(Add your LinkedIn profile here)*

---

#  If you found this project useful

Please consider giving it a star on GitHub.
