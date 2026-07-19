#  TriModal-Bind

### Learning a Shared Embedding Space for Image, Audio and Text

TriModal-Bind is a multimodal representation learning project that explores learning a **shared embedding space** across images, text, and environmental sounds using contrastive learning.

The current implementation demonstrates **text-to-image retrieval**, where natural language queries are mapped into the same embedding space as images to retrieve semantically similar visual concepts.

 **Live Demo**

https://trimodal-bind-txe2npvpwtcmxtjovtrecn.streamlit.app/

---

#  Features

- Text → Image retrieval
- Contrastive learning using InfoNCE Loss
- Shared multimodal embedding space
- ResNet18 image encoder
- DistilBERT text encoder
- ESC-50 environmental sound classes
- Stable Diffusion generated representative images
- Cosine similarity search
- Streamlit web interface
- PyTorch implementation

---

#  Project Structure

```text
trimodal-bind/
│
├── src/
│   ├── streamlit_app.py
│   ├── models.py
│   ├── dataset.py
│   ├── train.py
│   └── utils.py
│
├── generated_images/
├── notebooks/
│   └── trimodal_bind_v1.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Model Architecture

```
              Text Query
                   │
          DistilBERT Encoder
                   │
                   ▼
          Text Embedding (256D)
                   │
                   ▼
         Shared Embedding Space
                   ▲
                   │
         Image Embedding (256D)
                   ▲
             ResNet18 Encoder
                   ▲
                 Images
```

The encoders are trained using **contrastive learning (InfoNCE Loss)** so that semantically related samples are positioned close together in the embedding space.

---

#  Dataset

The project uses the **ESC-50** environmental sound dataset.

- 50 sound categories
- 2,000 labelled audio clips
- 5-second environmental recordings

Representative images for each category were generated using Stable Diffusion and are used for image retrieval.

Example classes include:

- Dog
- Rain
- Helicopter
- Chainsaw
- Fireworks
- Crying Baby
- Clock Alarm
- Wind

---

#  Training

Training uses:

- PyTorch
- Contrastive Learning
- InfoNCE Loss
- Adam Optimizer

The objective encourages semantically related samples to have nearby embeddings while separating unrelated samples.

---

#  Retrieval Pipeline

Given a text query:

1. Encode text using DistilBERT
2. Project into the shared embedding space
3. Compare against image embeddings
4. Compute cosine similarity
5. Retrieve the Top-5 most similar images

---

#  Running Locally

Clone the repository

```bash
git clone https://github.com/gotnochill815-web/trimodal-bind.git
```

Enter the project

```bash
cd trimodal-bind
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch the application

```bash
streamlit run src/streamlit_app.py
```

---

#  Technologies

- Python
- PyTorch
- TorchVision
- Transformers
- Hugging Face Hub
- Streamlit
- PIL
- NumPy

---

#  Future Work

The long-term goal is to support complete multimodal retrieval.

Planned additions include:

- Audio → Image retrieval
- Text → Audio retrieval
- Image → Audio retrieval
- CLAP-based audio encoder
- FAISS indexing
- Hard negative mining
- Recall@K and mAP evaluation
- CLIP Vision encoder
- Larger multimodal datasets

---

#  Author

**Prakhya Khandelwal**

AI/ML Researcher

GitHub:
https://github.com/gotnochill815-web

---

#  Support

If you found this project useful, consider giving the repository a star on GitHub.
