# TriModal Bind

A lightweight multimodal retrieval system that aligns **image, audio, and text** into a shared embedding space using contrastive learning.

## Key Features

- Tri-modal representation learning
- Text → Image retrieval
- Audio / Text semantic alignment
- Gradio interactive demo
- Trained in Google Colab using GPU

## Architecture

- Image Encoder: ResNet18
- Audio Encoder: CNN on Mel Spectrograms
- Text Encoder: DistilBERT
- Shared Embedding Size: 256

## Training Results

Loss decreased from:

1.56 → 1.18 in 5 epochs

## Demo

Typing `dog` retrieves dog image using learned embeddings.

## Tech Stack

PyTorch, Transformers, timm, librosa, Gradio

## Future Improvements

- More classes
- Real image datasets
- Audio-to-image retrieval
- Hugging Face deployment
