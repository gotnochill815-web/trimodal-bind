import torch
import torch.nn.functional as F


def contrastive_loss(a, b, temperature=0.07):
    """
    Symmetric InfoNCE loss between two modalities.
    """

    logits = (a @ b.T) / temperature
    labels = torch.arange(a.size(0), device=a.device)

    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)

    return (loss_a + loss_b) / 2


def tri_modal_loss(img_emb, aud_emb, txt_emb):
    """
    Average pairwise contrastive loss.
    """

    loss_it = contrastive_loss(img_emb, txt_emb)
    loss_ia = contrastive_loss(img_emb, aud_emb)
    loss_at = contrastive_loss(aud_emb, txt_emb)

    return (loss_it + loss_ia + loss_at) / 3


def cosine_topk(query_embedding, gallery_embeddings, k=5):
    """
    Returns indices of the Top-K most similar embeddings.
    """

    similarities = gallery_embeddings @ query_embedding.T

    values, indices = torch.topk(similarities.squeeze(), k)

    return indices, values