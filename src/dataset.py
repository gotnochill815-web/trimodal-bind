import librosa
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F


img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class TriModalDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        category = row["category"]
        text = row["text"]

        # Image
        image = Image.open(
            f"generated_images/{category}.png"
        ).convert("RGB")

        image = img_transform(image)

        # Audio
        y, sr = librosa.load(
            f"ESC-50-master/audio/{row['filename']}",
            sr=16000
        )

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128
        )

        mel = librosa.power_to_db(mel)

        mel = torch.tensor(mel).float().unsqueeze(0)

        return image, mel, text


def collate_fn(tokenizer):
    def collate(batch):

        images, audios, texts = zip(*batch)

        images = torch.stack(images)

        max_width = max(audio.shape[-1] for audio in audios)

        padded_audio = []

        for audio in audios:
            pad = max_width - audio.shape[-1]
            padded_audio.append(F.pad(audio, (0, pad)))

        audios_tensor = torch.stack(padded_audio)

        tokens = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        return images, audios_tensor, tokens

    return collate