import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from transformers import ViTImageProcessor, ViTForImageClassification
from tqdm import tqdm


class NpySpectrogramDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor

        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith(".npy"):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

        print("Classes:", self.classes)
        print("Total samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        spec = np.load(path)

        # Handle possible shapes: [H, W], [1, H, W], [H, W, 1]
        spec = np.squeeze(spec)

        # Normalize to 0-255
        spec = spec.astype(np.float32)
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        spec = (spec * 255).astype(np.uint8)

        # Convert grayscale spectrogram to RGB because pretrained ViT expects 3 channels
        img = Image.fromarray(spec).convert("RGB")

        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long)
        }


def train():
    root_dir = "sorted_each_300_files"

    model_name = "google/vit-base-patch16-224-in21k"

    processor = ViTImageProcessor.from_pretrained(model_name)

    dataset = NpySpectrogramDataset(root_dir, processor)

    num_classes = len(dataset.classes)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label={i: c for i, c in enumerate(dataset.classes)},
        label2id={c: i for i, c in enumerate(dataset.classes)}
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits

            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values=pixel_values)
                preds = torch.argmax(outputs.logits, dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}: "
            f"loss={total_loss/len(train_loader):.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"val_acc={val_acc:.4f}"
        )

    model.save_pretrained("vit_spectrogram_classifier")
    processor.save_pretrained("vit_spectrogram_classifier")


if __name__ == "__main__":
    train()