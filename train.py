from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose

from data import NucleusDataset
from losses import bce_and_dice
from metrics import iou_score
from model import UNet

from transform import Rescale


class Trainer:
    """Helper trainer class that collects all components for training and runs training loop.
    """
    def __init__(self, dataset, model, optimizer, batch_size, device="cpu", val_dataset=None, save_every_epoch=5,
                 output_dir=None):
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.device = device
        self.val_dataset = val_dataset
        self.save_every_epoch = save_every_epoch
        self.output_dir = output_dir or Path()

        self.train_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.val_data_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

    def run_train_loop(self, epochs):
        # Run training
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 10)

            running_loss = 0.0
            running_score = 0.0
            for batch_idx, (images, masks) in enumerate(self.train_data_loader):
                # Obtain batches
                images, masks = images.to(self.device), masks.to(self.device)

                # Zero previous gradients
                self.optimizer.zero_grad()

                # Run forward pass
                output = self.model(images)

                # Compute loss
                loss = bce_and_dice(input=output, target=masks)

                # Compute IoU score
                score = iou_score(inputs=output, masks=masks)

                # Compute gradients
                loss.backward()

                # Update weights
                self.optimizer.step()

                running_loss += loss.item()
                running_score += score.item()

            # Run evaluation
            epoch_loss = running_loss / len(self.train_data_loader)
            print(f"Train - Loss: {epoch_loss:.4f}")

            epoch_score = running_score / len(self.train_data_loader)
            print(f"Train - Score: {epoch_score:.4f}")

            if self.val_data_loader:
                self.run_val_loop()

            # Save weights every n-th epoch
            if (epoch + 1) % self.save_every_epoch == 0:
                self.model.save(self.output_dir / f"weights_e:{epoch+1}_loss:{epoch_loss:.4f}.pt")

    def run_val_loop(self):
        with torch.no_grad():
            running_loss = 0.
            running_score = 0.
            for batch_idx, (images, masks) in enumerate(self.val_data_loader):
                # Obtain batches
                images, masks = images.to(self.device), masks.to(self.device)

                # Run forward pass
                output = self.model(images)

                # Compute loss
                loss = bce_and_dice(input=output, target=masks)

                # Compute IoU score
                score = iou_score(inputs=output, masks=masks)

                running_loss += loss.item()
                running_score += score.item()

            # Run evaluation
            epoch_loss = running_loss / len(self.val_data_loader)
            print(f"Val - Loss: {epoch_loss:.4f}")

            epoch_score = running_score / len(self.val_data_loader)
            print(f"Val - Score: {epoch_score:.4f}")


def train():
    # Load the data sets
    train_dataset = NucleusDataset("data", train=True,
                                   transform=Compose([
                                       Rescale(256),
                                       ToTensor()
                                   ]),
                                   target_transform=Compose([
                                       Rescale(256),
                                       ToTensor()
                                   ]))

    # Use cuda if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set model to GPU/CPU
    if args.from_checkpoint:
        model = UNet.load(args.from_checkpoint)
    else:
        model = UNet()
    model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initialize trainer
    trainer = Trainer(dataset=train_dataset,
                      model=model,
                      optimizer=optimizer,
                      batch_size=args.batch_size,
                      device=args.device,
                      output_dir=output_dir)

    # Run the training
    trainer.run_train_loop(epochs=args.epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=30)
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("-lr", "--learning_rate", default=0.001)
    parser.add_argument("-fc", "--from_checkpoint", type=str, default=None)
    parser.add_argument("-d", "--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("-o", "--output_dir", type=str, default="./models/test1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    train()
