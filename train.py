from data import NucleusDataset, Rescale, ToTensor, Normalize
from model import UNet
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable


def train(epochs=100, batch_size=16, lr=0.001):
    train_loader = torch.utils.data.DataLoader(
        NucleusDataset('data', train=True,
                       transform=transforms.Compose([
                           Normalize(),
                           Rescale(256),
                           ToTensor()
                       ]),
                       target_transform=transforms.Compose([
                           Normalize(),
                           Rescale(256),
                           ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True)

    model = UNet()
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, masks = images.cuda(), masks.cuda()
            images, masks = Variable(images), Variable(masks)

            optimizer.zero_grad()

            output = model(images)
            loss = F.binary_cross_entropy(output, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(images), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), running_loss/100))
                running_loss = 0.0


if __name__ == "__main__":
    train()
