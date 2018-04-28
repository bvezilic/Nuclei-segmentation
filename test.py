import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from data import NucleusDataset, Rescale, ToTensor, Normalize


def test():
    test_loader = DataLoader(NucleusDataset('data',
                                            train=False,
                                            transform=transforms.Compose([
                                                Normalize(),
                                                Rescale(256),
                                                ToTensor()])),
                             batch_size=12,
                             shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load("models/model.pt")
    model.eval()
    model.to(device)

    with torch.no_grad():
        images = next(iter(test_loader)).to(device)

        outputs = model(images)

        images = tensor_to_numpy(images)
        outputs = tensor_to_numpy(outputs)

        show_images(images, outputs)


def tensor_to_numpy(tensor):
    t_numpy = tensor.cpu().numpy()
    t_numpy = np.transpose(t_numpy, [0, 2, 3, 1])
    t_numpy = np.squeeze(t_numpy)

    return t_numpy


def show_images(images, masks, columns=6):
    fig = plt.figure()
    rows = np.ceil((images.shape[0] + masks.shape[0]) / columns)
    index = 1
    for image, mask in zip(images, masks):
        f1 = fig.add_subplot(rows, columns, index)
        f1.set_title('input')
        plt.axis('off')
        plt.imshow(image)
        index += 1

        f2 = fig.add_subplot(rows, columns, index)
        f2.set_title('prediction')
        plt.axis('off')
        plt.imshow(mask)
        index += 1

    plt.show()


if __name__ == "__main__":
    test()
