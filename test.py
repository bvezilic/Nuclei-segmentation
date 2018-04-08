import random
from data import NucleusDataset, Rescale, ToTensor, Normalize
import torch
import cv2
from torchvision import transforms
from torch.utils.data.sampler import RandomSampler
from torch.autograd import Variable


def test():
    random_sampler = RandomSampler(NucleusDataset('data', train=False,
                                                  transform=transforms.Compose([
                                                      Normalize(),
                                                      Rescale(256),
                                                      ToTensor()
                                                  ])))

    data_set = NucleusDataset('data', train=False,
                                                  transform=transforms.Compose([
                                                      Normalize(),
                                                      Rescale(256),
                                                      ToTensor()
                                                  ]))
    model = torch.load("models/model.pt")
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # for idx, image in enumerate(random_sampler):
    image = data_set[7]
    image = image.unsqueeze(0)
    image = Variable(image.cuda())
    output = model(image)
    cv2.imshow('ouput', output.data.cpu().numpy().squeeze())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test()
