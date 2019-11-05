import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """UNet with skip-connection as in https://github.com/jocicmarko/ultrasound-nerve-segmentation
    """
    def __init__(self, kernel_size=3, padding=1):
        super(UNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=kernel_size, padding=padding)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding)
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=kernel_size, padding=padding)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding)
        self.maxpool4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=kernel_size, padding=padding)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=padding)
        self.conv5_t = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=kernel_size, padding=padding)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding)
        self.conv6_t = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=kernel_size, padding=padding)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding)
        self.conv7_t = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=kernel_size, padding=padding)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding)
        self.conv8_t = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=kernel_size, padding=padding)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding)

        self.conv10 = nn.Conv2d(32, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = F.relu(self.conv1_1(x))                             # [B, F=32, H=256, W=256]
        conv1 = F.relu(self.conv1_2(conv1))                         # [B, F=32, H=256, W=256]
        pool1 = self.maxpool1(conv1)                                # [B, F=32, H=128, W=128]

        conv2 = F.relu(self.conv2_1(pool1))                         # [B, F=64, H=128, W=128]
        conv2 = F.relu(self.conv2_2(conv2))                         # [B, F=64, H=128, W=128]
        pool2 = self.maxpool2(conv2)                                # [B, F=64, H=64, W=64]

        conv3 = F.relu(self.conv3_1(pool2))                         # [B, F=128, H=64, W=64]
        conv3 = F.relu(self.conv3_2(conv3))                         # [B, F=128, H=64, W=64]
        pool3 = self.maxpool3(conv3)                                # [B, F=128, H=32, W=32]

        conv4 = F.relu(self.conv4_1(pool3))                         # [B, F=256, H=32, W=32]
        conv4 = F.relu(self.conv4_2(conv4))                         # [B, F=256, H=32, W=32]
        pool4 = self.maxpool4(conv4)                                # [B, F=256, H=16, W=16]

        conv5 = F.relu(self.conv5_1(pool4))                         # [B, F=512, H=16, W=16]
        conv5 = F.relu(self.conv5_2(conv5))                         # [B, F=512, H=16, W=16]

        up6 = torch.cat((self.conv5_t(conv5), conv4), dim=1)        # [B, F=256, H=32, W=32] ⊕ [B, F=256, H=32, W=32] => [B, F=512, H=32, W=32]
        conv6 = F.relu(self.conv6_1(up6))                           # [B, F=256, H=32, W=32]
        conv6 = F.relu(self.conv6_2(conv6))                         # [B, F=256, H=32, W=32]

        up7 = torch.cat((self.conv6_t(conv6), conv3), dim=1)        # [B, F=128, H=64, W=64] ⊕ [B, F=128, H=64, W=64] => [B, F=256, H=64, W=64]
        conv7 = F.relu(self.conv7_1(up7))                           # [B, F=128, H=64, W=64]
        conv7 = F.relu(self.conv7_2(conv7))                         # [B, F=128, H=64, W=64]

        up8 = torch.cat((self.conv7_t(conv7), conv2), dim=1)        # [B, F=64, H=128, W=128] ⊕ [B, F=64, H=128, W=128] => [B, F=128, H=128, W=128]
        conv8 = F.relu(self.conv8_1(up8))                           # [B, F=64, H=128, W=128]
        conv8 = F.relu(self.conv8_2(conv8))                         # [B, F=64, H=128, W=128]

        up9 = torch.cat((self.conv8_t(conv8), conv1), dim=1)        # [B, F=32, H=256, W=256] ⊕ [B, F=32, H=256, W=256] => [B, F=64, H=256, W=256]
        conv9 = F.relu(self.conv9_1(up9))                           # [B, F=32, H=256, W=256]
        conv9 = F.relu(self.conv9_2(conv9))                         # [B, F=32, H=256, W=256]

        return self.sigmoid(self.conv10(conv9))                        # [B, F=1, H=256, W=256]

    @classmethod
    def load(cls, weights_path):
        print(f"Loading UNet from path `{weights_path}`")
        model = cls()
        model.load_state_dict(torch.load(weights_path))

        return model

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)
        print(f"Saved model on path: {save_path}")
