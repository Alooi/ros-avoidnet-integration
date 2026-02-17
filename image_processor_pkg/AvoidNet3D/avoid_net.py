import torch
import torch.nn as nn


class ImageReducer(nn.Module):
    def __init__(self):
        super(ImageReducer, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))

        return x


class ImageReducer_bounded(nn.Module):
    def __init__(self):
        super(ImageReducer_bounded, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))

        # Normalize between 0 and 1
        max_val = torch.max(x)
        if max_val > 0:
            x = x / max_val
        else:
            x = torch.zeros_like(x)
        return x


class ImageReducer_bounded_grayscale(nn.Module):
    def __init__(self):
        super(ImageReducer_bounded_grayscale, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))

        # Normalize between 0 and 1
        max_val = torch.max(x)
        if max_val > 0:
            x = x / max_val
        else:
            x = torch.zeros_like(x)
        return x

class ImageReducer_bounded_grayscale_q(nn.Module):
    def __init__(self):
        super(ImageReducer_bounded_grayscale_q, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=0)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        # Always use the fused path after quantization/fusion
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # Normalize between 0 and 1 only if not quantized
        if not x.is_quantized:
            max_val = torch.max(x)
            if max_val > 0:
                x = x / max_val
            else:
                x = torch.zeros_like(x)
        return x
    
    def fuse_model(self):
        # Fusing the model layers
        torch.quantization.fuse_modules(self, [['conv1', 'relu1'], 
                                               ['conv2', 'relu2'], 
                                               ['conv3', 'relu3'], 
                                               ['conv4', 'relu4'], 
                                               ['conv5', 'relu5']], inplace=True)
        return self


class ImageReducer_Large(nn.Module):
    def __init__(self):
        super(ImageReducer_Large, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = torch.relu(self.conv7(x))
        x = torch.sigmoid(self.conv8(x))

        return x


def get_model(arc):
    if arc == "ImageReducer":
        return ImageReducer()
    elif arc == "ImageReducer_bounded":
        return ImageReducer_bounded()
    elif arc == "ImageReducer_bounded_grayscale":
        return ImageReducer_bounded_grayscale()
    elif arc == "ImageReducer_bounded_grayscale_q":
        return ImageReducer_bounded_grayscale_q()
    elif arc == "ImageReducer_Large":
        return ImageReducer_Large()
    else:
        raise NotImplementedError(f"{arc} is not implemented yet")
