
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import numpy as np

from depth_anything.dpt import DepthAnything

class DepthAnythingProcessor:
    def __init__(self, encoder='vitl', device='cuda'):
        self.device = device
        self.model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(self.device).eval()
        self.transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def infer_pil(self, image):
        with torch.no_grad():
            img = self.transform(image).unsqueeze(0).to(self.device)
            depth = self.model(img)
            depth = depth.squeeze().cpu().numpy()
        
        # Convert to PIL image
        depth_pil = Image.fromarray((depth / depth.max() * 255).astype(np.uint8))
        return depth_pil

    def __call__(self, image):
        return self.infer_pil(image)
