
import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image

_BASE = os.path.dirname(__file__)

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

class DepthAnythingProcessor:
    def __init__(self, encoder='vits', checkpoint=None, device=None, input_size=518,
                 metric=False, max_depth=20.0):
        """
        Args:
            metric:    If True, load the metric-depth variant which outputs depth in metres.
            max_depth: Maximum depth range in metres (only used when metric=True).
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.input_size = input_size
        self.metric = metric
        self.max_depth = max_depth

        if metric:
            metric_path = os.path.join(_BASE, 'Depth_Anything_V2', 'metric_depth')
            if metric_path not in sys.path:
                sys.path.insert(0, metric_path)
            from depth_anything_v2.dpt import DepthAnythingV2 as _Model
            if checkpoint is None:
                checkpoint = os.path.join(metric_path, 'checkpoints', f'{encoder}_metric.pth')
            model_kwargs = {**MODEL_CONFIGS[encoder], 'max_depth': max_depth}
        else:
            da2_path = os.path.join(_BASE, 'Depth_Anything_V2')
            if da2_path not in sys.path:
                sys.path.insert(0, da2_path)
            from depth_anything_v2.dpt import DepthAnythingV2 as _Model
            if checkpoint is None:
                checkpoint = os.path.join(da2_path, 'checkpoints', f'{encoder}.pth')
            model_kwargs = MODEL_CONFIGS[encoder]

        self.model = _Model(**model_kwargs)
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        self.model = self.model.to(self.device).eval()

    def infer_pil(self, image):
        """Accept a PIL image, return a grayscale PIL depth image.

        For metric depth the pixel values encode depth in metres normalised to
        [0, 255] over the configured max_depth range.
        """
        raw = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        depth = self.model.infer_image(raw, self.input_size)  # H×W float32

        scale = self.max_depth if self.metric else depth.max()
        depth_u8 = np.clip(depth / scale * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(depth_u8)

    def infer_bgr(self, raw_image):
        """Accept a BGR numpy array (OpenCV format), return a float32 depth array.

        Metric mode: values are in metres (0..max_depth).
        Relative mode: values are relative (no physical unit).
        """
        return self.model.infer_image(raw_image, self.input_size)

    def __call__(self, image):
        return self.infer_pil(image)
