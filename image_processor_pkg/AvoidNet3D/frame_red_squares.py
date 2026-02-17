import torch
import numpy as np
from PIL import Image
from avoid_net import get_model
from dataset import SUIM_grayscale
from draw_obsticle import draw_red_squares

class RedSquaresGrid:
    def __init__(self, arc, run_name, use_gpu=False, que=False, threshold=0.5):
        if que:
            model = get_model(arc + '_q')
        else:
            model = get_model(arc)
        model.load_state_dict(torch.load(f"models/{arc}_{run_name}.pth"))
        device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        model.to(device).eval()
        self.model = model
        self.device = device
        dataset = SUIM_grayscale("")
        self.image_transform = dataset.get_transform()
        self.threshold = threshold

    def __call__(self, frame):
        # frame: numpy array (H, W, C)
        frame_tensor = Image.fromarray(frame)
        frame_tensor = self.image_transform(frame_tensor).to(self.device).unsqueeze(0)
        outputs = self.model(frame_tensor)
        outputs = outputs.detach().cpu()[0].permute(1, 2, 0)
        outputs = np.array(outputs)
        # frame_with_squares = draw_red_squares(frame, outputs, self.threshold)
        return outputs


# example usage:
# red_squares_grid = RedSquaresGrid("ImageReducer_bounded_grayscale", "run_2", use_gpu=True, que=True)
# frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
# output = red_squares_grid(frame)
# print("Output shape:", output.shape)
# print("Output type:", output.dtype)
# print("Output min:", output.min())
# print("Output max:", output.max())
