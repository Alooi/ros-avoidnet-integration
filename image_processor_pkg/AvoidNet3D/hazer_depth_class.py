import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .utils.fps_timer import fps


class HazerDepth:
    def __init__(
        self,
        show_video=False,
        show_graph=False,
        patch_size=15,
        H=640,
        W=480,
        binary=True,
    ):
        self.show_video = show_video
        self.show_graph = show_graph
        self.patch_size = patch_size
        self.binary = binary
        if show_graph:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection="3d")
            self.x, self.y = np.meshgrid(np.arange(0, W), np.arange(0, H))
        self.all_timer = fps(average_over=100)

    def process(self, img):
        dark_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dark_channel = cv2.erode(
            dark_channel, np.ones((self.patch_size, self.patch_size))
        )

        # increase contrast of the dark channel prior
        dark_channel = cv2.equalizeHist(dark_channel)

        # find transmission map
        transmission = dark_channel / 255.0
        transmission = cv2.blur(transmission, (self.patch_size, self.patch_size))
        transmission = cv2.resize(transmission, (img.shape[1], img.shape[0]))

        if self.show_video:
            # Visualize the dark channel prior besides the original image
            # cv2.imshow("transmission", transmission)
            cv2.imshow("dark_channel", dark_channel.astype(np.uint8))
            # cv2.imshow("original", img)

        if self.show_graph:
            # Display the dark channel prior in 3D
            self.ax.clear()
            self.ax.plot_surface(self.x, self.y, transmission, cmap="gray")
            plt.pause(0.00001)
        if self.binary:
            # create a binary mask of pixels that are above 255/3
            mask = transmission > 0.3
            # show mask
            mask = mask.astype(np.uint8) * 255
            # make the black part of the mask red
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask[:, :, 2] = 255
            cv2.imshow("mask", mask)
            return mask, dark_channel, transmission

        return None, dark_channel, transmission
