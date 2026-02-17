# init a video saver class

import cv2
import numpy as np
import os


class VideoSaver:
    def __init__(self, path, fps, size):
        self.path = path
        # check if path exists if not create one
        split_path = self.path.split("/")
        split_path = split_path[:-1]
        # check if split_path exists if not create one
        if not os.path.exists("/".join(split_path)):
            os.makedirs("/".join(split_path))
        self.fps = fps
        self.size = size
        # as mp4
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(self.path, self.fourcc, self.fps, self.size)

    def write(self, frame):
        self.out.write(frame)

    def release(self):
        self.out.release()
