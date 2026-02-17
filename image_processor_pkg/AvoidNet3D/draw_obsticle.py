import cv2
import numpy as np

def draw_red_squares(frame, grid, threshold):
    # get the size of the sqaures
    square_size_y = frame.shape[0] // grid.shape[0]
    square_size_x = frame.shape[1] // grid.shape[1]
    
    # get the indices of the squares that are above the threshold
    indices = np.where(grid > threshold)
    
    # draw the squares
    for i in range(len(indices[0])):
        y = indices[0][i]
        x = indices[1][i]
        frame = cv2.rectangle(
            frame,
            (x * square_size_x, y * square_size_y),
            ((x + 1) * square_size_x, (y + 1) * square_size_y),
            (0, 0, 255),
            2,
        )

    

    return frame
