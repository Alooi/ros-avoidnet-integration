import numpy as np


def determain_trajectory(grid, threshold=0.5):
    """
    Determine the trajectory of the object in the grid
    """
    
    flag = False
    new_trej = None

    search_area = 5

    indices = np.where(grid > threshold)

    center = (grid.shape[1] // 2, grid.shape[0] // 2)

    # see if any of the squares in the center of the grid plus search area are above the threshold
    center_area = grid[center[1] - search_area : center[1] + search_area, center[0] - search_area : center[0] + search_area]
    if np.any(center_area > threshold):
        flag = True
    
    # search up, left and right of the center for a clear path and set the new trajectory
    if flag:
        # get the average of all values of the grid above the centerr area
        up = np.mean(grid[:center[1], center[0] - search_area : center[0] + search_area])
        # get the average of all values of the grid to the left of the centerr area
        left = np.mean(grid[center[1] - search_area : center[1] + search_area, :center[0]])
        # get the average of all values of the grid to the right of the centerr area
        right = np.mean(grid[center[1] - search_area : center[1] + search_area, center[0]:])

        # find the lowest average and set the new trajectory
        if up < left and up < right:
            new_trej = "up"
        elif left < up and left < right:
            new_trej = "left"
        elif right < up and right < left:
            new_trej = "right"
        else:
            new_trej = "up"
        
        
    return flag, new_trej
