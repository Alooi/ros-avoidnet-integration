import torch
from PIL import Image
import cv2
import numpy as np

# --- Model and Device Setup ---
from depth_anything_processor import DepthAnythingProcessor
from frame_red_squares import RedSquaresGrid

import matplotlib
# <<< CHANGE 1: Import the main matplotlib library to set the backend
matplotlib.use('Agg') # Use a non-interactive backend
# ---
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

class ObstacleDetector:
    def __init__(self, encoder='vits', red_squares_arc="ImageReducer_bounded_grayscale", red_squares_run_name="run_2", use_gpu=True):
        """
        Initializes the obstacle detection system.

        Args:
            encoder (str): The encoder to use for the Depth Anything model.
            red_squares_arc (str): The architecture for the RedSquaresGrid model.
            red_squares_run_name (str): The run name for the RedSquaresGrid model.
            use_gpu (bool): Whether to use GPU if available.
        """
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print(f"Using device: {self.device}")

        self.depth_processor = DepthAnythingProcessor(encoder=encoder, device=self.device)
        self.obstacle_grid_model = RedSquaresGrid(arc=red_squares_arc, run_name=red_squares_run_name, use_gpu=use_gpu, threshold=0.2)
        
        self.target_size = (640, 480)

    def process_frame(self, frame):
        """
        Processes a single video frame to detect obstacles and estimate their depth.

        Args:
            frame (np.ndarray): The input video frame in BGR format (from OpenCV).

        Returns:
            np.ndarray: A 2D grid where each cell contains the average depth of a detected obstacle.
                        Non-obstacle cells will have a value of np.nan.
        """
        # 1. Preprocess the frame
        frame_resized = cv2.resize(frame, self.target_size)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # 2. Get depth map and obstacle grid
        depth_pil = self.depth_processor.infer_pil(image)
        depth_np = np.array(depth_pil)
        # Flip the depth inside out
        min_d, max_d = np.nanmin(depth_np), np.nanmax(depth_np)
        flipped_depth_np = max_d + min_d - depth_np
        output_obstacles = self.obstacle_grid_model(frame_resized)

        # 3. Combine depth and obstacle information
        h, w = flipped_depth_np.shape
        grid_h, grid_w = output_obstacles.shape[:2]
        cell_h, cell_w = h / grid_h, w / grid_w

        obstacle_depth_map = np.full((grid_h, grid_w), np.nan, dtype=np.float32)

        for i in range(grid_h):
            for j in range(grid_w):
                # Check if the cell is an obstacle (assuming channel 0 has the obstacle probability)
                if output_obstacles[i, j, 0] > 0.4:
                    # Define the region in the depth map corresponding to this grid cell
                    y1, y2 = int(i * cell_h), int((i + 1) * cell_h)
                    x1, x2 = int(j * cell_w), int((j + 1) * cell_w)
                    
                    # Extract the depth values for the cell
                    depth_cell = flipped_depth_np[y1:y2, x1:x2]
                    
                    # Only compute mean if the region is non-empty
                    if depth_cell.size > 0:
                        cell_depth = np.nanmean(depth_cell)
                        if not np.isnan(cell_depth):
                            obstacle_depth_map[i, j] = cell_depth
        
        return obstacle_depth_map
    
    def adjust_depth_to_middle_pixel(self, pred_depth, gt_depth):
        h, w = pred_depth.shape
        pred_mid = pred_depth[h // 2, w // 2]
        if pred_mid == 0:
            return pred_depth  # Avoid division by zero
        scale = gt_depth / pred_mid
        return pred_depth * scale
    
    def visualize_obstacles(self, frame, obstacle_depth_map, fig, ax):
        """
        Visualizes the full depth map and obstacles in a live 3D plot, and returns it as an OpenCV image.

        Args:
            frame (np.ndarray): The input video frame in BGR format.
            obstacle_depth_map (np.ndarray): The 2D grid of obstacle depths.
            fig (matplotlib.figure.Figure): The figure for plotting.
            ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axes for plotting.

        Returns:
            np.ndarray: The visualization as an OpenCV-compatible BGR image.
        """
        # 1. Prepare data for visualization
        frame_resized = cv2.resize(frame, self.target_size)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # Get the full depth map for the background
        depth_pil = self.depth_processor.infer_pil(Image.fromarray(frame_rgb))
        depth_np = np.array(depth_pil)
        min_d, max_d = np.nanmin(depth_np), np.nanmax(depth_np)
        flipped_depth_np = max_d + min_d - depth_np

        # Calculate a small offset to lift obstacle dots above the surface
        z_offset = 0
        if not np.all(np.isnan(flipped_depth_np)):
            depth_range = np.nanmax(flipped_depth_np) - np.nanmin(flipped_depth_np)
            z_offset = depth_range * 0.02  # 2% of depth range

        grid_h, grid_w = obstacle_depth_map.shape
        cell_h, cell_w = h / grid_h, w / grid_w

        # 2. Clear and redraw the plot
        ax.clear()
        ax.set_title("Live 3D Depth Map with Obstacles")

        # Subsample the depth map for performance
        depth_subsampled = flipped_depth_np[::10, ::10]
        
        # Get the shape of the *actual data* we are plotting
        zh, zw = depth_subsampled.shape
        
        # Create coordinate grids that match the subsampled depth map's dimensions
        x_coords = np.linspace(0, w - 1, zw)
        y_coords = np.linspace(0, h - 1, zh)
        x, y = np.meshgrid(x_coords, y_coords)

        ax.plot_surface(x, y, depth_subsampled, cmap='viridis', edgecolor='none', alpha=0.7)

        # Plot obstacles as red squares
        xs, ys, zs = [], [], []
        for i in range(grid_h):
            for j in range(grid_w):
                if not np.isnan(obstacle_depth_map[i, j]):
                    y1, y2 = int(i * cell_h), int((i + 1) * cell_h)
                    x1, x2 = int(j * cell_w), int((j + 1) * cell_w)
                    cell_depth = obstacle_depth_map[i, j]
                    xs.append((x1 + x2) / 2)
                    ys.append((y1 + y2) / 2)
                    zs.append(cell_depth - z_offset)

        if len(xs) > 0:
            ax.scatter(xs, ys, zs, color='red', s=50, edgecolors='black', depthshade=False)

        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        ax.set_zlabel("Depth")
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        if not np.all(np.isnan(flipped_depth_np)):
            ax.set_zlim(np.nanmin(flipped_depth_np), np.nanmax(flipped_depth_np) - z_offset)
        ax.view_init(elev=290, azim=-90)

        # 3. Convert Matplotlib plot to an OpenCV image
        fig.canvas.draw()
        img_buf = fig.canvas.buffer_rgba()
        plot_img = np.asarray(img_buf)
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        
        # Resize frame to match plot height for side-by-side view
        plot_h = plot_img.shape[0]
        frame_h, frame_w = frame_resized.shape[:2]
        new_w = int(plot_h * frame_w / frame_h)
        frame_display = cv2.resize(frame_resized, (new_w, plot_h))

        # Concatenate frame and plot
        img = cv2.hconcat([frame_display, plot_img])
        
        return img

if __name__ == '__main__':
    # Example usage:
    # This part demonstrates how to use the ObstacleDetector class.
    # It reads from a sample video, processes each frame, and provides a live visualization.

    # --- Configuration ---
    video_path = "recent_video.mp4"
    visualize = False  # Set to True to see the live visualization
    save_figure = True  # Set to True if you want to save the visualization images
    
    vis_image = None  # Initialize the visualization image variable

    # --- Initialization ---
    detector = ObstacleDetector()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        exit()

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video with {total_frames} frames...")

    # --- Initialization for Live Visualization ---
    if visualize or save_figure:
        # <<< CHANGE 2: Removed plt.ion() as it's not needed with the 'Agg' backend
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    # --- Processing Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame to get the obstacle map
        obstacle_map = detector.process_frame(frame)

        if visualize:
            # Generate the visualization image from the plot
            vis_image = detector.visualize_obstacles(frame, obstacle_map, fig, ax)
            
            # Display the combined visualization in an OpenCV window
            cv2.imshow('Obstacle Detection: Frame and 3D Visualization', vis_image)

            # Check for 'q' key to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # --- Example: Print the output for the first few frames ---
        if frame_count < 5:
            print(f"\n--- Frame {frame_count} ---")
            print("Obstacle Depth Map (NaN means no obstacle):")
            with np.printoptions(precision=2, suppress=True, nanstr="."):
                print(obstacle_map)
        
        frame_count += 1
        
        if (frame_count % 10) == 0:
            print(f"Processed {frame_count}/{total_frames} frames...", end='\r')
            
        if save_figure:
            if not visualize:
                vis_image = detector.visualize_obstacles(frame, obstacle_map, fig, ax)
            if 'video_writer' not in locals():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_height, out_width = vis_image.shape[:2]
                video_writer = cv2.VideoWriter('visualization_output.mp4', fourcc, 20.0, (out_width, out_height))
            video_writer.write(vis_image)

    # --- Cleanup ---
    cap.release()
    if visualize:
        cv2.destroyAllWindows()
        # <<< CHANGE 3: Removed plt.ioff()
        plt.close(fig) # Still good practice to close the figure object
    print(f"\nFinished processing all {frame_count} frames.")