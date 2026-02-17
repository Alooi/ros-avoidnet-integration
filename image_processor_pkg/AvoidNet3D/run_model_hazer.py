import cv2
import numpy as np
import time
from avoid_net import get_model
import argparse
import torch
from dataset import SUIM, SUIM_grayscale
from PIL import Image
import numpy as np
from draw_obsticle import draw_red_squares
from trajectory import determain_trajectory
from hazer_depth_class import HazerDepth
import sys
from dehaze_class import DehazeClass
import os
from datetime import datetime


def run_model(arc, run_name, source, video_path=None, use_gpu=False, save_video=False, que=False, show_transmission_heatmap=False, show_nn_heatmap=False, transmission_scale=0.5, dehaze_t0=0.6, dehaze_atmos_factor=0.8, color_diff_scale=0.5, show_frame=True, use_depth=False, show_3d_plot_live=False):
    # print the current selected parameters
    print(f"Architecture: {arc}")
    print(f"Run name: {run_name}")
    print(f"Source: {source}")
    print(f"Video path: {video_path}")
    print(f"Use GPU: {use_gpu}")
    print(f"Save video: {save_video}")
    print(f"Show transmission heatmap: {show_transmission_heatmap}")
    print(f"Show NN heatmap: {show_nn_heatmap}")
    print(f"Transmission scale: {transmission_scale}")
    print(f"Dehaze t0: {dehaze_t0}")
    print(f"Dehaze atmospheric light factor: {dehaze_atmos_factor}")
    print(f"Color difference scale: {color_diff_scale}")
    print(f"Show frame: {show_frame}")
    print(f"Use depth: {use_depth}")
    print(f"Show 3D plot live: {show_3d_plot_live}")
    if use_depth:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from depth_anything_processor import DepthAnythingProcessor
        if show_3d_plot_live:
            plt.ion()  # Enable interactive mode for real-time updates
    else:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    
    threshold = 0.3
    frame_start = 13700 # obstacles found at 13700 and 15200 and 16400
    if que:
        print("Running in quantized mode...")
        model = get_model(arc+'_q')
    else:
        model = get_model(arc)
    model.load_state_dict(torch.load(f"models/{arc}_{run_name}.pth"))

    DEVICE = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    print(f"Running on: {DEVICE}...")
    model.to(DEVICE).eval()

    if use_depth:
        depth_anything_processor = DepthAnythingProcessor(encoder='vits', device=DEVICE)
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(131) # Original frame
        ax2 = fig.add_subplot(132, projection='3d') # Depth map
        ax3 = fig.add_subplot(133, projection='3d') # Obstacles 3D
        if show_3d_plot_live:
            plt.show(block=False)
    

    dataset = SUIM_grayscale("")
    image_transform = dataset.get_transform()
    mask_transform = dataset.get_mask_transform()

    # read from source using opencv
    if source == "webcam":
        cap = cv2.VideoCapture(0)
        output_name = "webcam"

    elif source == "video":
        cap = cv2.VideoCapture(video_path)
        output_name = video_path.split("/")[-1].split(".")[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    size = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    hazer_model = HazerDepth(show_video=False, show_graph=False, patch_size=1, H=int(size[1]), W=int(size[0]), binary=False)
    save_size = (int(size[0]), int(size[1]))
    print(f"Video size: {int(size[0])}x{int(size[1])}")
    print(f"Video save_size: {save_size}")
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    dehazer = DehazeClass(t0=dehaze_t0, atmospheric_light_estimation_factor=dehaze_atmos_factor)

    frame_count = 0
    start_time = time.time()
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    if save_video:
        os.makedirs("results", exist_ok=True)
        out = cv2.VideoWriter(
            f"results/{output_name}_{current_time}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            frame_rate,
            save_size,
        )
        if save_video and use_depth:
            # Need to get the size of the matplotlib figure for video writer
            # Render a dummy frame to get the size
            fig.canvas.draw()
            plt.pause(0.001)  # Brief pause to allow GUI update
            width, height = fig.canvas.get_width_height()
            video_writer_3d = cv2.VideoWriter(
                f"results/{output_name}_3d_plot_{current_time}.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                frame_rate,
                (width, height)
            )
        else:
            video_writer_3d = None
    else:
        video_writer_3d = None
    frame_count = frame_start
    while True:
        start_time = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot fetch the frame.")
            break

        frame_count += 1
        
        # _, _, transmission_map = hazer_model.process(frame)

        # Estimate atmospheric light (updates history)
        # dehazer.estimate_atmospheric_light(frame) # This is now handled internally by dehaze
        # Get averaged atmospheric light from history
        # averaged_atmospheric_light = dehazer.get_averaged_atmospheric_light() # This is now handled internally by dehaze

        # Dehaze the frame using averaged atmospheric light
        atmospheric_light = dehazer.estimate_atmospheric_light(frame) # Estimate and average internally
        # frame = dehazer.dehaze(frame, transmission_map) # Now internal dehaze gets averaged atmospheric light

        # Estimate atmospheric light for color difference calculation
        # No longer needed to call here, as dehazer.estimate_atmospheric_light(frame) already updates history
        color_difference_map = dehazer.get_color_difference_map(frame, atmospheric_light)
        
        # prepare the frame for inference
        frame_tensor = Image.fromarray(frame)
        frame_tensor = image_transform(frame_tensor).to(DEVICE).unsqueeze(0)
        outputs = model(frame_tensor)
        
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time
        sys.stdout.write(f"\rFrame: {frame_count} | FPS: {fps:.2f}")
        sys.stdout.flush()

        # move the output tensors to cpu for visualization
        outputs = outputs.detach().cpu()
        outputs = outputs[0].permute(1, 2, 0)
        outputs = np.array(outputs)  # Convert to numpy array


        
        # Resize color_difference_map to match outputs shape and add to confidence
        color_difference_map_resized = cv2.resize(color_difference_map, (outputs.shape[1], outputs.shape[0]))
        color_difference_map_resized *= color_diff_scale
        
        if show_nn_heatmap:
            # Scale outputs[:, :, 0] to 0-255 and apply colormap
            nn_confidence = outputs[:, :, 0]
            nn_confidence_display = (nn_confidence * 255).astype(np.uint8)
            nn_heatmap = cv2.applyColorMap(nn_confidence_display, cv2.COLORMAP_JET)
            cv2.imshow("Neural Network Heatmap", nn_heatmap)
        
        # outputs[:, :, 0] += transmission_map_resized
        outputs[:, :, 0] += color_difference_map_resized
        outputs = np.clip(outputs, 0, 1) # Ensure values remain within valid range (0-1)
        
        # print(f"After addition & clip - Combined Confidence (outputs[:, :, 0]): min={outputs[:, :, 0].min():.4f}, max={outputs[:, :, 0].max():.4f}, mean={outputs[:, :, 0].mean():.4f}")
        # --- End Debugging prints ---

        # show the output
        frame = draw_red_squares(frame, outputs, threshold)

        if use_depth:
            frame_rgb_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            depth_pil = depth_anything_processor.infer_pil(frame_rgb_pil)
            depth_np = np.array(depth_pil)
            # normalize the values to between 0 and 1 for better visualization
            depth_np = (depth_np - np.nanmin(depth_np)) / (np.nanmax(depth_np) - np.nanmin(depth_np) + 1e-8) # Add small epsilon to avoid division by zero in case of uniform depth
            # invert the values
            depth_np = 1 - depth_np

            h, w = depth_np.shape
            grid_h, grid_w = outputs.shape[:2]
            cell_h, cell_w = h / grid_h, w / grid_w

            # Plotting for ax1 (Original Frame with Obstacles)
            ax1.clear()
            ax1.set_title("Original Frame with Obstacles")
            ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # Display the frame with red squares
            ax1.axis('off')

            # Plotting for ax2 (Depth Map 3D Surface)
            ax2.clear()
            ax2.set_title("Depth Map (3D Surface, Flipped)")
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            ax2.plot_surface(x, y, depth_np, cmap='viridis', edgecolor='none', alpha=0.7)
            ax2.set_xlabel("Width"); ax2.set_ylabel("Height"); ax2.set_zlabel("Depth")
            ax2.view_init(elev=290, azim=-90)  # Rotated 180 degrees along X-axis

            # Plotting for ax3 (Obstacles 3D Scatter)
            ax3.clear()
            ax3.set_title("Obstacles (Red Squares, 3D, Flipped)")
            xs, ys, zs = [], [], []
            for i in range(grid_h):
                for j in range(grid_w):
                    if outputs[i, j, 0] > threshold: # Use the same threshold as for draw_red_squares
                        y1, y2 = int(i * cell_h), int((i + 1) * cell_h)
                        x1, x2 = int(j * cell_w), int((j + 1) * cell_w)
                        cell_depth = np.nanmean(depth_np[y1:y2, x1:x2])
                        if not np.isnan(cell_depth):
                            xs.append((x1 + x2) / 2)
                            ys.append((y1 + y2) / 2)
                            zs.append(cell_depth)
            
            if xs and ys and zs: # Only plot if there are obstacles
                ax3.scatter(xs, ys, zs, color='red', s=100, edgecolors='black')
            ax3.set_xlabel("Width"); ax3.set_ylabel("Height"); ax3.set_zlabel("Depth")
            # Set limits for consistent view, using max/min of current frame depth
            min_d, max_d = np.nanmin(depth_np), np.nanmax(depth_np)
            ax3.set_xlim(0, w); ax3.set_ylim(0, h); ax3.set_zlim(min_d, max_d)
            ax3.view_init(elev=290, azim=-90)  # Rotated 180 degrees along X-axis

            fig.canvas.draw()
            if show_3d_plot_live:
                plt.pause(0.001)  # Brief pause to allow GUI update
            img_array_3d = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            width_3d, height_3d = fig.canvas.get_width_height()
            img_array_3d = img_array_3d.reshape(height_3d, width_3d, 3)
            img_array_3d_bgr = cv2.cvtColor(img_array_3d, cv2.COLOR_RGB2BGR)

            if video_writer_3d is not None:
                video_writer_3d.write(img_array_3d_bgr)

        if show_transmission_heatmap:
            # Scale transmission_map to 0-255 and apply colormap
            transmission_display = (color_difference_map_resized * 255).astype(np.uint8)
            transmission_heatmap = cv2.applyColorMap(transmission_display, cv2.COLORMAP_JET)
            cv2.imshow("Transmission Map Heatmap", transmission_heatmap)



        obstacle, new_trej = determain_trajectory(outputs, threshold=threshold)
        # put text on the to pleft corner of the frame
        if obstacle:
            cv2.putText(frame, f"Obstacle!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Turn {new_trej}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            # draw an arrow in the middle of the frame to indicate the direction
            # compute centers from the current frame shape so arrows remain correct after any resizing
            center_x = int(frame.shape[1] / 2)
            center_y = int(frame.shape[0] / 2)
            if new_trej == "left":
                cv2.arrowedLine(frame, (center_x, center_y), (center_x - 100, center_y), (0, 255, 255), 24)
            elif new_trej == "right":
                cv2.arrowedLine(frame, (center_x, center_y), (center_x + 100, center_y), (0, 255, 255), 24)
            elif new_trej == "up":
                cv2.arrowedLine(frame, (center_x, center_y), (center_x, center_y - 100), (0, 255, 255), 24)
        else:
            cv2.putText(frame, "Path Clear!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if show_frame:
            cv2.imshow("frame", frame)
        if save_video:
            out.write(frame)
            print("recording..", end="", flush=True)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if save_video:
        out.release()
        print()
        print(f"Video saved to results/{output_name}_{current_time}.mp4")
    if video_writer_3d is not None:
        video_writer_3d.release()
        print(f"3D Plot Video saved to results/{output_name}_3d_plot_{current_time}.mp4")
    if use_depth:
        plt.close(fig)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arc",
        type=str,
        default="ImageReducer_bounded_grayscale",
        help="Architecture to be used for training",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="run_2_1",
        help="Name of the run to be used for training",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="video",
        help="Source to be used for inference",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="/home/ali/codebases/V-DVL/input/canal_data/output_cam1.mp4",
        help="Path to the video to be used for inference",
    )
    parser.add_argument(
        "--use_gpu",
        type=bool,
        default=False,
        help="Use GPU for inference",
    )
    parser.add_argument(
        "--save_video",
        type=bool,
        default=False,
        help="Save the output video",
    )
    
    parser.add_argument(
        "--que",
        type=bool,
        default=False,
        help="Run in quantized mode",
    )
    parser.add_argument(
        "--show_transmission_heatmap",
        type=bool,
        default=False,
        help="Show the transmission map heatmap",
    )
    parser.add_argument(
        "--show_nn_heatmap",
        type=bool,
        default=False,
        help="Show the neural network confidence heatmap",
    )
    parser.add_argument(
        "--transmission_scale",
        type=float,
        default=0.5,
        help="Scaling factor for the transmission map before adding to confidence",
    )
    parser.add_argument(
        "--dehaze_t0",
        type=float,
        default=0.6,
        help="Minimum transmission value for dehazing. Smaller values make dehazing more aggressive.",
    )
    parser.add_argument(
        "--dehaze_atmos_factor",
        type=float,
        default=0.8,
        help="Factor for estimating atmospheric light in dehazing. Smaller values make dehazing more aggressive.",
    )
    parser.add_argument(
        "--color_diff_scale",
        type=float,
        default=0.4,
        help="Scaling factor for the color difference map before adding to confidence.",
    )
    parser.add_argument(
        "--show_frame",
        type=bool,
        default=False,
        help="Showing the actual frame with the obstacles"
    )
    parser.add_argument(
        "--use_depth",
        type=bool,
        default=False,
        help="Utilize depth processing for 3D visualization"
    )
    parser.add_argument(
        "--show_3d_plot_live",
        type=bool,
        default=False,
        help="Show 3D plot of obstacles and depth map live"
    )

    args = parser.parse_args()

    run_model(args.arc, args.run_name, args.source, args.video_path, args.use_gpu, args.save_video, args.que, args.show_transmission_heatmap, args.show_nn_heatmap, args.transmission_scale, args.dehaze_t0, args.dehaze_atmos_factor, args.color_diff_scale, args.show_frame, args.use_depth, args.show_3d_plot_live)

# example usage:
# python run_model.py --arc ImageReducer_bounded_grayscale --run_name run_2 --source video --video_path vlogs/output_2024-05-27_19-49-32.mp4 --use_gpu True --save_video False --que True
# run with video from youtube
# python run_model_hazer.py --save_video true --show_frame true --show_transmission_heatmap true --show_nn_heatmap true --video_path videoplayback.mp4