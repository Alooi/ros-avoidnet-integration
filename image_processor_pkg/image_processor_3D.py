import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, Joy
import cv2
import numpy as np
import torch
from cv_bridge import CvBridge
from PIL import Image as PILImage
from ament_index_python.packages import get_package_share_directory
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless operation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import custom modules
from image_processor_pkg.AvoidNet.avoid_net import get_model
from image_processor_pkg.AvoidNet.dataset import SUIM_grayscale
from image_processor_pkg.AvoidNet.draw_obsticle import draw_red_squares
from image_processor_pkg.AvoidNet.trajectory import determain_trajectory
from image_processor_pkg.AvoidNet3D.hazer_depth_class import HazerDepth
from image_processor_pkg.AvoidNet3D.dehaze_class import DehazeClass
from image_processor_pkg.AvoidNet3D.depth_anything_processor import DepthAnythingProcessor

class ImageProcessor(Node):
    def __init__(self, arc, run_name, use_gpu=False, threshold=0.5, dehaze_t0=0.6, dehaze_atmos_factor=0.8, color_diff_scale=0.5):
        super().__init__('image_processor')
        self.bridge = CvBridge()
        self.threshold = threshold
        self.obstacle = 0

        # Dynamically get the model path using get_package_share_directory
        model_path = os.path.join(
            get_package_share_directory('image_processor_pkg'),
            'models',
            f"{arc}_{run_name}.pth"
        )

        # Model setup
        self.model = get_model(arc)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.model.to(self.device).eval()

        # Transformation setup
        self.image_transform = SUIM_grayscale.get_transform()

        # Depth processing setup
        self.depth_processor = DepthAnythingProcessor(encoder='vits', device=self.device)
        
        # Hazer depth model setup (will be initialized with first frame dimensions)
        self.hazer_model = None
        self.frame_height = None
        self.frame_width = None
        
        # Dehaze setup
        self.dehazer = DehazeClass(t0=dehaze_t0, atmospheric_light_estimation_factor=dehaze_atmos_factor)
        self.color_diff_scale = color_diff_scale
        
        # 3D visualization setup
        self.fig = plt.figure(figsize=(20, 10))
        self.ax1 = self.fig.add_subplot(131)  # Original frame
        self.ax2 = self.fig.add_subplot(132, projection='3d')  # Depth map
        self.ax3 = self.fig.add_subplot(133, projection='3d')  # Obstacles 3D

        # Subscription to the image topic
        self.subscription = self.create_subscription(
            Image, '/cam1/camera/image_raw', self.image_callback, 10)
        
        self.get_logger().info("Subscribed to /cam1/camera/image_raw")
        
        # Add a counter to track received images
        self.image_count = 0
        
        self.subscription_joy = self.create_subscription(
            Joy, '/joy', self.joy_callback, 10)

        # Publisher for the processed image
        self.processed_image_publisher = self.create_publisher(Image, 'processed_image_topic', 80)
        self.joy_manipulation = self.create_publisher(Joy, '/joy', 80)

        
        self.get_logger().info(f"Node initialized with model: {arc} on {self.device} - 3D version with depth processing")

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_count += 1
            
            # Initialize hazer model on first frame
            if self.hazer_model is None:
                self.frame_height = frame.shape[0]
                self.frame_width = frame.shape[1]
                self.hazer_model = HazerDepth(show_video=False, show_graph=False, patch_size=1, 
                                            H=self.frame_height, W=self.frame_width, binary=False)
            
            # Log every 10th image to avoid spam
            if self.image_count % 10 == 0:
                self.get_logger().info(f"Processed {self.image_count} images. Current: {frame.shape[1]}x{frame.shape[0]} pixels")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {str(e)}")
            return

        try:
            # Pre-process the frame for model inference
            frame_tensor = PILImage.fromarray(frame)
            frame_tensor = self.image_transform(frame_tensor).to(self.device).unsqueeze(0)

            # Run inference
            outputs = self.model(frame_tensor)
            outputs = outputs.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()

            # Dehaze processing
            atmospheric_light = self.dehazer.estimate_atmospheric_light(frame)
            color_difference_map = self.dehazer.get_color_difference_map(frame, atmospheric_light)
            
            # Resize color_difference_map to match outputs shape
            color_difference_map_resized = cv2.resize(color_difference_map, (outputs.shape[1], outputs.shape[0]))
            color_difference_map_resized *= self.color_diff_scale
            
            # Add color difference to model confidence
            outputs[:, :, 0] += color_difference_map_resized
            outputs = np.clip(outputs, 0, 1)

            # Draw obstacles on frame
            frame = draw_red_squares(frame, outputs, self.threshold)

            # Depth processing
            frame_rgb_pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            depth_pil = self.depth_processor.infer_pil(frame_rgb_pil)
            depth_np = np.array(depth_pil)
            
            # Normalize depth values to 0-1
            depth_np = (depth_np - np.nanmin(depth_np)) / (np.nanmax(depth_np) - np.nanmin(depth_np) + 1e-8)
            depth_np = 1 - depth_np  # Invert for visualization

            # Trajectory determination
            obstacle, new_trej = determain_trajectory(outputs, threshold=self.threshold)

            # Add trajectory information to frame
            if obstacle:
                cv2.putText(frame, "Obstacle!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Turn {new_trej}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                h, w = frame.shape[:2]
                if new_trej == "left":
                    cv2.arrowedLine(frame, (w//2, h//2), (w//2 - 100, h//2), (0, 255, 255), 2)
                elif new_trej == "right":
                    cv2.arrowedLine(frame, (w//2, h//2), (w//2 + 100, h//2), (0, 255, 255), 2)
                elif new_trej == "up":
                    cv2.arrowedLine(frame, (w//2, h//2), (w//2, h//2 - 100), (0, 255, 255), 2)
                
                self.obstacle = 1
            else:
                cv2.putText(frame, "Path Clear!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                self.obstacle = 0

            # Generate 3D visualization
            plot_image = self._generate_3d_plot(frame, depth_np, outputs)
            
            # Publish the 3D plot as the processed image
            processed_msg = self.bridge.cv2_to_imgmsg(plot_image, encoding='bgr8')
            self.processed_image_publisher.publish(processed_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {str(e)}")
            import traceback
            traceback.print_exc()

    def _generate_3d_plot(self, frame, depth_np, outputs):
        """Generate 3D visualization and convert to image array"""
        try:
            h, w = depth_np.shape
            grid_h, grid_w = outputs.shape[:2]
            cell_h, cell_w = h / grid_h, w / grid_w

            # Clear previous plots
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()

            # Plotting for ax1 (Original Frame with Obstacles)
            self.ax1.set_title("Original Frame with Obstacles")
            self.ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.ax1.axis('off')

            # Plotting for ax2 (Depth Map 3D Surface)
            self.ax2.set_title("Depth Map (3D Surface, Flipped)")
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            self.ax2.plot_surface(x, y, depth_np, cmap='viridis', edgecolor='none', alpha=0.7)
            self.ax2.set_xlabel("Width")
            self.ax2.set_ylabel("Height")
            self.ax2.set_zlabel("Depth")
            self.ax2.view_init(elev=290, azim=-90)

            # Plotting for ax3 (Obstacles 3D Scatter)
            self.ax3.set_title("Obstacles (Red Squares, 3D, Flipped)")
            xs, ys, zs = [], [], []
            for i in range(grid_h):
                for j in range(grid_w):
                    if outputs[i, j, 0] > self.threshold:
                        y1, y2 = int(i * cell_h), int((i + 1) * cell_h)
                        x1, x2 = int(j * cell_w), int((j + 1) * cell_w)
                        cell_depth = np.nanmean(depth_np[y1:y2, x1:x2])
                        if not np.isnan(cell_depth):
                            xs.append((x1 + x2) / 2)
                            ys.append((y1 + y2) / 2)
                            zs.append(cell_depth)
            
            if xs and ys and zs:  # Only plot if there are obstacles
                self.ax3.scatter(xs, ys, zs, color='red', s=100, edgecolors='black')
            
            self.ax3.set_xlabel("Width")
            self.ax3.set_ylabel("Height")
            self.ax3.set_zlabel("Depth")
            
            # Set limits for consistent view
            min_d, max_d = np.nanmin(depth_np), np.nanmax(depth_np)
            self.ax3.set_xlim(0, w)
            self.ax3.set_ylim(0, h)
            self.ax3.set_zlim(min_d, max_d)
            self.ax3.view_init(elev=290, azim=-90)

            # Draw the figure and convert to image array
            self.fig.canvas.draw()
            img_array = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            width, height = self.fig.canvas.get_width_height()
            img_array = img_array.reshape(height, width, 3)
            img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            return img_array_bgr

        except Exception as e:
            self.get_logger().error(f"Error generating 3D plot: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a blank frame if plot generation fails
            return np.zeros((480, 640, 3), dtype=np.uint8)


    def joy_callback(self, msg):
        # Print the received joystick message
        print(msg)

        # # Check if button 4 is pressed (indexing starts at 0)
        # if len(msg.buttons) > 4 and msg.buttons[4]:
        #     # Create a copy of the message and modify axes to pitch up
        #     modified_msg = Joy()
        #     modified_msg.header = msg.header
        #     modified_msg.axes = list(msg.axes)
        #     modified_msg.buttons = list(msg.buttons)

        #     # Ensure there is a second axis for pitch
        #     if len(modified_msg.axes) < 2:
        #         modified_msg.axes += [0.0] * (2 - len(modified_msg.axes))

        #     # Set pitch up command
        #     modified_msg.axes[1] = 1.0

        #     # Publish the modified message
        #     self.joy_manipulation.publish(modified_msg)
        return



def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor(
        arc="ImageReducer_bounded_grayscale",
        run_name="run_2_1",
        use_gpu=False,
        threshold=0.3,
        dehaze_t0=0.6,
        dehaze_atmos_factor=0.8,
        color_diff_scale=0.4
    )
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
