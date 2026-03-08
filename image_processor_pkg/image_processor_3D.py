import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage, Image, Joy
import cv2
import numpy as np
import torch
import time
from cv_bridge import CvBridge
from PIL import Image as PILImage
from ament_index_python.packages import get_package_share_directory
import os
# Matplotlib removed - visualization now done on surface laptop

# Import custom modules
from image_processor_pkg.AvoidNet.avoid_net import get_model
from image_processor_pkg.AvoidNet.dataset import SUIM_grayscale
from image_processor_pkg.AvoidNet.draw_obsticle import draw_red_squares
from image_processor_pkg.AvoidNet.trajectory import determain_trajectory
from image_processor_pkg.AvoidNet3D.hazer_depth_class import HazerDepth
from image_processor_pkg.AvoidNet3D.dehaze_class import DehazeClass
from image_processor_pkg.AvoidNet3D.depth_anything_processor import DepthAnythingProcessor
from std_msgs.msg import Bool

class ImageProcessor(Node):
    def __init__(self, arc, run_name, use_gpu=False, threshold=0.5, dehaze_t0=0.6, dehaze_atmos_factor=0.8, color_diff_scale=0.5,
                 target_fps=2.0):
        super().__init__('image_processor')
        self.bridge = CvBridge()
        self.threshold = threshold
        self.obstacle = 0

        # AI processing is disabled until enabled via /ai_processing_enable
        self.processing_enabled = False

        # Frame throttling - critical for preventing system overload
        self.target_fps = target_fps
        self.min_interval = 1.0 / target_fps
        self.last_process_time = 0.0
        self.is_processing = False

        # Dynamically get the model path using get_package_share_directory
        model_path = os.path.join(
            get_package_share_directory('image_processor_pkg'),
            'models',
            f"{arc}_{run_name}.pth"
        )

        # Model setup
        self.model = get_model(arc)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.model.to(self.device).eval()

        # Transformation setup
        self.image_transform = SUIM_grayscale.get_transform()

        # Depth processing setup
        self.depth_processor = DepthAnythingProcessor(encoder='vits', device=self.device, input_size=308)
        
        # Hazer depth model setup (will be initialized with first frame dimensions)
        self.hazer_model = None
        self.frame_height = None
        self.frame_width = None
        
        # Dehaze setup
        self.dehazer = DehazeClass(t0=dehaze_t0, atmospheric_light_estimation_factor=dehaze_atmos_factor)
        self.color_diff_scale = color_diff_scale
        
        # Subscription to the image topic (uncompressed AI feed from batman_camera_hw)
        # Use BestEffort QoS to match camera publisher
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.subscription = self.create_subscription(
            Image, '/local/cam1/camera/image_ai', self.image_callback, qos_profile)

        self.get_logger().info("Subscribed to /local/cam1/camera/image_ai")
        
        self.subscription_joy = self.create_subscription(
            Joy, '/joy', self.joy_callback, 10)

        # Subscribe to AI processing enable/disable from surface laptop / NVR node
        self.ai_enable_sub = self.create_subscription(
            Bool, '/ai_processing_enable', self.ai_enable_callback, 10)

        # Publisher for the processed image (use BestEffort QoS for RViz compatibility)
        processed_image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # Always publish compressed annotated frame
        self.processed_image_publisher = self.create_publisher(
            CompressedImage, '/ai/annotated_frame/compressed', processed_image_qos)

        # Publish depth map as compressed image for surface visualization
        self.depth_image_publisher = self.create_publisher(
            CompressedImage, '/ai/depth_map/compressed', processed_image_qos)

        # Publish obstacle grid as compressed image
        self.obstacle_grid_publisher = self.create_publisher(
            CompressedImage, '/ai/obstacle_grid/compressed', processed_image_qos)

        self.joy_manipulation = self.create_publisher(Joy, '/joy', 80)

        # Subscribe to enable/disable control from NVR node (triggered by RViz AI Process button)
        self.create_subscription(
            Bool, '/ai_processing_enable', self.ai_enable_callback, 10)

        self.get_logger().info(f"Node initialized with model: {arc} on {self.device} - 3D version with depth processing")
        self.get_logger().info("Waiting for /ai_processing_enable to start processing")

    def ai_enable_callback(self, msg):
        was_enabled = self.processing_enabled
        self.processing_enabled = msg.data
        if self.processing_enabled != was_enabled:
            self.get_logger().info(f"AI processing {'enabled' if self.processing_enabled else 'disabled'}")

    def image_callback(self, msg):
        # Skip if AI processing is not enabled
        if not self.processing_enabled:
            return
        # Throttle: skip frames if processing too fast or already processing
        current_time = time.time()
        if self.is_processing or (current_time - self.last_process_time) < self.min_interval:
            return  # Skip this frame to prevent backlog

        self.is_processing = True
        self.last_process_time = current_time

        try:
            # Convert ROS Image message to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Initialize hazer model on first frame
            if self.hazer_model is None:
                self.frame_height = frame.shape[0]
                self.frame_width = frame.shape[1]
                self.hazer_model = HazerDepth(show_video=False, show_graph=False, patch_size=1,
                                            H=self.frame_height, W=self.frame_width, binary=False)

        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {str(e)}")
            self.is_processing = False
            return

        try:
            t0 = time.time()

            # Pre-process the frame for model inference
            frame_tensor = PILImage.fromarray(frame)
            frame_tensor = self.image_transform(frame_tensor).to(self.device).unsqueeze(0)
            t1 = time.time()

            # Run AvoidNet inference
            outputs = self.model(frame_tensor)
            outputs = outputs.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
            t2 = time.time()

            # Dehaze processing
            atmospheric_light = self.dehazer.estimate_atmospheric_light(frame)
            color_difference_map = self.dehazer.get_color_difference_map(frame, atmospheric_light)
            color_difference_map_resized = cv2.resize(color_difference_map, (outputs.shape[1], outputs.shape[0]))
            color_difference_map_resized *= self.color_diff_scale
            outputs[:, :, 0] += color_difference_map_resized
            outputs = np.clip(outputs, 0, 1)
            t3 = time.time()

            # Draw obstacles on frame
            frame = draw_red_squares(frame, outputs, self.threshold)

            # Depth processing (DepthAnything - heavy!)
            frame_rgb_pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            depth_pil = self.depth_processor.infer_pil(frame_rgb_pil)
            depth_np = np.array(depth_pil)
            t4 = time.time()

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

            # Publish annotated frame as compressed JPEG
            stamp = self.get_clock().now().to_msg()
            _, jpeg_data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_msg = CompressedImage()
            frame_msg.header.stamp = stamp
            frame_msg.format = 'jpeg'
            frame_msg.data = jpeg_data.tobytes()
            self.processed_image_publisher.publish(frame_msg)

            # Publish depth map as grayscale image (normalized 0-255)
            depth_uint8 = (depth_np * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)
            _, depth_jpeg = cv2.imencode('.jpg', depth_colored, [cv2.IMWRITE_JPEG_QUALITY, 75])
            depth_msg = CompressedImage()
            depth_msg.header.stamp = stamp
            depth_msg.format = 'jpeg'
            depth_msg.data = depth_jpeg.tobytes()
            self.depth_image_publisher.publish(depth_msg)

            # Publish obstacle grid as heatmap
            obstacle_heatmap = (outputs[:, :, 0] * 255).astype(np.uint8)
            obstacle_resized = cv2.resize(obstacle_heatmap, (frame.shape[1], frame.shape[0]))
            obstacle_colored = cv2.applyColorMap(obstacle_resized, cv2.COLORMAP_HOT)
            _, obstacle_jpeg = cv2.imencode('.jpg', obstacle_colored, [cv2.IMWRITE_JPEG_QUALITY, 75])
            obstacle_msg = CompressedImage()
            obstacle_msg.header.stamp = stamp
            obstacle_msg.format = 'jpeg'
            obstacle_msg.data = obstacle_jpeg.tobytes()
            self.obstacle_grid_publisher.publish(obstacle_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_processing = False

    def joy_callback(self, msg):
        pass

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
        use_gpu=True,
        threshold=0.3,
        dehaze_t0=0.6,
        dehaze_atmos_factor=0.8,
        color_diff_scale=0.4,
        target_fps=15.0  # Testing without dehaze
    )
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
