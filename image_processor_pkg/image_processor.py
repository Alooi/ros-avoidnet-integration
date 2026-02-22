import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
import cv2
import numpy as np
import torch
import time
from cv_bridge import CvBridge
from PIL import Image as PILImage
from ament_index_python.packages import get_package_share_directory
import os

# Import custom modules
from image_processor_pkg.AvoidNet.avoid_net import get_model
from image_processor_pkg.AvoidNet.dataset import SUIM_grayscale
from image_processor_pkg.AvoidNet.draw_obsticle import draw_red_squares
from image_processor_pkg.AvoidNet.trajectory import determain_trajectory

class ImageProcessor(Node):
    def __init__(self, arc, run_name, use_gpu=False, threshold=0.9, target_fps=5.0, enable_viz=False):
        super().__init__('image_processor')
        self.bridge = CvBridge()
        self.threshold = threshold
        self.enable_viz = enable_viz  # Disable heavy visualization by default

        # Frame throttling - only process at target_fps to avoid CPU overload
        self.target_fps = target_fps
        self.min_interval = 1.0 / target_fps  # seconds between frames
        self.last_process_time = 0.0
        self.is_processing = False  # Prevent re-entry while processing

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

        # QoS profile matching the camera node's AI topic (BEST_EFFORT, volatile, depth 1)
        ai_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribe to the uncompressed AI topic (not the H.264 compressed "image_raw")
        # The camera node publishes uncompressed BGR images specifically for AI processing
        self.subscription = self.create_subscription(
            Image, '/local/cam1/camera/image_raw_uncompressed', self.image_callback, ai_qos)

        # Publisher for the processed image - use BEST_EFFORT to prevent network choking
        self.processed_image_publisher = self.create_publisher(
            CompressedImage, 'processed_image_topic', ai_qos)

        self.get_logger().info(f"Node initialized with model: {arc} on {self.device}, throttled to {target_fps} FPS")

    def image_callback(self, msg):
        # Throttle: skip frames if processing too fast or already processing
        current_time = time.time()
        if self.is_processing or (current_time - self.last_process_time) < self.min_interval:
            return  # Skip this frame

        self.is_processing = True
        self.last_process_time = current_time

        try:
            # Convert ROS Image message to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # downscale to 640x480
            frame = cv2.resize(frame, (640, 480))

            # Pre-process the frame
            frame_tensor = PILImage.fromarray(frame)
            frame_tensor = self.image_transform(frame_tensor).to(self.device).unsqueeze(0)

            # Run inference
            outputs = self.model(frame_tensor)
            outputs = outputs.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()

            # Get trajectory decision (lightweight)
            obstacle, new_trej = determain_trajectory(outputs, threshold=self.threshold)

            # Log result instead of drawing (much lighter than visualization)
            if obstacle:
                self.get_logger().debug(f"Obstacle detected, turn {new_trej}")

            # Only do heavy visualization if enabled
            if self.enable_viz:
                # Process model output - draw obstacles
                frame = draw_red_squares(frame, outputs, self.threshold)

                # Display obstacle information
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
                else:
                    cv2.putText(frame, "Path Clear!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Resize the frame for display
                frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

                # Convert to compressed JPEG to reduce network bandwidth
                _, jpeg_data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                processed_msg = CompressedImage()
                processed_msg.header.stamp = self.get_clock().now().to_msg()
                processed_msg.format = 'jpeg'
                processed_msg.data = jpeg_data.tobytes()
                self.processed_image_publisher.publish(processed_msg)
        finally:
            self.is_processing = False

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor(
        arc="ImageReducer_bounded_grayscale",
        run_name="run_2",
        use_gpu=True,  # Use Jetson Orin GPU for inference
        target_fps=10.0,  # Process 10 frames per second
        enable_viz=False  # Disable visualization to reduce CPU load
    )
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

