import cv2
import numpy as np
import collections # Import collections module

class DehazeClass:
    def __init__(self, t0=0.1, atmospheric_light_estimation_factor=0.1):
        """
        Initializes the DehazeClass.

        Args:
            t0 (float): Minimum transmission value to prevent division by zero or excessive noise.
                        Smaller values lead to more aggressive dehazing.
            atmospheric_light_estimation_factor (float): Factor for estimating atmospheric light
                                                         from the brightest pixels.
        """
        self.t0 = t0
        self.atmospheric_light_estimation_factor = atmospheric_light_estimation_factor
        self.atmospheric_light_history = collections.deque(maxlen=15) # Fixed window size to 10

    def estimate_atmospheric_light(self, image):
        """
        Estimates the atmospheric light from the hazy image.
        For aggressive dehazing, this can be simplified to taking the max intensity
        from the top X% brightest pixels.
        The estimated light for the current frame is stored in history, and the
        average of the history is returned.

        Returns:
            np.array: The averaged atmospheric light (BGR, 0-255 range).
        """
        # Convert image to grayscale to find brightest pixels
        # Ensure image is in BGR format for consistency
        if len(image.shape) == 2: # grayscale
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            gray = image
        else: # assuming BGR
            image_bgr = image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        flat_gray = gray.flatten()
        num_pixels = int(flat_gray.size * self.atmospheric_light_estimation_factor)
        
        # Handle case where num_pixels is 0 or flat_gray is empty
        if num_pixels == 0 or flat_gray.size == 0:
            current_atmospheric_light = np.array([255., 255., 255.]) # Default white
        else:
            # Get the indices of the brightest pixels
            indices = np.argsort(flat_gray)[-num_pixels:]
            
            # Calculate the average intensity of these brightest pixels in each channel
            # This is a simple approximation for atmospheric light A
            b, g, r = image_bgr[np.unravel_index(indices, image_bgr.shape[:2])].mean(axis=0)
            current_atmospheric_light = np.array([b, g, r])

        self.atmospheric_light_history.append(current_atmospheric_light)
        
        if not self.atmospheric_light_history:
            return np.array([255., 255., 255.]) # Default white if no history
        return np.mean(list(self.atmospheric_light_history), axis=0)


    def dehaze(self, frame, transmission_map):
        """
        Dehazes a single frame using the provided transmission map.
        Atmospheric light is estimated internally using the averaged history.

        Args:
            frame (np.array): The hazy input frame (BGR).
            transmission_map (np.array): The estimated transmission map (grayscale, 0-1).

        Returns:
            np.array: The dehazed frame (BGR).
        """
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32) / 255.0  # Normalize to 0-1

        # Ensure transmission map is float32 and has the same dimensions as frame (H, W)
        transmission_map_resized = cv2.resize(transmission_map, (frame.shape[1], frame.shape[0]))
        transmission_map_resized = np.clip(transmission_map_resized, self.t0, 1.0) # Clip with t0

        atmospheric_light_mean = self.estimate_atmospheric_light(frame * 255) # Re-estimate to update history and get average
        atmospheric_light_normalized = atmospheric_light_mean / 255.0 # Normalize A to 0-1

        # Reshape atmospheric_light_normalized for element-wise operation (1, 1, 3)
        A = atmospheric_light_normalized[np.newaxis, np.newaxis, :]

        # Dehazing formula: J = (I - A) / max(t, t0) + A
        # Expand transmission_map_resized to 3 channels for element-wise operation
        t_expanded = np.expand_dims(transmission_map_resized, axis=2)

        dehazed_frame = ((frame - A) / t_expanded) + A
        
        # Clip to ensure values are within [0, 1] and convert back to uint8
        dehazed_frame = np.clip(dehazed_frame * 255, 0, 255).astype(np.uint8)

        return dehazed_frame

    def get_color_difference_map(self, frame, atmospheric_light):
        """
        Calculates a map representing the Euclidean color distance of each pixel
        from the estimated atmospheric light (haze color).

        Args:
            frame (np.array): The hazy input frame (BGR).
            atmospheric_light (np.array): The estimated atmospheric light (BGR, 0-255 range).

        Returns:
            np.array: A grayscale map (0-1) where higher values indicate greater
                      color difference from the haze color.
        """
        if frame.dtype != np.float32:
            frame_float = frame.astype(np.float32)
        else:
            frame_float = frame.copy() # Ensure we have a writable copy if already float32

        # Reshape atmospheric_light for element-wise operation (1, 1, 3)
        A = atmospheric_light[np.newaxis, np.newaxis, :]

        # Calculate Euclidean distance in RGB space
        # (R_pixel - R_haze)^2 + (G_pixel - G_haze)^2 + (B_pixel - B_haze)^2
        color_diff_sq = np.sum((frame_float - A)**2, axis=2)
        color_diff_map = np.sqrt(color_diff_sq)

        # Normalize to 0-1 range
        max_diff = np.max(color_diff_map)
        if max_diff > 0:
            normalized_color_diff_map = color_diff_map / max_diff
        else:
            normalized_color_diff_map = np.zeros_like(color_diff_map)
            
        return normalized_color_diff_map
