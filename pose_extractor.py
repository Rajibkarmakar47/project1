# pose_extractor.py
import numpy as np
import cv2
from ultralytics import YOLO
import config

# Load the core tracking model
pose_model = YOLO('yolov8n-pose.pt')

def extract_gesture_matrix_from_image(image_path_or_frame):
    """
    Standardizes human skeleton tracking data using anatomical unit scaling.
    Uses the distance between shoulders as a stable scale factor.
    """
    if isinstance(image_path_or_frame, str):
        frame = cv2.imread(image_path_or_frame)
    else:
        frame = image_path_or_frame

    if frame is None:
        return np.zeros((config.MAX_FRAMES, 34), dtype=np.float32)
        
    results = pose_model(frame, verbose=False)
    sequence_data = np.zeros((config.MAX_FRAMES, 17, 2), dtype=np.float32)
    
    if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        kp = results[0].keypoints.xy[0].cpu().numpy()  # Shape: (17, 2)
        
        if kp.shape[0] == 17:
            left_shoulder = kp[5]
            right_shoulder = kp[6]
            
            # 1. Calculate a stable center point anchor (Midpoint of shoulders)
            neck_anchor = (left_shoulder + right_shoulder) / 2.0
            
            # 2. Calculate the physical shoulder width using the Euclidean Distance formula
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
            
            # If tracking glitches or width is zero, default to 1.0 to prevent division errors
            if shoulder_width == 0:
                shoulder_width = 1.0
                
            # 3. Apply the transformation: Center data, then scale strictly by body size
            # This completely preserves human shape proportions perfectly!
            normalized_kp = (kp - neck_anchor) / shoulder_width
            
            # Populate our matrix tracking block
            sequence_data[:] = normalized_kp

    # Collapse into our standard 2D matrix shape: (30, 34)
    spatio_temporal_map = sequence_data.reshape(config.MAX_FRAMES, -1)
    
    return spatio_temporal_map.astype(np.float32)