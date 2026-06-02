# predict.py
import os
import torch
import numpy as np
from model import RobotGestureResNet
from pose_extractor import extract_gesture_matrix_from_image

class RobotPredictionEngine:
    def __init__(self, weights_path='mother_model_weights.pth'):
        """
        Initializes the trained model structure and maps the labels 
        exactly how your dataset loader cataloged them.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hardcoded class mapping based directly on your training logs
        self.class_mapping = {
            0: 'OpenPalm', 
            1: 'ShoulderPain', 
            2: 'StomachPain', 
            3: 'ThumbDown', 
            4: 'ThumbUp', 
            5: 'headache', 
            6: 'neckpain'
        }
        
        # 1. Reconstruct the neural network architecture frame
        self.model = RobotGestureResNet(num_classes=len(self.class_mapping)).to(self.device)
        
        # 2. Load the optimized training weights
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file '{weights_path}' not found! Please run training first.")
            
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()  # Set layers to evaluation mode (turns off dropout, batchnorm training)
        print(f"🤖 Prediction Engine successfully loaded weights from '{weights_path}' onto {self.device}.")

    def predict_gesture(self, image_path, confidence_threshold=0.60):
        """
        Takes a new raw image file, processes the skeleton geometry, 
        and predicts the patient's gesture.
        """
        if not os.path.exists(image_path):
            return {"status": "error", "message": "Image path does not exist."}
            
        # 1. Process the raw image into our 30x34 spatio-temporal tracking matrix
        matrix = extract_gesture_matrix_from_image(image_path)
        
        # 2. Reshape to match the exact dimensional tensor input expected by PyTorch: (Batch, Channel, Height, Width)
        matrix_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 3. Pass data through the model without calculating training gradients
        with torch.no_grad():
            outputs = self.model(matrix_tensor)
            
            # Convert raw model outputs (logits) into clean, readable probabilities (0% to 100%)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Find the index with the highest probability value
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        conf_score = confidence.item()
        pred_class = self.class_mapping[predicted_idx.item()]
        
        # 4. Fallback filter for high safety contexts (Elderly Care)
        # If the model is completely guessing or confused, reject the prediction to prevent a false notification.
        if conf_score < confidence_threshold:
            return {
                "status": "uncertain",
                "prediction": "Unknown / Unclear Gesture",
                "confidence": conf_score,
                "raw_probabilities": {self.class_mapping[i]: float(prob) for i, prob in enumerate(probabilities)}
            }
            
        return {
            "status": "success",
            "prediction": pred_class,
            "confidence": conf_score,
            "is_emergency": "Pain" in pred_class or "head" in pred_class
        }
    def predict_live_frame(self, cv2_frame, confidence_threshold=0.50):
        """
        Accepts a raw frame matrix directly from a live webcam feed stream,
        extracts the keypoint matrix, and evaluates the prediction.
        """
        from ultralytics import YOLO
        import cv2
        import config
        import numpy as np

        # Initialize the model internally if not already running globally
        if not hasattr(self, 'pose_model'):
            self.pose_model = YOLO('yolov8n-pose.pt')

        results = self.pose_model(cv2_frame, verbose=False)
        
        # Initialize a clean timeline array matching config frames
        sequence_data = np.zeros((config.MAX_FRAMES, 17, 2), dtype=np.float32)
        
        if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kp = results[0].keypoints.xy[0].cpu().numpy()
            if kp.shape[0] == 17:
                # FIX: Broadcast keypoints to ALL frame rows in the matrix 
                # This perfectly mimics the static signature your model trained on!
                sequence_data[:] = kp

        # Package frame matrix array
        matrix = sequence_data.reshape(config.MAX_FRAMES, -1)
        max_val = np.max(matrix) if np.max(matrix) > 0 else 1.0
        matrix = matrix / max_val
        
        matrix_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(matrix_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        conf_score = confidence.item()
        
        # Lowered threshold slightly for a smoother live tracking response
        if conf_score < confidence_threshold:
            return "Unclear"
            
        return self.class_mapping[predicted_idx.item()]

# ==============================================================
# TEST RUNNING CODE
# ==============================================================
if __name__ == "__main__":
    # Initialize the system
    engine = RobotPredictionEngine(weights_path='mother_model_weights.pth')
    
    # Put a test image path here to try a manual prediction
    TEST_IMAGE = "path/to/a/new/unseen/test_picture.jpg" 
    
    if os.path.exists(TEST_IMAGE):
        result = engine.predict_gesture(TEST_IMAGE)
        print("\n--- Model Prediction Result ---")
        print(f"Detected Action: {result['prediction']}")
        print(f"Confidence Score: {result['confidence'] * 100:.2f}%")
        print(f"Emergency Alert Triggered: {result.get('is_emergency', False)}")
    else:
        print(f"\nSetup complete! To run a manual test, place an image file at: {TEST_IMAGE}")