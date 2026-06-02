# main_robot_control.py
import cv2
import time
from collections import Counter
from predict import RobotPredictionEngine

def trigger_medical_alert(illness_type):
    print(f"\n🚨 [ALERT SYSTEM] Sending notification: 'Patient has {illness_type}!'")

def say_to_patient(message):
    print(f"🔊 [ROBOT SPEAKER] Speaking out loud: '{message}'")

def main():
    # 1. Initialize our custom trained AI Engine brain
    robot_brain = RobotPredictionEngine(weights_path='mother_model_weights.pth')
    
    # 2. Boot up local webcam hardware stream
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ CRITICAL ERROR: Could not access webcam stream hardware.")
        return

    print("\n🤖 Live Video Analysis Online.")
    print("👉 Hold your gesture steady in front of the camera.")
    print("👉 Press the 'ESC' key on your keyboard inside the window to exit cleanly.")
    print("------------------------------------------------------------------\n")

    # Tracking variables for our 2-second observation chunks
    frame_predictions_buffer = []
    start_time = time.time()
    last_final_decision = "None"
    
    # NEW: Updated time window constant set to 2.0 seconds
    WINDOW_DURATION = 2.0 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab camera frame.")
            break

        # Flip frame horizontally for a more natural mirror view reflection
        frame = cv2.flip(frame, 1)
        
        # Step A: Run instant frame inference evaluation
        current_frame_prediction = robot_brain.predict_live_frame(frame, confidence_threshold=0.55)
        
        # Keep track of predictions (ignore 'Unclear' frames to protect metric accuracy)
        if current_frame_prediction != "Unclear":
            frame_predictions_buffer.append(current_frame_prediction)

        # Step B: Render visual HUD indicators on screen for the user
        elapsed_time = time.time() - start_time
        remaining_time = max(0.0, WINDOW_DURATION - elapsed_time)
        
        # Draw text directly on the camera frame matrix
        cv2.putText(frame, f"Observing Window: {remaining_time:.1f}s", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Last Decision: {last_final_decision}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Robot Monitoring System Perspective', frame)

        # Step C: Evaluate window block every 2 seconds
        if elapsed_time >= WINDOW_DURATION:
            print(f"--- {WINDOW_DURATION}-Second Window Complete. Evaluating {len(frame_predictions_buffer)} predictions... ---")
            
            if len(frame_predictions_buffer) > 0:
                # Calculate mathematical mode (the element that appeared most frequently)
                votes_counter = Counter(frame_predictions_buffer)
                most_frequent_gesture, vote_count = votes_counter.most_common(1)[0]
                
                last_final_decision = most_frequent_gesture
                print(f"🏆 Decision Result: '{most_frequent_gesture}' won with {vote_count} frame votes.")
                
                # Execute Robot actions based on the majority win result
                if most_frequent_gesture in ['headache', 'StomachPain', 'neckpain', 'ShoulderPain']:
                    say_to_patient(f"I notice you are holding your body due to {most_frequent_gesture}. Notifying assistance staff.")
                    trigger_medical_alert(most_frequent_gesture)
                elif most_frequent_gesture == 'ThumbUp':
                    say_to_patient("Excellent. Everything looks stable.")
            else:
                last_final_decision = "No Stable Gesture Detected"
                print("⚠️ Decision Result: No clear gesture frames recorded during this window interval.")

            # Reset monitoring clocks and empty frame vote buckets for next loop iteration
            frame_predictions_buffer = []
            start_time = time.time()

        # Step D: Exit immediately if the user hits the 'ESC' key (Keycode 27)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("\nShutting down robot vision loop cleanly...")
            break

    # Release background operational hooks cleanly
    cap.release()
    cv2.destroyAllWindows()
    print("🤖 Robot Assistant System Offline.")

if __name__ == "__main__":
    main()