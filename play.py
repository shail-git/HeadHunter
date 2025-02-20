import os
import time
import cv2
import numpy as np
from controller import GameController
from template_matching import PlayerDetector

def setup_game():
    """Setup initial game state"""
    controller = GameController()
    
    # Give time to switch to game window
    print("Switching to game window...")
    time.sleep(2)
    
    # Buy AK47 (optional)
    print("Buying weapon...")
    for c in "buy ak47":
        controller.press_key(ord(c))
        time.sleep(0.1)
        controller.release_key(ord(c))
    
    return controller

def main():
    # Initialize
    controller = setup_game()
    detector = PlayerDetector()
    
    print("Starting game...")
    time.sleep(1)
    
    # For visualization
    IS_DEMO = True  # Set to True to see detection visualization
    
    try:
        while True:
            loop_start_time = time.time()
            
            # Capture game screen
            frame = controller.grab_screen()
            
            # Detect players
            targets = detector.detect_players(frame)
            
            if targets:
                # Get screen center (crosshair position)
                center_x = frame.shape[1] // 2
                center_y = frame.shape[0] // 2
                
                # Find closest target to crosshair
                closest_target = min(targets, key=lambda t: 
                    ((t['head_position'][0] - center_x)**2 + 
                     (t['head_position'][1] - center_y)**2)**0.5)
                
                # Calculate distance to crosshair
                distance = ((closest_target['head_position'][0] - center_x)**2 + 
                          (closest_target['head_position'][1] - center_y)**2)**0.5
                
                # Shoot if target is close enough
                if distance < 50:  # Adjust threshold as needed
                    controller.shoot()
                    
                # Visualization for demo mode
                if IS_DEMO:
                    # Draw detection boxes
                    cv2.rectangle(frame, 
                                closest_target['position'],
                                (closest_target['position'][0] + detector.standing_template.shape[1],
                                 closest_target['position'][1] + detector.standing_template.shape[0]),
                                (0, 255, 0), 2)
                    # Draw head position
                    cv2.circle(frame, closest_target['head_position'], 5, (0, 0, 255), -1)
                    # Draw crosshair
                    cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)
            
            # Show frame if in demo mode
            if IS_DEMO:
                cv2.imshow('Game Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Maintain consistent frame rate
            while time.time() < loop_start_time + 1/30:  # 30 FPS
                time.sleep(0.001)
                
    except KeyboardInterrupt:
        print("\nStopping game...")
    finally:
        if IS_DEMO:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
