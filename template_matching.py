import cv2
import numpy as np

class PlayerDetector:
    def __init__(self):
        # Load template images
        self.standing_template = cv2.imread('templates/standing.png', 0)
        self.crouching_template = cv2.imread('templates/crouching.png', 0)
        
        # Define head offset ratios (these need to be calibrated based on your templates)
        self.head_offset_standing = 0.15  # Approximate head position from top of template
        self.head_offset_crouching = 0.20
        
    def detect_players(self, frame):
        """Detect players in the frame using template matching"""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect standing players
        standing_results = cv2.matchTemplate(gray_frame, self.standing_template, cv2.TM_CCOEFF_NORMED)
        standing_locations = np.where(standing_results >= 0.8)  # Threshold needs tuning
        
        # Detect crouching players
        crouching_results = cv2.matchTemplate(gray_frame, self.crouching_template, cv2.TM_CCOEFF_NORMED)
        crouching_locations = np.where(crouching_results >= 0.8)  # Threshold needs tuning
        
        return self._process_detections(standing_locations, crouching_locations)
    
    def _process_detections(self, standing_locs, crouching_locs):
        """Process detections and calculate head positions"""
        targets = []
        
        # Process standing players
        for pt in zip(*standing_locs[::-1]):
            head_pos = (
                pt[0] + self.standing_template.shape[1]//2,  # X center
                pt[1] + int(self.standing_template.shape[0] * self.head_offset_standing)  # Y with offset
            )
            targets.append({
                'position': pt,
                'head_position': head_pos,
                'stance': 'standing'
            })
            
        # Process crouching players
        for pt in zip(*crouching_locs[::-1]):
            head_pos = (
                pt[0] + self.crouching_template.shape[1]//2,  # X center
                pt[1] + int(self.crouching_template.shape[0] * self.head_offset_crouching)  # Y with offset
            )
            targets.append({
                'position': pt,
                'head_position': head_pos,
                'stance': 'crouching'
            })
            
        return targets
