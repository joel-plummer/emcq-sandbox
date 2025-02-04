import cv2
import numpy as np
import json
from typing import Dict, Tuple

class TemplateCalibrator:
    def __init__(self):
        self.coordinates = {}
        self.reference_points = [
            (1, "Q1 Option A"),
            (25, "Q25 Option A"),
            (26, "Q26 Option A"),
            (50, "Q50 Option A"),
            (51, "Q51 Option A"),
            (75, "Q75 Option A"),
            (76, "Q76 Option A"),
            (100, "Q100 Option A")
        ]
        self.current_point_idx = 0
    
    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_point_idx < len(self.reference_points):
                q_num, desc = self.reference_points[self.current_point_idx]
                self.coordinates[q_num] = (x, y)
                
                # Draw marker and label
                cv2.drawMarker(self.image, (x, y), (0, 255, 0), 
                              cv2.MARKER_CROSS, 10, 2)
                cv2.putText(self.image, f"Q{q_num}", (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow('Template Calibration', self.image)
                
                print(f"Marked Q{q_num} at ({x}, {y})")
                
                self.current_point_idx += 1
                
                if self.current_point_idx < len(self.reference_points):
                    next_q, next_desc = self.reference_points[self.current_point_idx]
                    print(f"\nNow click: {next_desc}")
                else:
                    print("\nAll points marked! Press 'q' to save and quit.")

    def calibrate(self, template_path: str) -> Dict[int, Tuple[int, int]]:
        """Calibrate using the template image"""
        self.image = cv2.imread(template_path)
        if self.image is None:
            raise ValueError("Could not read template image")
        
        # Scale down image if it's too large
        max_height = 900
        height, width = self.image.shape[:2]
        if height > max_height:
            scale = max_height / height
            new_width = int(width * scale)
            self.image = cv2.resize(self.image, (new_width, max_height))
        
        cv2.namedWindow('Template Calibration')
        cv2.imshow('Template Calibration', self.image)
        cv2.setMouseCallback('Template Calibration', self.click_event)
        
        print("Template Calibration Instructions:")
        print("Click on reference points in this order:")
        for q_num, desc in self.reference_points:
            print(f"- {desc}")
        print("\nPress 'r' to reset, 'q' to save and quit")
        print(f"\nFirst, click: {self.reference_points[0][1]}")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("\nResetting points...")
                self.coordinates = {}
                self.current_point_idx = 0
                self.image = cv2.imread(template_path)
                if height > max_height:
                    self.image = cv2.resize(self.image, (new_width, max_height))
                cv2.imshow('Template Calibration', self.image)
                print(f"Click: {self.reference_points[0][1]}")
        
        cv2.destroyAllWindows()
        
        if len(self.coordinates) < len(self.reference_points):
            raise ValueError("Not all reference points were marked")
        
        return self.coordinates
    
    def save_coordinates(self, coordinates: Dict[int, Tuple[int, int]], 
                        filename: str = 'template_coordinates.json'):
        """Save coordinates to a JSON file"""
        # Convert coordinates to serializable format
        serializable_coords = {str(k): list(v) for k, v in coordinates.items()}
        
        with open(filename, 'w') as f:
            json.dump(serializable_coords, f, indent=4)
        print(f"\nTemplate coordinates saved to {filename}")
        
        # Also print as Python dictionary format
        print("\nReference coordinates for AutomatedDeskewer:")
        print("self.reference_points = {")
        for q_num, (x, y) in coordinates.items():
            print(f"    {q_num}: ({x}, {y}),")
        print("}")

if __name__ == "__main__":
    try:
        calibrator = TemplateCalibrator()
        coords = calibrator.calibrate("blank.png")  # Use your template image
        calibrator.save_coordinates(coords)
    except Exception as e:
        print(f"Error: {e}")