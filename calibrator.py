import cv2
import numpy as np

class BubbleSheetCalibrator:
    def __init__(self):
        self.coordinates = []
        self.current_point = None
        
    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.coordinates.append((x, y))
            # Draw a small crosshair at clicked point for precision
            cv2.drawMarker(self.image, (x, y), (0, 255, 0), 
                          cv2.MARKER_CROSS, 10, 2)
            cv2.imshow('Calibration', self.image)
            print(f"Point {len(self.coordinates)}: ({x}, {y})")
            
            if len(self.coordinates) >= 2:
                # Calculate distance between last two points
                x1, y1 = self.coordinates[-2]
                x2, y2 = self.coordinates[-1]
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                print(f"Distance from last point: {distance:.2f} pixels")

    def calibrate(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Could not read image")
            
        cv2.imshow('Calibration', self.image)
        cv2.setMouseCallback('Calibration', self.click_event)
        
        print("\nPrecise Calibration Instructions:")
        print("Click the exact center of the following bubbles in order:")
        print("1. Question 1, option A")
        print("2. Question 1, option E")
        print("3. Question 25, option A")
        print("4. Question 26, option A")
        print("This will give us multiple measurement points to average")
        print("\nPress 'q' to quit when done")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        cv2.destroyAllWindows()
        
        if len(self.coordinates) >= 4:
            # Calculate average spacings using multiple points
            option_spacing = (self.coordinates[1][0] - self.coordinates[0][0]) / 4  # Divide by 4 for A to E
            row_spacing = (self.coordinates[2][1] - self.coordinates[0][1]) / 24   # Divide by 24 for 25 rows
            column_spacing = self.coordinates[3][0] - self.coordinates[0][0]
            
            print("\nCalibration Results:")
            print(f"Starting coordinates (x, y): {self.coordinates[0]}")
            print(f"Option spacing: {option_spacing:.2f} pixels")
            print(f"Row spacing: {row_spacing:.2f} pixels")
            print(f"Column spacing: {column_spacing:.2f} pixels")
            
            # Verify calculations
            print("\nVerification distances:")
            print(f"A to E distance: {option_spacing * 4:.2f} pixels (should match visual distance)")
            print(f"Q1 to Q25 distance: {row_spacing * 24:.2f} pixels (should match visual distance)")
            
            return {
                'start_x': self.coordinates[0][0],
                'start_y': self.coordinates[0][1],
                'option_spacing': option_spacing,
                'row_spacing': row_spacing,
                'column_spacing': column_spacing
            }

if __name__ == "__main__":
    calibrator = BubbleSheetCalibrator()
    try:
        results = calibrator.calibrate("blank.png")
        print("\nConfiguration values for grader:")
        print("self.start_x =", results['start_x'])
        print("self.start_y =", results['start_y'])
        print("self.option_spacing =", results['option_spacing'])
        print("self.row_spacing =", results['row_spacing'])
        print("self.column_spacing =", results['column_spacing'])
    except Exception as e:
        print(f"Error during calibration: {e}")