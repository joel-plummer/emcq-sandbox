import cv2
import numpy as np
import imutils
from imutils import contours
import four_point
from collections import defaultdict

class BubbleSheetGrader:
    def __init__(self):
        # Existing initialization code remains the same
        self.questions_per_column = 25
        self.columns = 4
        self.options = 5
        
        self.start_x = 204
        self.start_y = 254
        self.option_spacing = 52
        self.row_spacing = 45
        self.column_spacing = 336
        
        self.bubble_radius = 17
        self.darkness_threshold = 180
        
        self.colors = {
            'sampling': (0, 255, 0),    
            'filled': (0, 0, 255),      
            'correct': (0, 255, 0),     
            'incorrect': (0, 0, 255),   
            'expected': (255, 165, 0)   
        }
        
        # Add template path
        self.template_path = "blank.png"  # Save your template image with this name

    def align_sheet(self, target_path):
        """Aligns the target image with the template using feature matching"""
        # Read images
        template = cv2.imread(self.template_path)
        target = cv2.imread(target_path)
        
        if template is None or target is None:
            raise ValueError("Could not read template or target image")
            
        # Convert to grayscale
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Detect keypoints and compute descriptors
        kp1, des1 = sift.detectAndCompute(template_gray, None)
        kp2, des2 = sift.detectAndCompute(target_gray, None)
        
        # Initialize matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Match descriptors
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # Get corresponding points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Calculate homography matrix
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        # Warp image
        aligned = cv2.warpPerspective(target, H, (template.shape[1], template.shape[0]))
        
        return aligned

    def grade_with_alignment(self, image_path, answer_key, show_visualization=True, save_aligned=True):
        """Grades a sheet with automatic alignment"""
        try:
            # Align the sheet first
            aligned_image = self.align_sheet(image_path)
            
            # Save aligned image if requested
            if save_aligned:
                aligned_path = f"aligned_{image_path.split('/')[-1]}"
                cv2.imwrite(aligned_path, aligned_image)
                print(f"Saved aligned image as: {aligned_path}")
            
            # Convert aligned image to grayscale
            gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
            
            answers = {}
            correct_count = 0
            
            # Process each question
            for q in range(100):
                coords = self.get_bubble_coordinates(q)
                answers[q] = None
                
                # Check each option (A-E)
                for opt, (x, y) in enumerate(coords):
                    if self.check_bubble(gray, x, y, aligned_image if show_visualization else None):
                        answers[q] = opt
                        if answer_key.get(q) == opt:
                            correct_count += 1
                        break
            
            # Create visualization if requested
            if show_visualization:
                debug_image = self.visualize_grading(aligned_image, answers, answer_key)
                cv2.imshow('Grading Visualization', debug_image)
                cv2.imwrite('grading_visualization.png', debug_image)
                print("Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return {
                'answers': answers,
                'correct': correct_count,
                'score': (correct_count / len(answer_key)) * 100,
                'aligned_image': aligned_image
            }
            
        except Exception as e:
            raise Exception(f"Error in grading process: {e}")

    def get_bubble_coordinates(self, question_num):
        """Returns list of (x,y) coordinates for all options of a question"""
        column = question_num // self.questions_per_column
        row = question_num % self.questions_per_column
        
        coordinates = []
        base_x = self.start_x + (column * self.column_spacing)
        y = self.start_y + (row * self.row_spacing)
        
        for option in range(self.options):
            x = base_x + (option * self.option_spacing)
            coordinates.append((x, y))
            
        return coordinates

    def check_bubble(self, image, center_x, center_y, debug_image=None):
        """Check if a bubble is filled and optionally visualize the sampling"""
        # Create a circular mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), self.bubble_radius, 255, -1)
        
        # Get average intensity in the bubble region
        mean_intensity = cv2.mean(image, mask=mask)[0]
        
        # If debug image provided, draw the sampling circle
        if debug_image is not None:
            cv2.circle(debug_image, (center_x, center_y), self.bubble_radius, 
                      self.colors['sampling'], 1)
        
        return mean_intensity < self.darkness_threshold

    def visualize_grading(self, image, answers, answer_key):
        """Create a visualization of the grading process"""
        debug_image = image.copy()
        
        for q in range(100):
            coords = self.get_bubble_coordinates(q)
            student_answer = answers.get(q)
            correct_answer = answer_key.get(q)
            
            # Draw all bubbles being sampled
            for opt, (x, y) in enumerate(coords):
                # Draw the bubble area being sampled
                cv2.circle(debug_image, (x, y), self.bubble_radius, 
                          self.colors['sampling'], 1)
                
                # # If this bubble was marked by student
                if student_answer == opt:
                    # Determine if it was correct
                    color = self.colors['correct'] if student_answer == correct_answer \
                           else self.colors['incorrect']
                    cv2.circle(debug_image, (x, y), self.bubble_radius, color, 2)
                
                # If this was the correct answer and student got it wrong
                elif opt == correct_answer and student_answer != correct_answer:
                    cv2.circle(debug_image, (x, y), self.bubble_radius, 
                             self.colors['expected'], 2)
                
                # Add question numbers at the start of each row
                if opt == 0:  # First bubble of each question
                    cv2.putText(debug_image, str(q+1), (x-25, y+5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return debug_image

    def grade(self, image_path, answer_key, show_visualization=True):
        # Read and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        answers = {}
        correct_count = 0
        
        # Process each question
        for q in range(100):
            coords = self.get_bubble_coordinates(q)
            answers[q] = None
            
            # Check each option (A-E)
            for opt, (x, y) in enumerate(coords):
                if self.check_bubble(gray, x, y, image if show_visualization else None):
                    answers[q] = opt
                    if answer_key.get(q) == opt:
                        correct_count += 1
                    break
        
        # Create visualization if requested
        if show_visualization:
            debug_image = self.visualize_grading(image, answers, answer_key)
            
            # Show the visualization
            cv2.imshow('Grading Visualization', debug_image)
            print("Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Optionally save the visualization
            cv2.imwrite('grading_visualization.png', debug_image)
        
        return {
            'answers': answers,
            'correct': correct_count,
            'score': (correct_count / len(answer_key)) * 100
        }

# Example usage
if __name__ == "__main__":
    # Example answer key (0=A, 1=B, 2=C, 3=D, 4=E)
    answer_key = {
0: 1, 1: 4, 2: 0, 3: 2, 4: 1, 5: 3, 6: 1, 7: 2, 8: 4, 9: 2,
10: 3, 11: 0, 12: 1, 13: 4, 14: 2, 15: 0, 16: 3, 17: 1, 18: 4, 19: 2,
20: 0, 21: 3, 22: 1, 23: 4, 24: 2, 25: 0, 26: 3, 27: 1, 28: 4, 29: 2,
30: 0, 31: 3, 32: 1, 33: 4, 34: 2, 35: 0, 36: 3, 37: 1, 38: 4, 39: 2,
40: 0, 41: 3, 42: 1, 43: 4, 44: 2, 45: 0, 46: 3, 47: 1, 48: 4, 49: 2,
50: 0, 51: 3, 52: 1, 53: 4, 54: 2, 55: 0, 56: 3, 57: 1, 58: 4, 59: 2,
60: 0, 61: 3, 62: 1, 63: 4, 64: 2, 65: 0, 66: 3, 67: 1, 68: 4, 69: 2,
70: 0, 71: 3, 72: 1, 73: 4, 74: 2, 75: 0, 76: 3, 77: 1, 78: 4, 79: 2,
80: 0, 81: 3, 82: 1, 83: 4, 84: 2, 85: 0, 86: 3, 87: 1, 88: 4, 89: 2,
90: 0, 91: 3, 92: 1, 93: 4, 94: 2, 95: 0, 96: 3, 97: 1, 98: 4, 99: 2
}
    
    grader = BubbleSheetGrader()
    try:
        # Use the new grade_with_alignment method instead of grade
        results = grader.grade_with_alignment("scan1.png", answer_key)
        print(f"Score: {results['score']}%")
        print(f"Correct answers: {results['correct']}")
        print("Student answers:", results['answers'])
    except Exception as e:
        print(f"Error grading sheet: {e}")
    
