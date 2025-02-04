import cv2
import numpy as np
import imutils
from imutils import contours
import four_point
from collections import defaultdict

# Constants
QUESTIONS_PER_COLUMN = 25
COLUMNS = 4
OPTIONS = 5
START_X = 204
START_Y = 254
OPTION_SPACING = 52
ROW_SPACING = 45
COLUMN_SPACING = 336
BUBBLE_RADIUS = 17
DARKNESS_THRESHOLD = 180

# Visualization colors (BGR format)
COLORS = {
    'sampling': (0, 255, 0),    # Green: showing what's being sampled
    'filled': (0, 0, 255),      # Red: detected as filled
    'correct': (0, 255, 0),     # Green: correct answer
    'incorrect': (0, 0, 255),   # Red: incorrect answer
    'expected': (255, 165, 0)   # Orange: expected answer if incorrect
}

def get_bubble_coordinates(question_num):
    """Returns list of (x,y) coordinates for all options of a question"""
    column = question_num // QUESTIONS_PER_COLUMN
    row = question_num % QUESTIONS_PER_COLUMN
    
    coordinates = []
    base_x = START_X + (column * COLUMN_SPACING)
    y = START_Y + (row * ROW_SPACING)
    
    for option in range(OPTIONS):
        x = base_x + (option * OPTION_SPACING)
        coordinates.append((x, y))
        
    return coordinates

def check_bubble(image, center_x, center_y, debug_image=None):
    """Check if a bubble is filled and optionally visualize the sampling"""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), BUBBLE_RADIUS, 255, -1)
    
    mean_intensity = cv2.mean(image, mask=mask)[0]
    
    if debug_image is not None:
        cv2.circle(debug_image, (center_x, center_y), BUBBLE_RADIUS, 
                  COLORS['sampling'], 1)
    
    return mean_intensity < DARKNESS_THRESHOLD

def visualize_grading(image, answers, answer_key):
    """Create a visualization of the grading process"""
    debug_image = image.copy()
    
    for q in range(100):
        coords = get_bubble_coordinates(q)
        student_answer = answers.get(q)
        correct_answer = answer_key.get(q)
        
        for opt, (x, y) in enumerate(coords):
            cv2.circle(debug_image, (x, y), BUBBLE_RADIUS, 
                      COLORS['sampling'], 1)
            
            if student_answer == opt:
                color = COLORS['correct'] if student_answer == correct_answer \
                       else COLORS['incorrect']
                cv2.circle(debug_image, (x, y), BUBBLE_RADIUS, color, 2)
            
            elif opt == correct_answer and student_answer != correct_answer:
                cv2.circle(debug_image, (x, y), BUBBLE_RADIUS, 
                         COLORS['expected'], 2)
            
            if opt == 0:
                cv2.putText(debug_image, str(q+1), (x-25, y+5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return debug_image

def align_sheet(template_path, target_path):
    """Aligns the target image with the template using feature matching"""
    template = cv2.imread(template_path)
    target = cv2.imread(target_path)
    
    if template is None or target is None:
        raise ValueError("Could not read template or target image")
        
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template_gray, None)
    kp2, des2 = sift.detectAndCompute(target_gray, None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    aligned = cv2.warpPerspective(target, H, (template.shape[1], template.shape[0]))
    
    return aligned

def grade_sheet(image_path, answer_key, template_path="blank.png", 
                show_visualization=True, save_aligned=True):
    """Grades a sheet with automatic alignment"""
    try:
        aligned_image = align_sheet(template_path, image_path)
        
        if save_aligned:
            aligned_path = f"aligned_{image_path.split('/')[-1]}"
            cv2.imwrite(aligned_path, aligned_image)
            print(f"Saved aligned image as: {aligned_path}")
        
        gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
        answers = {}
        correct_count = 0
        
        for q in range(100):
            coords = get_bubble_coordinates(q)
            answers[q] = None
            
            for opt, (x, y) in enumerate(coords):
                if check_bubble(gray, x, y, aligned_image if show_visualization else None):
                    answers[q] = opt
                    if answer_key.get(q) == opt:
                        correct_count += 1
                    break
        
        if show_visualization:
            debug_image = visualize_grading(aligned_image, answers, answer_key)
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
    
    try:
        results = grade_sheet("scan1.png", answer_key)
        print(f"Score: {results['score']}%")
        print(f"Correct answers: {results['correct']}")
        print("Student answers:", results['answers'])
    except Exception as e:
        print(f"Error grading sheet: {e}")
    
