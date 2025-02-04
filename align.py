import cv2
import numpy as np

def align_answer_sheet(template_path, target_path, output_path):
    # Read images
    template = cv2.imread(template_path)
    target = cv2.imread(target_path)
    
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
    
    # Save the aligned image
    cv2.imwrite(output_path, aligned)
    
    return aligned

def show_alignment(template, aligned):
    # Create a visualization of the alignment
    alpha = 0.5
    beta = 1.0 - alpha
    overlay = cv2.addWeighted(template, alpha, aligned, beta, 0.0)
    
    # Display results
    cv2.imshow('Template', template)
    cv2.imshow('Aligned Image', aligned)
    cv2.imshow('Overlay', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    template_path = "blank.png"  # Save the provided image as template.png
    target_path = "scan1.png"      # The image you want to align
    output_path = "aligned1.png"     # Where to save the aligned image
    
    # Read template
    template = cv2.imread(template_path)
    
    # Perform alignment
    aligned = align_answer_sheet(template_path, target_path, output_path)
    
    # Show results
    show_alignment(template, aligned)