import cv2
import numpy as np
import imutils
from imutils import contours
import four_point
from collections import defaultdict

Answer_key = {
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

image = cv2.imread("blank_cropped.png") 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 100, 250)


cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

paper = image
warped = gray
thresh = cv2.threshold(warped, 50, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

questionCnts = []
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if w >= 20 and h >= 20 and ar >= 0.8 and ar <= 1.2:
        questionCnts.append(c)


questionCnts = sorted(questionCnts, key=lambda ctr: (cv2.boundingRect(ctr)[1], cv2.boundingRect(ctr)[0]))



debug_image = image.copy() # Copy of the original image for debugging purposes
correct = 0
column1,column2,column3,column4 = [],[],[],[]
answers = {}
min_filled_area = 150
x_tolerance = 400
y_tolerance = 50
qNum = 0

avg_coords = [(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] / 2, cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] / 2) for c in questionCnts]

# Group contours that have similar x and y coordinates into the same question
questions = defaultdict(list)
for i, c in enumerate(questionCnts):
    x, y = avg_coords[i]
    questions[(int(y / y_tolerance), int(x / x_tolerance))].append(c)

# Sort the questions based on their average y-coordinate, then their x-coordinate
questions = [questions[key] for key in sorted(questions)]


debug_image = cv2.drawContours(debug_image, questionCnts, -1, (0, 255, 0), 2)
cv2.imshow("Question", debug_image) # Display the image with question contours highlighted
cv2.waitKey(0)

for question in questions:
    if qNum % 4 == 0:
        column1.append(question)
        qNum += 1
    elif qNum % 4 == 1:
        column2.append(question)
        qNum += 1
    elif qNum % 4 == 2:
        column3.append(question)
        qNum += 1
    else:
        column4.append(question)
        qNum += 1

qNum = -1
for question in column1:
    bubbled = None
    blank = True
    qNum += 1
    debug_image = cv2.drawContours(debug_image, question, -1, (0, 255, 0), 2)
    for j, c in enumerate(question):
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
       
        
        if total > min_filled_area and (bubbled is None or total > bubbled[0]):
            bubbled = (total, j)
            color = (0, 0, 255)
            blank = False
    k = Answer_key[qNum]
    if blank:
        answers[qNum] = None
    else:
        answers[qNum] = bubbled[1]
        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1
for question in column2:
    bubbled = None
    blank = True
    qNum += 1
    debug_image = cv2.drawContours(debug_image, question, -1, (0, 255, 0), 2)
    for j, c in enumerate(question):
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
       
        
        if total > min_filled_area and (bubbled is None or total > bubbled[0]):
            bubbled = (total, j)
            color = (0, 0, 255)
            blank = False
    k = Answer_key[qNum]
    if blank:
        answers[qNum] = None
    else:
        answers[qNum] = bubbled[1]
        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1
for question in column3:
    bubbled = None
    blank = True
    qNum += 1
    debug_image = cv2.drawContours(debug_image, question, -1, (0, 255, 0), 2)
    for j, c in enumerate(question):
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
        if total > min_filled_area and (bubbled is None or total > bubbled[0]):
            bubbled = (total, j)
            color = (0, 0, 255)
            blank = False
    k = Answer_key[qNum]
    if blank:
        answers[qNum] = None
    else:
        answers[qNum] = bubbled[1]
        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1
for question in column4:
    bubbled = None
    blank = True
    qNum += 1
    debug_image = cv2.drawContours(debug_image, question, -1, (0, 255, 0), 2)
    for j, c in enumerate(question):
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
       
        
        if total > min_filled_area and (bubbled is None or total > bubbled[0]):
            bubbled = (total, j)
            color = (0, 0, 255)
            blank = False
    k = Answer_key[qNum]
    if blank:
        answers[qNum] = None
    else:
        answers[qNum] = bubbled[1]
        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1

for question in questions:
    debug_image = cv2.drawContours(debug_image, question, -1, (0, 255, 0), 2)


print("Correct answers:", correct)
print("Answers:", answers)
