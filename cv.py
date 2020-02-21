# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
from modelRunner import run_example

# load the example image
image = cv2.imread("sample_image.png")
# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500)
imageToPrint = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)
# find contours in the edge map, then sort them by their
# size in descending order
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if the contour has four vertices, then we have found
	# the thermostat display
	if len(approx) == 4:
		displayCnt = approx
		break
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []
# loop over the digit area candidates
for c in cnts:
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)
	# if the contour is sufficiently large, it must be a digit
	if w >= 15 and (h >= 30):
		digitCnts.append(c)
# sort the contours from left-to-right, then initialize the
# actual digits themselves
digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
# loop over each of the digits
for c in digitCnts:
	# extract the digit ROI
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.imwrite("digit.jpg",image[y-5:y + h+5, x-5:x + w+5])
	digit = run_example()
	print(digit)
	cv2.rectangle(imageToPrint, (x, y), (x + w, y + h), (0, 255, 0), 1)
	l = 0
	for string in digit:
		cv2.putText(imageToPrint, str(string), (x - 5, y - 5-l), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 0)
		l+=20
cv2.imshow("edged",imageToPrint)
cv2.waitKey(0)