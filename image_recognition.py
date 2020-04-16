import numpy as np
import imutils
import cv2
from keras.models import load_model

model = load_model('final_model.h5')
image = cv2.imread("sample_image2.jpg")
image = imutils.resize(image, width=320, height=320)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
_, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# thresh = cv2.dilate(thresh, None)
(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
avgCntArea = np.mean([cv2.contourArea(k) for k in cnts])
digits = []
boxes = []
for (i, c) in enumerate(cnts):
    if cv2.contourArea(c) < avgCntArea / 10:
        continue
    mask = np.zeros(gray.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    hull = cv2.convexHull(c)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    mask = cv2.bitwise_and(thresh, thresh, mask=mask)
    maskedDigit = mask[y - 4:y + h + 4, x - 4:x + w + 4]
    resized = cv2.resize(maskedDigit, (28, 28), interpolation=cv2.INTER_AREA)
    img = resized.reshape(1, 28, 28, 1).astype('float32') / 255.0
    digit = model.predict_classes(img)[0]
    print(digit)
    cv2.putText(image, str(digit), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 0)
cv2.imshow("Recognized", image)
cv2.waitKey(0)