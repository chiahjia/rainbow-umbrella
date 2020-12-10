import cv2
import numpy as np

img = cv2.imread("people2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#img2 = cv2.resize(gray, (300, 300))

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade2 = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
eye2_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
# Change cascade below to detect eyes instead.
faces = cascade.detectMultiScale(gray, 1.1, 4)
print(faces)
for (x, y, w, h) in faces:
	cv2.rectangle(img, (x, y), (x+w, y+h), (200, 20, 0), 2)


cv2.imshow("ting", cv2.resize(img, (1000, 1000)))
cv2.waitKey(0)
