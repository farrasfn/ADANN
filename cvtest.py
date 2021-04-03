
import cv2
import argparse
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="parot.jpg", help="path to the input image")
args = vars(ap.parse_args())

#
#
#
image = cv2.imread(args["image"])
(B, G, R) = cv2.split(image)


# R=R*0.299
# G=G*0.587
# B=B*0.114
merged = cv2.merge([B,G,R])
cv2.imshow("Merged",merged)

Gray = 0.299*R + 0.587 * G + 0.114 * B
cv2.imshow("grae",Gray)
cv2.waitKey(0)
cv2.destroyAllWindows()