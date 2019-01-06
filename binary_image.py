import cv2
import numpy as np

img = cv2.imread('baboon.jpg',0)
cv2.imshow('Gray scale',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

h,w = img.shape

img2 = np.array([[0 for i in range(w)] for j in range(h)],dtype = np.uint8)

for x in range(h):
	for y in range(w):
		if(img[x][y]<127):
			img2[x][y] = 0
		else:
			img2[x][y] = 255


print(np.max(img2))
print(img2)
cv2.imshow("Binary Image",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()  
