import numpy as np
import cv2 as cv

# references
# (draw epilines) - > https://docs.opencv.org/3.2.0/da/de9/tutorial_py_epipolar_geometry.html

# read images
img_left = cv.imread('tsucuba_left.png',1)
img_right = cv.imread('tsucuba_right.png',1)

UBIT = 'ruturajt'

# key point detection (1)
sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img_left,None)
kp2, des2 = sift.detectAndCompute(img_right,None)
sift_1=cv.drawKeypoints(img_left,kp1,None)
sift_2=cv.drawKeypoints(img_right,kp2,None)
cv.imwrite('task2_sift1.jpg',sift_1)
cv.imwrite('task2_sift2.jpg',sift_2)

# BFMatcher with default params
bruteForceMatcher = cv.BFMatcher()
matches = bruteForceMatcher.knnMatch(des1,des2, k=2)

# finding good matches
good_matches = []
good_matches_list = []
pts1 = []
pts2 = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_matches.append([m])
        good_matches_list.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatches(img_left,kp1,img_right,kp2,good_matches_list,None,flags=2)

cv.imwrite('task2_matches_knn.jpg',img3)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
Fundamental_matrix, mask = cv.findFundamentalMat(pts1,pts2,cv.RANSAC)


# task 2.2 finding fundamental matrix
print(Fundamental_matrix)

# selecting only inliners
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

np.random.seed(sum([ord(c) for c in UBIT]))
random_indices = np.random.randint(low=0, high=len(pts2), size=11)

# finding random indices for colors
clr = []
for i in range(11):
	clr.append(np.random.randint(0, 255, 3))
print("clr-",clr)

# defining new points
new_pts1 = []
new_pts2 = []
for i in random_indices:
	new_pts1.append(pts1[i])
	new_pts2.append(pts2[i])

def drawEpilines(img1, img2, lines, pts1, pts2):
    
    r,c = img1.shape[:2]
    for r, col in zip(lines, clr):
        color = tuple(col.tolist())
        x_0,y_0 = map(int, [0, -r[2]/r[1] ])
        x_1,y_1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x_0,y_0), (x_1,y_1), color,1)
        
    return img1,img2
# finding epi polar line on right image
line_right = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, Fundamental_matrix)
line_right = line_right.reshape(-1, 3)
img4, img5 = drawEpilines(img_left, img_right, line_right, new_pts1, new_pts2)

# finding epipolar line on left image
line_left = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, Fundamental_matrix)
line_left = line_left.reshape(-1, 3)
img6, img7 = drawEpilines(img_right, img_left,line_left, new_pts2, new_pts1)

cv.imwrite('task2_epi_left.jpg', img4)
cv.imwrite('task2_epi_right.jpg', img6)


# creating stereo object
stereo_obj = cv.StereoSGBM_create(numDisparities=64, blockSize=25)
disparity_image = stereo_obj.compute(img_left,img_right)
cv.imwrite('task2_disparity.jpg',disparity_image)
