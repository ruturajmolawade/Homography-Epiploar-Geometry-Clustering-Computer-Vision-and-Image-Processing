import numpy as np
import cv2 as cv

# references
# https://docs.opencv.org/3.0-beta/modules/cudawarping/doc/warping.html
# (warping images) -> https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective/20355545#20355545
UBIT = 'ruturajt'
np.random.seed(sum([ord(c) for c in UBIT]))

# read images
img1 = cv.imread('mountain1.jpg',1)
img2 = cv.imread('mountain2.jpg',1)


# key point detection (1)
sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
sift_1=cv.drawKeypoints(img1,kp1,None)
sift_2=cv.drawKeypoints(img2,kp2,None)
cv.imwrite('task1_sift1.jpg',sift_1)
cv.imwrite('task1_sift2.jpg',sift_2)


# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)


# Apply ratio test
good = []
good_list = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        good_list.append(m)

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,good_list,None,flags=2)

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_list ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_list ]).reshape(-1,1,2)

M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)

matchesMask = mask.ravel().tolist()
indices = [i for i, x in enumerate(matchesMask) if x == 1]
random_indices = np.random.randint(low=0, high=len(indices), size=10)
random_list = []
for i in range(len(random_indices)):
	random_list.append(good_list[random_indices[i]])
print("random_list - ",random_list)

h,w,c = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#dst = cv.perspectiveTransform(pts,M)

img5 = cv.warpPerspective(img1, M, (img2.shape[1] + h,img2.shape[0]))

#print(mask)

draw_params = dict(matchColor = (0,0,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)

img4 = cv.drawMatches(img1,kp1,img2,kp2,random_list,None,**draw_params)

# img5 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)


print(M)

cv.imwrite('task1_matches_knn.jpg',img3)

cv.imwrite('task1_matches.jpg',img4)

# cv.imwrite('task1_pano.jpg',img5)

def warpImages(img1, img2, H):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [min_x, min_y] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-min_x,-min_y]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    wrap = cv.warpPerspective(img2, Ht.dot(H), (xmax-min_x, ymax-min_y))
    wrap[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return wrap

wrap = warpImages(img2, img1, M)
cv.imwrite('task1_pano.jpg',wrap)
