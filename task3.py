import numpy.matlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import cv2 as cv

points = [(5.9,3.2),(4.6,2.9),(6.2,2.8),(4.7,3.2),(5.5,4.2),(5.0,3.0),(4.9,3.1),(6.7,3.1),(5.1,3.8),(6.0,3.0)]
k = 3
N = 10

UBIT = 'ruturajt'
np.random.seed(sum([ord(c) for c in UBIT]))

print(matplotlib.__version__)
# given centroids
redMu_coordinates = [6.2,3.2]
greenMu_coordinates = [6.6,3.7]
blueMu_coordinates = [6.5,3.0]

def compute_mean(clusterVector):
    clusterVector = np.array(clusterVector)
    x_array = clusterVector[:,0]
    y_array = clusterVector[:,1]
    x_mean = sum(x_array)/len(x_array)
    y_mean = sum(y_array)/len(y_array)
    return [x_mean,y_mean]


def plotAllPoints( clr, cluster):
    x_cord_array = []
    y_cord_array = []
    for pt in cluster:
        x_cord_array.append(pt[0])
        y_cord_array.append(pt[1])
        text = '(' + str(pt[0])+',' + str(pt[1]) + ')'
        plt.text(pt[0], pt[1],text, family="monospace")

    plt.scatter(x_cord_array, y_cord_array,   marker='^', facecolors=clr, edgecolors=clr)

def plotCentroid(centroidCordinates, clr):
    Mu_X_cord = centroidCordinates[0]
    Mu_Y_cord = centroidCordinates[1]
    text = '(' + str(Mu_X_cord) + ','+ str(Mu_Y_cord) +')'
    plt.text(Mu_X_cord,Mu_Y_cord,text, family="monospace")
    plt.scatter(Mu_X_cord,Mu_Y_cord,  c=clr)

def getClusterVector(points, redCentroid, greenCentroid, blueCentroid):
    redClusterVector = []
    greenClusterVector= [] 
    blueClusterVector = []
    for pt in points:
        point = [pt[0], pt[1]]
        dist_from_red_centroid =  sqrt((pt[0] - redCentroid[0])**2 + (pt[1] - redCentroid[1])**2)
        dist_from_green_centroid =  sqrt((pt[0] - greenCentroid[0])**2 + (pt[1] - greenCentroid[1])**2)
        dist_from_blue_centroid =  sqrt((pt[0] - blueCentroid[0])**2 + (pt[1] - blueCentroid[1])**2)
        minimum_distance = min(dist_from_red_centroid, dist_from_green_centroid, dist_from_blue_centroid)
        if(minimum_distance == dist_from_red_centroid):
            redClusterVector.append(point)
        elif(minimum_distance == dist_from_green_centroid):
            greenClusterVector.append(point)
        elif(minimum_distance == dist_from_blue_centroid):
            blueClusterVector.append(point)
    return redClusterVector, greenClusterVector, blueClusterVector        


#part 1
redClusterVector_itr1, greenClusterVector_itr1, blueClusterVector_itr1 = getClusterVector(points, redMu_coordinates, greenMu_coordinates, blueMu_coordinates)
print(' redClusterVector= ', redClusterVector_itr1)
print(' greenClusterVector = ', greenClusterVector_itr1)
print(' blueClusterVector = ', blueClusterVector_itr1)

# plot all points
plt.figure(1)
plotAllPoints("red", redClusterVector_itr1)
plotAllPoints("green", greenClusterVector_itr1)
plotAllPoints("blue", blueClusterVector_itr1)
plt.savefig('task3_iter1_a.jpg')


# part 2
# update centroids
newCentroid_red_itr1 = compute_mean(redClusterVector_itr1)
newCentroid_green_itr1 = compute_mean(greenClusterVector_itr1)
newCentroid_blue_itr1 = compute_mean(blueClusterVector_itr1)
print('newCentroid_red',newCentroid_red_itr1)
print('newCentroid_green',newCentroid_green_itr1)
print('newCentroid_blue',newCentroid_blue_itr1)

# plot newly calculated centroids
plt.figure(2)
plotCentroid(newCentroid_red_itr1,'red')
plotCentroid(newCentroid_green_itr1,'green')
plotCentroid(newCentroid_blue_itr1,'blue')
plt.savefig('task3_iter1_b.jpg')

# part 3
# classify points according to newly updated centroids (Mu)
redClusterVector_itr2, greenClusterVector_itr2, blueClusterVector_itr2 = getClusterVector(points, newCentroid_red_itr1, newCentroid_green_itr1, newCentroid_blue_itr1)
print('new redClusterVector= ', redClusterVector_itr2)
print('new greenClusterVector = ', greenClusterVector_itr2)
print('new blueClusterVector = ', blueClusterVector_itr2)
plt.figure(3)
# plot all points
plotAllPoints("red", redClusterVector_itr2)
plotAllPoints("green", greenClusterVector_itr2)
plotAllPoints("blue", blueClusterVector_itr2)

plt.savefig('task3_iter2_a.jpg')

# part 4
# compute new centroids
newCentroid_red_itr2 = compute_mean(redClusterVector_itr2)
newCentroid_green_itr2 = compute_mean(greenClusterVector_itr2)
newCentroid_blue_itr2 = compute_mean(blueClusterVector_itr2)
print('newCentroid_red',newCentroid_red_itr2)
print('newCentroid_green',newCentroid_green_itr2)
print('newCentroid_blue',newCentroid_blue_itr2)

# plot newly computed centroids
plt.figure(4)
plotCentroid(newCentroid_red_itr2,'red')
plotCentroid(newCentroid_green_itr2,'green')
plotCentroid(newCentroid_blue_itr2,'blue')
plt.savefig('task3_iter2_b.jpg')

# part 4 

# color quantization

def getRandomPoints():
    
    return (np.random.randint(low=0, high=255, size=1),np.random.randint(low=0, high=255, size=1), np.random.randint(low=0, high=255, size=1))


def copyImage(img):
    h,w,c = img.shape
    temp = [[0 for i in range(w)] for i in range(h)]
    for i in range(h):
        for j in range(w):
          temp[i][j]=img[i][j]
    
    return temp

def findDistance(rgbCord, meanColorsCord):
    
    r = (rgbCord[0]-meanColorsCord[0]) ** 2
    g = (rgbCord[1]-meanColorsCord[1]) ** 2
    b = (rgbCord[2]-meanColorsCord[2]) ** 2
    return sqrt(r + g + b)

def findClusters(meanColors,clusters,red,green,blue,clustersArray):
    for i in range(height):
        for j in range(width):
            distance_array = np.zeros([clusters,],dtype ='uint8')
            # find minimum distance of each point with cluster mean
            for z in range(clusters):
                rgbCord = red[j][i], green[j][i], blue[j][i]
                distance_array[z] = findDistance(rgbCord,meanColors[z])
            min_index = np.argmin(distance_array)
            clustersArray[min_index].append((j,i))
    return clustersArray

def drawNewImage(temp_Img,k,clustersArray,meanColors,red,green,blue):
    for kIterator in range(k):
        a = meanColors[kIterator][0]
        b = meanColors[kIterator][1]
        rgb_color = red[a][b],green[a][b],blue[a][b]
        for z in range(len(clustersArray[kIterator])):
            x_cord = clustersArray[kIterator][z][0]
            y_cord = clustersArray[kIterator][z][1]
            temp_Img[x_cord][y_cord] = rgb_color
    temp_Img = np.asarray(temp_Img, dtype="float32")
    return temp_Img

def getClusteredImage(k,img):
    print("K value = ",k)
    itr = 0
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    clustersArray = [[] for i in range(k)]
    temp_Img = copyImage(img)
    lastIterationMean = [0 for i in range(k)]
    matchingMean = False
    while (matchingMean == False):
        meanColors = [0 for i in range(k)]
        for i in range(k):
            if(itr == 0):
                meanColors[i] = getRandomPoints()
            
            if(itr > 0):
                meanColors[i] = calculateMean(clustersArray[i],img)
                
        if(meanColors == lastIterationMean):
            matchingMean = True
            break
        lastIterationMean = meanColors
        itr+= 1
        print("Iteration count = ",itr)
        clustersArray = findClusters(meanColors,k,red,green,blue,clustersArray)
    temp_Img = drawNewImage(temp_Img,k,clustersArray,meanColors,red,green,blue)
    print("Image created for K value - ",k)
    temp_Img = np.asarray(temp_Img, dtype="float32")
    img_name = "task3_baboon_"+str(k)+".jpg"
    cv.imwrite(img_name, temp_Img)

    
def calculateMean(cluster,img):
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    sum_red = 0 
    sum_blue = 0
    sum_green = 0
    # if cluster has no elements in it ; to avoid divide by zero condition
    if(len(cluster) == 0):
        return (10000,10000,10000)
    for i in range(len(cluster)):
        sum_red+= red[cluster[i][0]][cluster[i][1]]
        sum_blue+= blue[cluster[i][0]][cluster[i][1]]
        sum_green+= green[cluster[i][0]][cluster[i][1]]
    return sum_red//len(cluster), sum_green//len(cluster), sum_blue//len(cluster)    


baboonOriginal_Img = cv.imread("baboon.jpg")
height, width, ch = baboonOriginal_Img.shape

k_array = [3,5,10,20]

for i in range(len(k_array)):
    getClusteredImage(k_array[i],baboonOriginal_Img)

