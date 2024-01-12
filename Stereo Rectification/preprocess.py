import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys


def display(title, img):
    cv.imshow(title, img)
    key = cv.waitKey(0)
    cv.destroyAllWindows()

#Draw epipolar lines:
#
#img1: first image
#img1: second image
#
#lines: corresponding epilines
#pts1: point in first image
#pts2: point in second image
#
#colors: array of random colors for visualization
#
#Return: img1, img2 original images with drawn epilines

def drawlines(img1,img2,lines,pts1,pts2,colors,F):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''

    print(img1.shape)

#r: image height
#c: image width
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    cnt=0;
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color=colors[cnt,:].tolist()

#Get points in homogeneous form
        pt1Hom=np.ones(3);
        pt1Hom[0]=pt1[0]
        pt1Hom[1]=pt1[1]

        pt2Hom=np.ones(3);
        pt2Hom[0]=pt2[0]
        pt2Hom[1]=pt2[1]

#Compute epipolar lines in the first image
#        r1=F@pt1Hom
        r1=F.transpose()@pt2Hom

        x0,y0 = map(int, [0, -r1[2]/r1[1] ])
        x1,y1 = map(int, [c, -(r1[2]+r1[0]*c)/r1[1] ])

#Draw epipolar line
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
#Draw feature point location
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
        cnt=cnt+1
    return img1,img2


#Load image
# fileName1=sys.argv[1]
# fileName2=sys.argv[2]

fileName1="./images/a.png"
fileName2="./images/b.png"


img1 = cv.imread(fileName1, cv.IMREAD_GRAYSCALE)  #queryimage # left image
img2 = cv.imread(fileName2, cv.IMREAD_GRAYSCALE) #trainimage # right image

#img1 = cv.imread('Dev0_Image_w1920_h1200_fn400.jpg', cv.IMREAD_GRAYSCALE)  #queryimage # left image
#img2 = cv.imread('Dev1_Image_w1920_h1200_fn400.jpg', cv.IMREAD_GRAYSCALE) #trainimage # right image


#Apply SIFT feature detection
#Also retreive the desriptiors

sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


#Apply FLANN algorithm for matching
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)


#Store the points in Nx2 matrices

pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)


#Why???
pts1= np.int32(pts1)
pts2 = np.int32(pts2)


#Estimate fundamental matrix using RANSAC.
#(Point normalization included)
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_RANSAC)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]


#Number of inliers:

NUM_INLIERS=pts1.shape[0]



#Generate random colorrs for visualization
colors=np.zeros((NUM_INLIERS,3))
for idx in range(0,NUM_INLIERS):
    colors[idx,0]=np.random.randint(0,255)
    colors[idx,1]=np.random.randint(0,255)
    colors[idx,2]=np.random.randint(0,255)



# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2,colors,F)


# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1,colors,F.transpose())

# cv.imshow("Matches", img5)
# cv.waitKey(30000)

# cv.imwrite("EpipLines1.png",img3)
# cv.imwrite("Features1.png",img4)
# cv.imwrite("EpipLines2.png",img5)
# cv.imwrite("Features2.png",img6)
 
display("EpipLines1.png",img5)
display("Features1.png",img4)
display("EpipLines2.png",img3)
display("Features2.png",img6)


np.savetxt("F_RANSAC.mat",F)
np.savetxt("pts1.mat",pts1)
np.savetxt("pts2.mat",pts2)

# Create a mask for white pixels
white_mask = (img1 == 255)

# Apply the mask to the images
img4[white_mask] = 255
img6[white_mask] = 255

display("EpipLines",np.hstack([img5,img3]))
