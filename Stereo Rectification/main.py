import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def display(title, img):
    cv.imshow(title, img)
    key = cv.waitKey(0)
    cv.destroyAllWindows()

fileName1="./images/a.png"
fileName2="./images/b.png"

# fileName1="/Users/sivadineshponrajan/Monkey/3D sensors/Rectification_python/images/Dev1_Image_w1920_h1200_fn400.jpg"
# fileName2="/Users/sivadineshponrajan/Monkey/3D sensors/Rectification_python/images/Dev0_Image_w1920_h1200_fn400.jpg"


img1 = cv.imread(fileName1, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(fileName2, cv.IMREAD_GRAYSCALE) 

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

# Keep good matches: calculate distinctive image features
# Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
matchesMask = [[0, 0] for i in range(len(matches))]
good = []


pts1 = []
pts2 = []

for i, (m, n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        # Keep this keypoint pair
        matchesMask[i] = [1, 0]
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)


# Draw the keypoint matches between both pictures
# Still based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
# draw_params = dict(matchColor=(0, 255, 0),
#                    singlePointColor=(255, 0, 0),
#                    matchesMask=matchesMask[300:500],
#                    flags=cv.DrawMatchesFlags_DEFAULT)

# keypoint_matches = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches[300:500], None, **draw_params)
# display("Keypoint matches", keypoint_matches)

# ------------------------------------------------------------
# STEREO RECTIFICATION

# Calculate the fundamental matrix for the cameras
# https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

# We select only inlier points
pts1 = pts1[inliers.ravel() == 1]
pts2 = pts2[inliers.ravel() == 1]

#Number of inliers:

NUM_INLIERS=pts1.shape[0]

#Generate random colorrs for visualization
colors=np.zeros((NUM_INLIERS,3))
for idx in range(0,NUM_INLIERS):
    colors[idx,0]=np.random.randint(0,255)
    colors[idx,1]=np.random.randint(0,255)
    colors[idx,2]=np.random.randint(0,255)

# Visualize epilines
# Adapted from: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
def drawlines1(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1src.shape
    img1color = cv.cvtColor(img1src, cv.COLOR_GRAY2BGR)
    img2color = cv.cvtColor(img2src, cv.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv.circle(img2color, tuple(pt2), 5, color, -1)
    return img1color, img2color

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

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2, colors, fundamental_matrix)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1, colors, fundamental_matrix.transpose())

# plt.subplot(121), plt.imshow(img5)
# plt.subplot(122), plt.imshow(img3)
# plt.suptitle("Epilines in both images")
# plt.show()

display("Epilines", np.hstack((img5, img3)))
# print(fundamental_matrix)
# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------

origImg1=img1.copy()
origImg2=img2.copy()

#Image sizes:
WIDTH1=img1.shape[1]
HEIGHT1=img1.shape[0]

WIDTH2=img2.shape[1]
HEIGHT2=img2.shape[0]

h1, w1, h2, w2 = HEIGHT1, WIDTH1, HEIGHT2, WIDTH2
_, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1))

# Undistort (rectify) the images and save them
# Adapted from: https://stackoverflow.com/a/62607343
img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))
display("rectified_1.png", img1_rectified)
display("rectified_2.png", img2_rectified)

# Draw the rectified images
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
axes[0].imshow(img1_rectified, cmap="gray")
axes[1].imshow(img2_rectified, cmap="gray")
# for i in range(100, 75, 75):
#     axes[0].axhline(i)
#     axes[1].axhline(i)
axes[0].axhline(250)
axes[1].axhline(250)
axes[0].axhline(350)
axes[1].axhline(350)
axes[0].axhline(450)
axes[1].axhline(450)
axes[0].axhline(550)
axes[1].axhline(550)
axes[0].axhline(750)
axes[1].axhline(750)
plt.suptitle("Rectified images")
plt.savefig("rectified_images.png")
plt.show()