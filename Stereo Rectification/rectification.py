import numpy as np
import cv2 as cv
import math
import sys

def display(title, img):
    cv.imshow(title, img)
    key = cv.waitKey(0)
    cv.destroyAllWindows()

#TransformCorners: transforms corner points by a Homography
#
#H: Homography
#(WIDTH,HEIGHT): image sizes
#
#Return: 3x4 matrix with the transformed corners. Points are given in homogeneous coordinates.
def TransformCorners(H,WIDTH,HEIGHT):
    pts=np.ones((3,4))

#Corners:
    pts[0,0]=0;
    pts[1,0]=0;

    pts[0,1]=WIDTH;
    pts[1,1]=0;

    pts[0,2]=0;
    pts[1,2]=HEIGHT;

    pts[0,3]=WIDTH;
    pts[1,3]=HEIGHT;


#Transformation
    pts=H@pts #Corrected
    return pts




#H1: homography for first image
#(WIDTH1,HEIGHT1): size of first image
#
#H2: homography for first image
#(WIDTH2,HEIGHT2): size of first image
#(EXPECTED_WIDTH,EXPECTED_HEIGHT): expected sizef of the resulting rectified images
#
#Return: Correction homography


def FinalScaleOffset(H1,WIDTH1,HEIGHT1,H2,WIDTH2,HEIGHT2,EXPECTED_WIDTH,EXPECTED_HEIGHT):

    ptsIm=np.ones((8,2))


    pts1Hom=TransformCorners(H1,WIDTH1,HEIGHT1)
    pts2Hom=TransformCorners(H2,WIDTH2,HEIGHT2)

    print("pts1Hom:",pts1Hom)
    print("pts2Hom:",pts2Hom)


    print(pts1Hom)

    for idx in range(0,4):
        pt1=pts1Hom[0:2,idx]/pts1Hom[2,idx]
        pt2=pts2Hom[0:2,idx]/pts2Hom[2,idx]

        ptsIm[idx,:]=pt1
        ptsIm[idx+4,:]=pt2 # Corrected

    minPts=ptsIm.min(0)
    maxPts=ptsIm.max(0)

    print("Pts:",ptsIm)

    print("minPts",minPts)
    print("maxPts",maxPts)

    alphaW=EXPECTED_WIDTH/(maxPts[0]-minPts[0])
    betaW=-1.0*alphaW*minPts[0]

    alphaH=EXPECTED_HEIGHT/(maxPts[1]-minPts[1])
    betaH=-1.0*alphaH*minPts[1]

    Corr=np.eye(3)
    Corr[0,0]=alphaW
    Corr[1,1]=alphaH
    Corr[0,2]=betaW
    Corr[1,2]=betaH




    return Corr


#Estimate the verticalScaleand Offset
#Fmod
#pts1Hom,pts2Hom: Points after transformation in homogeneous coordinates
#Return: Desired transformation
#
#The scale and offset is given as y1=alpha*y2+beta2
#

def EstimateVerticalOffsetAndScale(Fmod,pts1Hom,pts2Hom):

#Number of points
    NUMP=pts1Hom.shape[1]

#Each point gives an equation for alpha and beta
    Cfs=np.zeros((NUMP,2))
    right=np.zeros((NUMP,1))

    for idx in range(0,NUMP):
#Get original homogeneous coordinates:
        p1Hom=pts1Hom[:,idx]
        p2Hom=pts2Hom[:,idx]
#y2:
        Cfs[idx,0]=p2Hom[1]/p2Hom[2]  # coordinate x
        Cfs[idx,1]=1.0

        right[idx,0]=p1Hom[1]/p1Hom[2]

#Over-determined inhomogeneous linear system of equations
    params=np.linalg.inv(Cfs.transpose()@Cfs)@Cfs.transpose()@right
    alpha=params[0]
    beta=params[1]


#First transformation is the identity, second one is the copmputed scale/offset
    H1=np.eye((3))
    H2=np.eye((3))


    H2[1,1]=alpha
    H2[1,2]=beta

    return H1,H2


#This method is the essence of rectification
#It moves the epipole to the infinity
#
#Input:
#
#epipole: computed epipole. It is input even if it can be computed as the kernel of F.
#F: fundamental matrix
#
#Return:
#Resulting homography

def EstimateRectifyingHomography(epipole,F):


#In the first step, the vertical coordinate of the epipole should be moved to y=0
    tform1=np.eye(3)
    tform1[1,2]=-epipole[1]

    #Tform 2 is the projective warp
    tform2=np.eye(3)
    tform2[2,0]=-1.0/epipole[0]

    H=tform2@tform1


    return H


# fileName1=sys.argv[1]
# fileName2=sys.argv[2]

fileName1="./images/a.png"
fileName2="./images/b.png"

img1 = cv.imread(fileName1)  #queryimage # left image
img2 = cv.imread(fileName2) #trainimage # right image #Corrected

#img1 = cv.imread('Dev0_Image_w1920_h1200_fn400.jpg')  #queryimage # left image
#img2 = cv.imread('Dev1_Image_w1920_h1200_fn400.jpg') #trainimage # right image


#Copy original images. It will be saved later

origImg1=img1.copy()
origImg2=img2.copy()



#Image sizes:
WIDTH1=img1.shape[1]
HEIGHT1=img1.shape[0]

WIDTH2=img2.shape[1]
HEIGHT2=img2.shape[0]


#White:
color=np.ones(3)


#Load fundamental matrix and matched (inlier) feature point:
F=np.loadtxt("F_RANSAC.mat")
pts1=np.loadtxt("pts1.mat")
pts2=np.loadtxt("pts2.mat")


#Compute epipoles

u,s,vt= np.linalg.svd(F)

#Left and right kernels give epipoles
epipole1=vt[2,:]
epipole2=u[:,2]

#Homogeneous division required!
epipole1=epipole1/epipole1[2]
epipole2=epipole2/epipole2[2]


#For debug:
print("Epipole1:",epipole1[0:2])
print("Epipole2:",epipole2[0:2])



##Parameter to visualize epipoles. Distance between drawn lines
SKIP=10

#Number of drawn epilines
NUM=int(math.floor(HEIGHT1/SKIP))



#Take and draw epipolar lines in the second image
for idx in range(0,NUM):
    ptHom=np.ones(3)
#Center of width
    ptHom[0]=int(WIDTH1/2)
#Uniform sampling:
    ptHom[1]=idx*SKIP

#Epipolar line in the second image
    line2=F@ptHom

#Get endpoints of line in the screen.
    x0,y0 = map(int, [0, -line2[2]/line2[1] ])
    x1,y1 = map(int, [WIDTH2, -(line2[2]+line2[0]*WIDTH2)/line2[1] ])
#Draw it
    img2 = cv.line(img2, (x0,y0), (x1,y1), color,1)

# display("img2",img2)

#Take and draw epipolar lines in the first image image

NUM=int(math.floor(HEIGHT2/SKIP))

for idx in range(0,NUM):
    ptHom=np.ones(3)
    ptHom[1]=idx*SKIP
    ptHom[0]=int(WIDTH2/2)

#F.transpose() gives the inverz of the fundamental matrix!
    line1=F.transpose()@ptHom

    x0,y0 = map(int, [0, -line1[2]/line1[1] ])
    x1,y1 = map(int, [WIDTH1, -(line1[2]+line1[0]*WIDTH1)/line1[1] ])
    img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)


# display("img1",img1) 

#Write images for debgging purposes

cv.imwrite("res1.png",img1)
cv.imwrite("res2.png",img2)
display("Lines Lines",np.hstack([img1,img2]))

#Move epipoles to the infinity:
H1=EstimateRectifyingHomography(epipole1,F)
H2=EstimateRectifyingHomography(epipole2,F.transpose())


tmp1 = cv.warpPerspective(origImg1, H1,(WIDTH1,HEIGHT1))
tmp2 = cv.warpPerspective(origImg2, H2,(WIDTH2,HEIGHT2))

cv.imwrite("tmp1.png",tmp1)
cv.imwrite("tmp2.png",tmp2)

display("tmp", np.hstack([tmp1, tmp2]))


#Modified fundamental matrix
Fmod=np.linalg.inv(H2.transpose())@F@np.linalg.inv(H1)



#Transform the point coordinates by the obtrained homography

#Number of points
NUMP=pts1.shape[0]

#Point should be considered in its homogeneous form
pts1Hom=np.ones((NUMP,3))
pts2Hom=np.ones((NUMP,3))




for idx in range(0,NUMP):
    pts1Hom[idx,0]=pts1[idx,0]
    pts1Hom[idx,1]=pts1[idx,1]

    pts2Hom[idx,0]=pts2[idx,0]
    pts2Hom[idx,1]=pts2[idx,1]



pts1Hom=H1@(pts1Hom.transpose())
pts2Hom=H2@(pts2Hom.transpose())


#Make horizontal epipolar lines parallel by estimating a scale and an offset
H1second,H2second=EstimateVerticalOffsetAndScale(Fmod,pts1Hom,pts2Hom)

H1=H1second@H1
H2=H2second@H2



#Finally, move the image to the center and resize the images

EXPECTED_WIDTH=1000
EXPECTED_HEIGHT=700


Corr= FinalScaleOffset(H1,WIDTH1,HEIGHT1,H2,WIDTH2,HEIGHT2,EXPECTED_WIDTH,EXPECTED_HEIGHT)


H1=Corr@H1
H2=Corr@H2



#Finally, draw image for debugging purposes:


transformed1 = cv.warpPerspective(img1, H1,(EXPECTED_WIDTH,EXPECTED_HEIGHT))
transformed2 = cv.warpPerspective(img2, H2,(EXPECTED_WIDTH,EXPECTED_HEIGHT))

cv.imwrite("transformed1.png",transformed1)
cv.imwrite("transformed2.png",transformed2)

#Rectify original images.

result1 = cv.warpPerspective(origImg1, H1,(EXPECTED_WIDTH,EXPECTED_HEIGHT))
result2 = cv.warpPerspective(origImg2, H2,(EXPECTED_WIDTH,EXPECTED_HEIGHT))

cv.imwrite("final1.png",result1)
cv.imwrite("final2.png",result2)

#Concatenate resulting images:
final=cv.hconcat([result1,result2])
cv.imwrite("finalRectified.png",final)
display("finalRectified.png",final)

final=cv.hconcat([origImg1,origImg2])
cv.imwrite("origRectified.png",final)
display("origRectified.png",final)
