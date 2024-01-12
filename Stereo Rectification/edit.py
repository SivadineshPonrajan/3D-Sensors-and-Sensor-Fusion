import cv2 as cv
import numpy as np

def drawlines(img1, img2, lines, pts1, pts2, colors):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    np.random.seed(0)
    cnt = 0
    for r, pt1, pt2 in zip(lines, img1, img2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

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