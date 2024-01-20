#Sample project to interpolate a GPS that contains 4Hz data when the sampler operated by 1HZ
#Points should be interpolated

import matplotlib.pyplot as plt
import numpy as np
import math
import sys


#Hanif's code to load data from text file
def load_points_file(filename):

    points = []
    with open(filename, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            x, y, _ = map(float, line.split())
            points.append((x, y))

    return points



#4Hz/1HZ = 4
SKIP=4


#Load Data
data2D = np.array(load_points_file("Trajectory.xyz"))

pt_size=data2D.shape

#Number of points
NUM=pt_size[0]



rangeObject = range(0,NUM,SKIP)


filteredPos=np.zeros((len(rangeObject),2))

counter=0;

for idx in rangeObject:
    posX=data2D[idx,0]
    posY=data2D[idx,1]
    filteredPos[counter,0]=posX
    filteredPos[counter,1]=posY
    counter=counter+1



NUM=len(rangeObject)


#Positions are stored

origpos=np.zeros((NUM,2));

#angles are also calculated
angles=np.zeros(NUM);

#new positions
poses=np.zeros((NUM,2));


#Endpoints are simply copied
poses[0,0]=filteredPos[0,0];
poses[0,1]=filteredPos[0,1];

poses[NUM-1,0]=filteredPos[NUM-1,0];
poses[NUM-1,1]=filteredPos[NUM-1,1];



rangeObject=range(1,(NUM-1))

counter=0

for idx in rangeObject:


    posPrev=np.zeros(2)
    posCurr=np.zeros(2)
    posNext=np.zeros(2)

#Previous
    posPrev[0]=filteredPos[idx-1,0]
    posPrev[1]=filteredPos[idx-1,1]

#Current
    posCurr[0]=filteredPos[idx,0]
    posCurr[1]=filteredPos[idx,1]

#Next
    posNext[0]=filteredPos[idx+1,0]
    posNext[1]=filteredPos[idx+1,1]


#Vec1: vector to current position
    vec1=posCurr-posPrev
#Vec2: vector from current position
    vec2=posNext-posCurr
#Orthogonal vector to Vec1
    ortVec1=np.array([-vec1[1],vec1[0]])




#    if ((norm(vec1)!=0.0)&&(norm(vec2)!=0.0))

    if (((vec1[0]!=0.0) or (vec1[1]!=0.0)) and ((vec2[0]!=0.0) or (vec2[1]!=0.0))):
#       dotProd=(vec1*vec2')/(norm(vec1)*norm(vec2));

#Calculate the angle between vec1 and vec2
        dotProd=(np.dot(vec1,vec2))
        dotProd=dotProd/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

        if (dotProd>1.0):
            dotProd=1.0
        if (dotProd<-1.0):
            dotProd=-1.0

        angle=math.acos(dotProd)

#This is the sign which represents if vec 2 goes down or up-

        tmpSign=np.dot(ortVec1,vec2)


        origpos[counter,0]=posCurr[0]
        origpos[counter,1]=posCurr[1]
        counter=counter+1


        #Up and down directions determine the sign of the angle

        if (tmpSign>0):
            angles[idx]=angle;
        else:
            angles[idx]=-1.0*angle;

#        poses(idx,:)=posCurr;
        poses[idx,0]=posCurr[0];
        poses[idx,1]=posCurr[1];



#If all vevtors are zero then GPS remains in the same position
    elif ((vec1[0]==0.0) and (vec1[1]==0.0) and (vec2[0]==0.0) and (vec2[1]==0.0)):
        poses[idx,0]=posCurr[0];
        poses[idx,1]=posCurr[1];


#If only vec1 is zero
    elif ((vec1[0]==0.0) and (vec1[1]==0.0)):
        poses[idx,0]=posCurr[0];
        poses[idx,1]=posCurr[1];

#If only vec2 is zero. It can be combined with the previous steps
    else:
        poses[idx,0]=posCurr[0];
        poses[idx,1]=posCurr[1];

NUMP=NUM-1



#Interpolation starts here.

#Data for the new positions
newPos=np.zeros((SKIP*NUMP-SKIP,2))

for idx in range(0,NUMP-1):



#Vector between adjacent positions
    diff=np.zeros(2)
    diff[0]=poses[idx+1,0]-poses[idx,0]
    diff[1]=poses[idx+1,1]-poses[idx,1]


#Mid-point
    centerDiff=np.zeros(2)
    centerDiff[0]=poses[idx,0]+diff[0]/2.0
    centerDiff[1]=poses[idx,1]+diff[1]/2.0


#Half angle
    angle=(angles[idx]/2.0)

#Start and end locations

    startP=np.zeros(2)
    endP=np.zeros(2)

    startP[0]=poses[idx,0]
    startP[1]=poses[idx,1]

    endP[0]=poses[idx+1,0]
    endP[1]=poses[idx+1,1]



#For non-small angles
    if (abs(angle)>1e-7):


#Norm of the vector: length between adjacent locations
        diffNorm=np.linalg.norm(diff)

        L=diffNorm/2.0

#Orthogonal direction as a unit vector

        n=np.zeros(2)

        n[0]=-1.0*diff[1]/diffNorm
        n[1]=diff[0]/diffNorm


#
        a=L/math.tan(angle)

#       Centre point of circle
        center=np.zeros(2)
        center[0]=centerDiff[0]+n[0]*a
        center[1]=centerDiff[1]+n[1]*a

#Vector between circle centre and initial point
        tmpVec=np.zeros(2)
        tmpVec2=np.zeros(2)
        tmpVec[0]=poses[idx,0]-center[0]
        tmpVec[1]=poses[idx,1]-center[1]
##        tmpVec2[0]=poses[idx+1,0]-center[0]
##        tmpVec2[1]=poses[idx+1,1]-center[1]

        radius=np.linalg.norm(tmpVec)

#            vv=startP-center
#            if isinf(vv(1))
#                "pause"
#                pause
#            endif


#Vector between first point and circle center
        vv=np.zeros(2)
        vv[0]=startP[0]-center[0]
        vv[1]=startP[1]-center[1]


#
#Angle of the direction vector w.r.t. horiZontal axis
        coordAngle=math.atan2(vv[1],vv[0])


#In the poitns, the locations should be interpolated

        for idx2 in range(0,SKIP):

#Angle with the apprOpriate ratio
            subangle=2*idx2*angle/SKIP;

#Circle inerpolation
            newP=np.zeros(2)
            newP[0]=radius*math.cos(subangle);
            newP[1]=radius*math.sin(subangle);




#A 2D rotation matrix
            rot=np.zeros((2,2))
            rot[0,0]=math.cos(coordAngle)
            rot[0,1]=-1.0*math.sin(coordAngle)
            rot[1,0]=math.sin(coordAngle)
            rot[1,1]=math.cos(coordAngle)


#Place the result into the coordinate system

            newP2=np.add(np.matmul(rot, newP),center)
#            newpos=[newpos;newP'];
            newPos[4*idx+idx2,0]=newP2[0]
            newPos[4*idx+idx2,1]=newP2[1]

#if angle is (close to) zero
    else:
#Linear interpolation (draw a straight line segment)
        for idx2 in range(0,SKIP):
            newPos[4*idx+idx2,0]=(startP[0]*idx2+endP[0]*(SKIP-idx2))/SKIP
            newPos[4*idx+idx2,1]=(startP[1]*idx2+endP[1]*(SKIP-idx2))/SKIP



#Finally, visualize the result by matplotlib

plt.plot(origpos[1:counter,0],origpos[1:counter,1], "r-o")
#
plt.plot(newPos[:,0], newPos[:,1], "g-x")

#plt.plot(points)
#plt.ylabel('some numbers')
plt.show()

np.savetxt('newPos.txt', newPos)
np.savetxt('origpos.txt', origpos)

