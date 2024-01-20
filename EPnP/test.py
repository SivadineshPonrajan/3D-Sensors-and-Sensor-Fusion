import numpy as np
import random
import math
from datafit import datafit

#Parameters:
#P: point
#C1, ..., C4: reference points
#
#Return:
#barycentric weights in an array
def CalculateBarycentricWeight3D(P,C1,C2,C3,C4):

#3X3 coefficient matrix
    A=np.zeros((3,3))
    A[:,[0]]=C1-C4
    A[:,[1]]=C2-C4
    A[:,[2]]=C3-C4


#Vector for the inhomogeneous linear equation system
    b=np.zeros(3)
    b[0]=P[0]-C4[0];
    b[1]=P[1]-C4[1];
    b[2]=P[2]-C4[2];

    res=np.zeros(4)

#Pseudo inverse
    res[0:3]=np.linalg.inv(A)@b
#Results. Fourth coodinate filled by barycentric constraint
    res[3]=1.0-sum(res)

    return res



#Number of points:
NUMP=16;


NOISE_LEVEL=1.1

#planar object:
#dir1=np.random.rand(3,1)
#dir2=np.random.rand(3,1)
#pt0=np.random.rand(3,1)

#pts=np.zeros((NUMP,3))

#for idx in range(1,NUMP):
#    rand1=np.random.rand()
#    rand2=np.random.rand()
#    pt=pt0+rand1*dir1+rand2*dir2
#    pts[idx,:]=pt.reshape(1,3)

#Spatial object:
pts=np.random.rand(NUMP,3)


#intrinsic (random) parameters
f=500.0*np.random.rand(1)+500.0;
u0=500.0*np.random.rand(1)+500.0;
v0=500.0*np.random.rand(1)+500.0;

#print(f," ",u0," ",v0)

#camera matrix:
K=np.zeros((3,3))
K[0,0]=K[1,1]=f
K[2,2]=1.0
K[0,2]=u0
K[1,2]=v0


#random rotation matrix
R,tmp1,tmp2 = np.linalg.svd(np.random.rand(3,3))


#random translation vector
t=np.random.rand(3,1);
t[2]=t[2]+3.0
#print(t)


#Compose homogeneous 3D points, stack them in a 4XP matrix
ptsHom=np.ones((4,NUMP))

#Copy metric coordinates
ptsHom[0:3,:]=pts.transpose()

#Projection matrix
ProjMtx=np.zeros((3,4))
ProjMtx[0:3,0:3]=K@R
ProjMtx[:,[3]]=K@t


#Projection
pts2DHom=ProjMtx@ptsHom
#print(pts2DHom)

#Homogeneous division
pts2DHom[0,:]=pts2DHom[0,:]/pts2DHom[2,:]
pts2DHom[1,:]=pts2DHom[1,:]/pts2DHom[2,:]
pts2DHom[2,:]=pts2DHom[2,:]/pts2DHom[2,:]


#Add noise:

pts2DHom[0:2,:]=pts2DHom[0:2,:]+np.random.normal(0, NOISE_LEVEL, size=(2, NUMP))


#----- Calculate baricentric coordinates


#First, reference points C1, C2, C3, C4 are determined

#C1 is the mean of 3D points
C1=np.mean(pts,0)



#Further reference points are given by principal directions
#Origo is moved to mean
centeredPts=pts-np.ones((NUMP,1))@C1.reshape(1,3)


eigenvalues, eigenvectors = np.linalg.eig(centeredPts.transpose()@centeredPts)

#print("eigenvalues",eigenvalues)
#print("eigenvectors",eigenvectors)

C2=C1+eigenvectors[:,0].reshape(1,3)
C3=C1+eigenvectors[:,1].reshape(1,3)
C4=C1+eigenvectors[:,2].reshape(1,3)

#print("C1",C1)
#print("C2",C2)
#print("C3",C3)
#print("C4",C4)


#Calculate berycentric coordinates

AllAlphas=np.zeros((NUMP,4));
for idx in range(0,NUMP):
    P=pts[idx,:]
    alphas=CalculateBarycentricWeight3D(P.reshape(3,1),C1.reshape(3,1),C2.reshape(3,1),C3.reshape(3,1),C4.reshape(3,1));
    AllAlphas[idx,:]=alphas.reshape(1,4);



#---- Compose coefficient matrix  M is of size 2Px12

M=np.zeros((2*NUMP,12))


for idx in range(0,NUMP):
#Projected coordinates
    ui=pts2DHom[0,idx]
    vi=pts2DHom[1,idx]


#Barycentric coordinates
    ai1=AllAlphas[idx,0]
    ai2=AllAlphas[idx,1]
    ai3=AllAlphas[idx,2]
    ai4=AllAlphas[idx,3]




    M[2*idx,0]=ai1*f
    M[2*idx,2]=(u0-ui)*ai1

    M[2*idx,3]=ai2*f
    M[2*idx,5]=(u0-ui)*ai2

    M[2*idx,6]=ai3*f
    M[2*idx,8]=(u0-ui)*ai3

    M[2*idx,9]=ai4*f
    M[2*idx,11]=(u0-ui)*ai4

    M[2*idx+1,1]=ai1*f
    M[2*idx+1,2]=(v0-vi)*ai1

    M[2*idx+1,4]=ai2*f
    M[2*idx+1,5]=(v0-vi)*ai2

    M[2*idx+1,7]=ai3*f
    M[2*idx+1,8]=(v0-vi)*ai3

    M[2*idx+1,10]=ai4*f
    M[2*idx+1,11]=(v0-vi)*ai4



#Retrieve nullspace
U, S, Vh = np.linalg.svd(M, full_matrices=True)

#Reference points in camera coordinates
pts1=Vh[11,:].reshape(4,3)


#Reference points in world coordinates
pts2=np.zeros((4,3))

for idx in range(0,3):
    pts2[0,idx]=C1[idx]
    pts2[1,idx]=C2[0][idx]
    pts2[2,idx]=C3[0][idx]
    pts2[3,idx]=C4[0][idx]



#Rotation is given by
offset1,offset2, rot, scale, error, errorValues=datafit(pts2,pts1)


print("Estimated Rot",rot,"\n\n")

print("Ground truth rotation",R)

print("Rotation error in degrees:")
val1=(np.trace(rot.transpose()@R)-1)/2.0
val2=(np.trace(rot.transpose()@(-1.0*R))-1)/2.0

if (val1>0.0):
    print(180*math.acos(val1)/3.1415)
else:
    print(180*math.acos(val2)/3.1415)


