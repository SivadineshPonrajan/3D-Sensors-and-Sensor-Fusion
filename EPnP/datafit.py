import numpy as np
import math


def datafit(pts1,pts2,isScale=True):

    DIM=pts1.shape[1]
    NUMP=pts1.shape[0]



#    print("NUMP:",NUMP)
#    print("DIM:",DIM)

#  % 1. lepes: Sulypontra hozas:
#  if (mode!=0)
#      offset1=mean(pts1')';
#      offset2=mean(pts2')';
#
#      for i=1:columns(pts1)
#         pts1(:,i)-=offset1;
#      end
#
#      for i=1:columns(pts2)
#        pts2(:,i)-=offset2;
#      end
#  endif


#First, calculate the center points. They yield the offsets.

#offset1=offset2=0;
    offset1=np.zeros(DIM)
    offset2=np.zeros(DIM)

#The mean is the optimal offset

    for dim in range(0,DIM):
        offset1[dim]=np.mean(pts1[0:NUMP,dim])
        offset2[dim]=np.mean(pts2[0:NUMP,dim])


    for idx in range(0,NUMP):
        for dim in range(0,DIM):
            pts1[idx,dim]=pts1[idx,dim]-offset1[dim]
            pts2[idx,dim]=pts2[idx,dim]-offset2[dim]



#
#  % 2. lepes: Optimalis elforgatas kiszamitasa
#  H=zeros(3);
#  for i=1:columns(pts2)
#     H+=(pts2(:,i)*(pts1(:,i))');
#  end
#
#  [U S V]=svd(H);
#
#  rot=V*U';

#Optimal rotation

    H=np.zeros((DIM,DIM))




    for idx in range(0,NUMP):
        vec2=np.array((pts2[idx,:DIM]))
        vec2=vec2.reshape(DIM,1)
        vec1=np.array((pts1[idx,:DIM]))
        vec1=vec1.reshape(1,DIM)

        H=np.add(H,np.matmul(vec2,vec1))
#        print(H)

    U, S, Vh = np.linalg.svd(H)

    rot=np.matmul(U,Vh)

#
#  % 3. lepes: Skalazasi kulonbseg megszuntetese:
#  ptsnew=rot*pts2;
#
#  numerator=denominator=0;
#  for i=1:columns(pts1)
#      vec1=pts1(:,i);
#      vec2=ptsnew(:,i);
#      numerator+=vec1'*vec2;
#      denominator+=vec2'*vec2;
#  end


    #ptsNew2=np.matmul(pts2,rot)
    ptsNew2=pts2@rot

    numerator=0.0
    denominator=0.0;
    for idx in range(0,NUMP):
        numerator=numerator+np.dot(pts1[idx,:DIM],ptsNew2[idx,:DIM])
        denominator=denominator+np.dot(ptsNew2[idx,:DIM],ptsNew2[idx,:DIM])


#    print("numerator",numerator)
#    print("denominator",denominator)





#  if (denominator == 0)
#    scale = 1;
#	printf("#######################################\n");
#	printf("#######################################\n");
#	numerator
#	pts1
#	pts2
#	rot
#	ptsnew
#	H
#	U
#	S
#	V
 # else
 #   scale=numerator/denominator;
 # end;

    if isScale:
        if (denominator == 0.0):
            scale=1.0
        else:
            scale=denominator/numerator
    else:
        scale=1.0;

#Calculate error
    scaledNewPts2=np.multiply(ptsNew2,1.0/scale)

    error=0.0;

    errorValues=np.zeros(NUMP);

    for idx in range(0,NUMP):
        vec2=scaledNewPts2[idx,:DIM]
        vec1=pts1[idx,:DIM]
        errorValues[idx]=math.sqrt(np.linalg.norm(np.subtract(vec1,vec2)))
        error+=errorValues[idx]

    error=np.mean(errorValues)

    return offset1,offset2, rot, scale, error, errorValues
