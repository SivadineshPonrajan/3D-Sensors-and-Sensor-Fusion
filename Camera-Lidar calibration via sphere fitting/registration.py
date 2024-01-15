import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

def point_set_registration(pts1, pts2):
    offset1 = np.mean(pts1, axis=0)
    offset2 = np.mean(pts2, axis=0)

    pts1 = pts1 - offset1
    pts2 = pts2 - offset2

    H = np.zeros((pts1.shape[1], pts1.shape[1]))

    for i in range(len(pts2)):
        H = H + np.outer(pts1[i, :], pts2[i, :])

    U, _, Vt = np.linalg.svd(H)
    V = Vt.T
    rot = np.dot(V, U.T)
    trans = offset2 - rot@offset1

    return rot, trans

import numpy as np

lidar_center_points = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
cam_center_points = np.array([[1, 1, 2], [2, 2, 3], [3, 3, 4]])

lidar_center_points = np.array(lidar_center_points)
cam_center_points = np.array(cam_center_points)

# Rot, Trans = point_set_registration_updated(cam_center_points, lidar_center_points)
Rot, Trans = point_set_registration(cam_center_points, lidar_center_points)
# Rot, Trans = arun(lidar_center_points, cam_center_points)

transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = Rot
transformation_matrix[:3, 3] = Trans.flatten()

# print(transformation_matrix)
# print(transformation_matrix.shape)

# Add an extra column to the points
pts_homogeneous = np.column_stack((cam_center_points, np.ones(len(cam_center_points))))

# Apply the transformation matrix
pts_transformed_homogeneous = np.dot(transformation_matrix, pts_homogeneous.T).T

# Remove the extra column
predicted_points = pts_transformed_homogeneous[:, :3]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(lidar_center_points[:, 0], lidar_center_points[:, 1], lidar_center_points[:, 2], c='green', label='Lidar Center Points')
ax.scatter(cam_center_points[:, 0], cam_center_points[:, 1], cam_center_points[:, 2], c='red', label='Cam Center Points')
ax.scatter(predicted_points[:, 0], predicted_points[:, 1], predicted_points[:, 2], c='blue', label='Predicted Cam Center Points')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()