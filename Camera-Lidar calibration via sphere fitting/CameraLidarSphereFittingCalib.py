import os
import cv2
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os

# Sphere radius in meters
sphere_radius = 0.25 

# ***************************************************************************
# ***************************************************************************
# ***************************************************************************

def lstsq_sphere_fitting_fixed_radius(pos_xyz, fixed_radius):
    # Add a column of ones to pos_xyz to construct matrix A
    row_num = pos_xyz.shape[0]
    A = np.ones((row_num, 4))
    A[:, 0:3] = pos_xyz

    # Construct vector f
    f = np.sum(np.multiply(pos_xyz, pos_xyz), axis=1) - fixed_radius**2
    
    sol, residuals, rank, singval = np.linalg.lstsq(A, f)

    # Use the fixed radius
    radius = fixed_radius

    return radius, sol[0] / 2.0, sol[1] / 2.0, sol[2] / 2.0

# ***************************************************************************

def lstsq_sphere_fitting(pos_xyz):
    row_num = pos_xyz.shape[0]
    A = np.ones((row_num, 4))
    A[:,0:3] = pos_xyz

    # construct vector f
    f = np.sum(np.multiply(pos_xyz, pos_xyz), axis=1)
    
    sol, residules, rank, singval = np.linalg.lstsq(A, f, rcond=None)

    # solve the radius
    radius = math.sqrt((sol[0]*sol[0]/4.0)+(sol[1]*sol[1]/4.0)+(sol[2]*sol[2]/4.0)+sol[3])

    return radius, sol[0]/2.0, sol[1]/2.0, sol[2]/2.0    

# ***************************************************************************

def  PLYWriter(filename, pointcloud):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex "+str(len(pointcloud))+"\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i, point in enumerate(pointcloud):
            f.write('%f %f %f %d %d %d\n' % (point[0], point[1], point[2], point[3], point[4], point[5]))

    print("PLY file saved successfully as "+filename)

# ***************************************************************************

def lidar_sphere_fitting(frame, sphere_radius, visualization):

    if visualization == "show": 
        segmented_lidar = True
        visualize_sphere = True
        visualize_lidar = True
    else:
        segmented_lidar = False
        visualize_sphere = False
        visualize_lidar = False

    lidar_data_path = "./data/gombcal_1_cartesians/test_fn"+str(frame)+".xyz"
    # Load the XYZ file from Lidar
    lidar_data = np.loadtxt(lidar_data_path)
    distances = np.linalg.norm(lidar_data[:, :3], axis=1)

    if frame > 40 and frame <= 50:
        filtered_points = lidar_data[distances <= 4]
        filtered_points = filtered_points[(filtered_points[:, 0] > -1) & (filtered_points[:, 0] < 2)]
        filtered_points = filtered_points[(filtered_points[:, 1] > 1) & (filtered_points[:, 1] < 2)]
        filtered_points = filtered_points[filtered_points[:, 2] < 0.4]
    elif frame > 50 and frame <= 60:
        filtered_points = lidar_data[distances <= 3]
        filtered_points = filtered_points[(filtered_points[:, 0] < 2) & (filtered_points[:, 0] > -1)]
        filtered_points = filtered_points[(filtered_points[:, 1] < 1.3) & (filtered_points[:, 1] > 0)]
        filtered_points = filtered_points[filtered_points[:, 2] < 0.20]
    elif frame > 60 and frame <= 70:
        filtered_points = lidar_data[distances <= 3]
        filtered_points = filtered_points[(filtered_points[:, 0] < 2) & (filtered_points[:, 0] > -1)]
        filtered_points = filtered_points[(filtered_points[:, 1] < 1.3) & (filtered_points[:, 1] > 0)]
        filtered_points = filtered_points[filtered_points[:, 2] < 0.20]
    elif frame > 70 and frame < 80:
        filtered_points = lidar_data[distances <= 3]
        filtered_points = filtered_points[(filtered_points[:, 0] < 2) & (filtered_points[:, 0] > -1)]
        filtered_points = filtered_points[(filtered_points[:, 1] < 1.5) & (filtered_points[:, 1] > 0)]
        filtered_points = filtered_points[filtered_points[:, 2] < 0.20]
    elif frame >= 80 and frame < 140:
        filtered_points = lidar_data[distances <= 3]
        filtered_points = filtered_points[(filtered_points[:, 0] > 0) & (filtered_points[:, 1] > 0)]
        filtered_points = filtered_points[filtered_points[:, 2] < 0.10]
    elif frame > 140:
        filtered_points = lidar_data[distances <= 3]
        filtered_points = filtered_points[(filtered_points[:, 0] > 0) & (filtered_points[:, 1] > 0) & (filtered_points[:, 1] < 2)]
        filtered_points = filtered_points[filtered_points[:, 2] < 0.40]
    else:
        filtered_points = lidar_data[distances <= 4]
        filtered_points = filtered_points[(filtered_points[:, 0] > -0.5) & (filtered_points[:, 0] < 0.5)]
        filtered_points = filtered_points[(filtered_points[:, 1] > 1) & (filtered_points[:, 1] < 2)]

    # Extract spatial coordinates (x, y, z)
    spatial_coordinates = filtered_points[:, :3]

    # Use DBSCAN for clustering
    dbscan = DBSCAN(eps=0.2, min_samples=7)  # Adjust parameters as needed
    labels = dbscan.fit_predict(spatial_coordinates)

    # Identify the cluster corresponding to the sphere
    sphere_cluster_label = -1  # Initialize to a value that is not a valid label
    max_cluster_size = 0

    for label in np.unique(labels):
        if label == -1:
            continue  # Skip noisy points
        cluster_size = np.sum(labels == label)
        if cluster_size > max_cluster_size:
            max_cluster_size = cluster_size
            sphere_cluster_label = label

    # Extract points belonging to the identified sphere cluster
    sphere_cluster_points = spatial_coordinates[labels == sphere_cluster_label]

    if segmented_lidar:
        # Visualize the identified sphere cluster
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sphere_cluster_points[:, 0], sphere_cluster_points[:, 1], sphere_cluster_points[:, 2], label='Segmented Point Clouds')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Sphere Cluster Visualization - Frame ' + str(frame))
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    # Fit a sphere to the identified cluster
    r_fit, xc_fit, yc_fit, zc_fit = lstsq_sphere_fitting(sphere_cluster_points[:, :3])
    # r_fit, xc_fit, yc_fit, zc_fit = lstsq_sphere_fitting_fixed_radius(sphere_cluster_points[:, :3], sphere_radius)

    # print("Radius:", r_fit)

    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    x_sphere = xc_fit + r_fit * np.sin(phi) * np.cos(theta)
    y_sphere = yc_fit + r_fit * np.sin(phi) * np.sin(theta)
    z_sphere = zc_fit + r_fit * np.cos(phi)

    if visualize_sphere:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sphere_cluster_points[:, 0], sphere_cluster_points[:, 1], sphere_cluster_points[:, 2], c='b', marker='.', label='Segmented Cloud points')
        ax.scatter(x_sphere, y_sphere, z_sphere, c='r', marker='.', label='Imaginary Fitted Sphere', alpha=0.5)  # Highlight in red
        ax.scatter(xc_fit, yc_fit, zc_fit, c='g', marker='o', label='center', alpha=0.8)  # Highlight in red
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        ax.legend()
        ax.set_title('Sphere Cluster Visualization - Frame ' + str(frame))
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    sphere_cluster_points = lidar_data[:, :3]
    distances = np.linalg.norm(sphere_cluster_points - np.array([xc_fit, yc_fit, zc_fit]), axis=1)
    
    color_condition = np.abs(distances - r_fit) <= 0.03
    region_condition = (distances <= 1)

    color_points = sphere_cluster_points[color_condition]
    sphere_cluster_points = sphere_cluster_points[np.logical_and(region_condition, ~color_condition)]

    if visualize_lidar:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sphere_cluster_points[:, 0], sphere_cluster_points[:, 1], sphere_cluster_points[:, 2], c='b', marker='.', label='Noisy Cloud Points')
        ax.scatter(color_points[:, 0], color_points[:, 1], color_points[:, 2], c='r', marker='o', label='Sphere fitted point clouds')
        ax.scatter(xc_fit, yc_fit, zc_fit, c='g', marker='o', label='Center of Fitted Sphere', alpha=0.8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        ax.legend()
        ax.set_title('Sphere Cluster Visualization - Frame ' + str(frame))
        plt.axis('equal')
        plt.grid(True)
        plt.show()
    
    # Export color points to PLY file
    ply_file = "./Result-PLY/Output_test_fn"+str(frame)+"_color.ply"

    sphere_cluster_points = lidar_data[:, :3]
    distances = np.linalg.norm(sphere_cluster_points - np.array([xc_fit, yc_fit, zc_fit]), axis=1)

    color_points = sphere_cluster_points[color_condition]
    colors = np.array([[0, 255, 0] for _ in range(len(color_points))])
    color_points = np.hstack((color_points, colors))

    other_points = sphere_cluster_points[~color_condition]
    colors = np.array([[0, 0, 255] for _ in range(len(other_points))])
    other_points = np.hstack((other_points, colors))

    color_points = np.vstack((color_points, other_points))

    PLYWriter(ply_file, color_points)

    return np.array([xc_fit, yc_fit, zc_fit])

# ***************************************************************************
# ***************************************************************************
# ***************************************************************************

def display(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ***************************************************************************

def XYZcenterFromEllipse(point, sphere_radius):
    point_3d = np.insert(point, 1, 1, axis=1)
    point_3d = np.sqrt(np.ones((len(point_3d), 1)) / np.sum(point_3d * point_3d, axis=1, keepdims=True)) @ np.ones((1, 3)) * point_3d

    # Find the axis drirection and angle of the cone
    wPerCosAlpha = np.linalg.lstsq(point_3d, np.ones(len(point_3d)), rcond=None)[0]
    cosAlpha = 1 / np.linalg.norm(wPerCosAlpha)
    w = cosAlpha * wPerCosAlpha

    # Direction vec of the cone axis
    d = sphere_radius / np.sqrt(1 - cosAlpha**2)

    # Calculate center
    S0 = d * w

    return S0

# ***************************************************************************

def pixel_to_meters(pixel_coords, K):
    # Extract focal lengths and principal point from the camera intrinsic matrix
    fu = K[0, 0]
    fv = K[1, 1]
    u0 = K[0, 2]
    v0 = K[1, 2]

    # Convert pixel coordinates to normalized image coordinates
    normalized_coords = (pixel_coords - np.array([u0, v0])) / np.array([fu, fv])

    return normalized_coords

# ***************************************************************************

def image_sphere_fitting(frame, sphere_radius, visualization):

    position_image = True 
    if visualization == "show":
        visualize_images = True
    else:
        visualize_images = False

    cam0 = "./data/gombcal_1_pictures/Dev0_Image_w1920_h1200_fn"+str(frame)+".jpg"
    cam1 = "./data/gombcal_1_pictures/Dev1_Image_w1920_h1200_fn"+str(frame)+".jpg"

    # Camera intrinsic matrix
    K = np.array([[1250, 0, 960], [0, 1250, 600], [0, 0, 1]])

    image = cv2.imread(cam0)

    image = cv2.undistort(image, K, None)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for yellow color in HSV
    lower_yellow = np.array([15, 30, 30])
    upper_yellow = np.array([255, 255, 255])

    # Create a binary mask for the yellow color
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel, iterations=5)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel, iterations=5)

    # Apply the mask to the original image
    segmented_image = cv2.bitwise_and(image, image, mask=yellow_mask)

    # Convert the segmented image to grayscale
    gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_segmented, 50, 150)
    edges_3d = cv2.cvtColor(gray_segmented, cv2.COLOR_GRAY2BGR)
    # cv2.RETR_EXTERNA - coz only extreme outer contours are needed
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(image, [largest_contour], 0, (0, 255, 0), 2)

    e = cv2.fitEllipse(largest_contour)

    cv2.ellipse(image, e, (0, 0, 255), 2)
    cv2.ellipse(edges_3d, e, (0, 0, 255), 2)
    cv2.ellipse(segmented_image, e, (0, 0, 255), 2)
    # cv2.circle(image, (int(e[0][0]), int(e[0][1])), 5, (0, 0, 225), -1)

    stack = np.hstack((image, edges_3d, segmented_image))
    if visualize_images:
        display("Fitted Circle - "+str(frame), stack)

    # # Extract ellipse parameters
    ellipse_center, (major_axis, minor_axis), ellipse_angle = e

    # print(ellipse_center)
    cv2.circle(image, (int(ellipse_center[0]), int(ellipse_center[1])), 3, (0, 255, 0), -1)

    fitted_sphere_in_meters = pixel_to_meters(largest_contour,K)

    c = XYZcenterFromEllipse(np.squeeze(fitted_sphere_in_meters), sphere_radius)
    
    if position_image:
        cam1_image = cv2.imread(cam1)
        cam1_image = cv2.undistort(cam1_image, K, None)
        cam_image = np.hstack((cam1_image, image))
        display("Fitted Circle - "+str(frame), cam_image)
    # return XY_points
    return c

# ***************************************************************************
# ***************************************************************************
# ***************************************************************************

def point_set_registration(pts1, pts2):
    offset1 = np.mean(pts1, axis=0)
    offset2 = np.mean(pts2, axis=0)

    pts1 = pts1 - offset1
    pts2 = pts2 - offset2

    # Normalize for robustness
    scale1 = np.std(pts1)
    scale2 = np.std(pts2)

    pts1 /= scale1
    pts2 /= scale2

    H = np.zeros((pts1.shape[1], pts1.shape[1]))

    for i in range(len(pts2)):
        H = H + np.outer(pts1[i, :], pts2[i, :])

    U, _, Vt = np.linalg.svd(H)
    V = Vt.T
    rot = np.dot(V, U.T)
    trans = offset2 - rot@offset1

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rot
    transformation_matrix[:3, 3] = trans.flatten()

    return rot, trans, transformation_matrix

# ***************************************************************************
# ***************************************************************************
# ***************************************************************************


def main():
    lidar_center_points = []
    cam_center_points = []

    final_list = [38, 55, 57, 60, 80, 82, 144]
    final_list = [38, 43, 55, 57, 60, 80, 82, 144]
    final_list = [38, 57, 60, 82, 144, 149]

    frame_list = final_list
    frame_list = [38, 144]

    visualization = "show" if len(frame_list) <= 3 else "run"

    folder_path = "./Result-PLY"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

    for counter in range(len(frame_list)):
        frame = frame_list[counter]

        sphere_center_lidar = lidar_sphere_fitting(frame, sphere_radius, visualization)
        sphere_center_cam = image_sphere_fitting(frame, sphere_radius, visualization)

        lidar_center_points.append(sphere_center_lidar)
        cam_center_points.append(sphere_center_cam)

        print("Sphere Center from Lidar:", sphere_center_lidar)
        print("Sphere Center from Camera:", sphere_center_cam)
        print("******************************")

    lidar_center_points = np.array(lidar_center_points)
    cam_center_points = np.array(cam_center_points)

    Rot, Trans, transformation_matrix = point_set_registration(cam_center_points, lidar_center_points)
    pts_homogeneous = np.column_stack((cam_center_points, np.ones(len(cam_center_points))))
    pts_transformed_homogeneous = np.dot(transformation_matrix, pts_homogeneous.T).T
    predicted_points = pts_transformed_homogeneous[:, :3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(lidar_center_points[:, 0], lidar_center_points[:, 1], lidar_center_points[:, 2], c='green', label='Lidar Center Points')
    for i, (x, y, z) in enumerate(lidar_center_points):
        ax.text(x, y, z, f'  {frame_list[i]}', color='green', fontsize=8)
    ax.scatter(cam_center_points[:, 0], cam_center_points[:, 1], cam_center_points[:, 2], c='red', label='Cam Center Points')
    # for i, (x, y, z) in enumerate(cam_center_points):
    #     ax.text(x, y, z, f'  {frame_list[i]}', color='red', fontsize=8)
    ax.scatter(predicted_points[:, 0], predicted_points[:, 1], predicted_points[:, 2], c='blue', label='Transformed Cam Center Points')
    # for i, (x, y, z) in enumerate(predicted_points):
    #     ax.text(x, y, z, f'  {frame_list[i]}', color='blue', fontsize=8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.axis('equal')
    plt.grid(True)
    ax.legend()
    plt.show()

    before_transformation_mean_error = np.mean(np.linalg.norm(lidar_center_points - cam_center_points, axis=1))
    print(f"Before Transformation - Mean Error: {before_transformation_mean_error} meter")
    after_transformation_mean_error = np.mean(np.linalg.norm(lidar_center_points - predicted_points, axis=1))
    print(f"After Transformation - Mean Error: {after_transformation_mean_error} meter")


if __name__ == "__main__":
    main()