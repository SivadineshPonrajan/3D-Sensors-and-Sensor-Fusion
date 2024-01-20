import os
import shutil
import cv2
import glob
import numpy as np
from termcolor import colored

def camera_calibration(input_path, square_size=10.0, rows=6, cols=8, show_images=False):

    # Discover landscape images in the given path
    image_files = glob.glob(f"{input_path}/*.jpg")
    landscape_images = [cv2.imread(file) for file in image_files if file and cv2.imread(file).shape[1] > cv2.imread(file).shape[0]]

    # Define object points in 3D space
    obj_points = np.zeros((rows * cols, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    # Lists to store 3D and 2D points
    object_points = []
    image_points = []

    # Iterate through each image
    progress_bar_length = len(landscape_images)
    for idx, image in enumerate(landscape_images):
        print(colored(f"Calibrating images: ", 'blue'), colored(f"{idx + 1}", 'green'),colored(f"/", 'blue'),colored(f"{progress_bar_length}", 'red'))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        # If corners are found, store object and image points
        if found:
            object_points.append(obj_points)
            image_points.append(corners)

        if show_images:
            # Draw and display chessboard corners
            cv2.drawChessboardCorners(image, (cols, rows), corners, found)

            # Resize image for visualization
            scale_percent = 30  # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow('Image with Chessboard Corners', resized)
            cv2.waitKey(500)

    if show_images:
        # Close OpenCV windows after displaying all images
        cv2.destroyAllWindows()

    # Calibrate the camera
    ret, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

    # Convert the first rotation vector to a rotation matrix
    R, _ = cv2.Rodrigues(rvecs[0])

    # Form the extrinsic matrix
    extrinsic_matrix = np.hstack((R, tvecs[0]))

    return matrix, distortion, extrinsic_matrix

if __name__ == "__main__":
    input_path = "./data/checkerboard/"
    square_size = 39.5
    rows = 6
    cols = 8
    show_images = True

    intrinsic_matrix, distortion_coefficients, extrinsic_matrix = camera_calibration(input_path, square_size, rows, cols, show_images)

    print(colored(f"Image Calibration Completed", 'green'))

    np.set_printoptions(suppress=True, precision=6)
    print("Camera Matrix:")
    print(intrinsic_matrix)
    print("\nDistortion Coefficients:")
    print(distortion_coefficients)
    print("\nExtrinsic Matrix:")
    print(extrinsic_matrix)


    output_folder = "./Output"
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    output_folder += "/Parameters/"
    os.makedirs(output_folder)

    np.save(output_folder+'intrinsic.npy', intrinsic_matrix)
    np.save(output_folder+'distortion.npy', distortion_coefficients)
    np.save(output_folder+'extrinsic.npy', extrinsic_matrix)

    print(colored(f"Camera parameters saved successfully!", 'green'))