import numpy as np
import matplotlib.pyplot as plt
import cv2
from datafit import *

def create_box_model(width, depth, height, visualize=False):

    # Define the vertices of the box
    box_vertices = np.array([
        (0, 0, 0),           # Front Lower left corner
        (width, 0, 0),       # Front Lower right corner
        (width, 0, height),  # Front Upper right corner
        (0, 0, height),      # Front Upper left corner
        (0, depth, height),  # Top Upper left corner
        (width, depth, height),  # Top Upper right corner
        (width, depth, 0),   # Back Bottom right corner
        (0, depth, 0)        # Back Bottom left corner
    ])

    # Define the 12 edges of the rectangular box
    box_edges = [[0, 1], [1, 2], [2, 3], [3, 0],
                 [4, 5], [5, 6], [6, 7], [7, 4],
                 [0, 7], [1, 6], [2, 5], [3, 4]]

    if visualize:
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the vertices with annotations
        for i, (x, y, z) in enumerate(box_vertices):
            ax.scatter3D(x, y, z, label=f'Vertex {i}', color='r')
            ax.text(x, y, z, f'{i}', fontsize=12, ha='left')

        # Plot the edges
        for edge in box_edges:
            ax.plot3D(*zip(*box_vertices[edge]), color='b')

        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Show the plot
        plt.show()

    return box_vertices, box_edges

def get_user_selected_points(image_path, n_points):
    global selected_points
    selected_points = []
    
    def handle_mouse_click(event, x, y, flags, params):
        # Check if the left mouse button was clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert the coordinates back to the original image size
            original_x = int(x)
            original_y = int(y)
            # Store the coordinates of the selected point
            selected_points.append((original_x, original_y))
            # Draw the point on the image
            cv2.drawMarker(image, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 30, 6)
            cv2.imshow("Image", image)

    # Load the image
    image = cv2.imread(image_path)

    # Scale down image for display
    screen_res = 1920, 1080  # Replace with your screen resolution
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)

    # Resize the image for display
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', window_width, window_height)

    # Display the image
    cv2.imshow("Image", image)

    # Set the mouse callback function to `handle_mouse_click`
    cv2.setMouseCallback("Image", handle_mouse_click)

    # Wait for the user to select n_points
    while len(selected_points) < n_points:
        # Wait for a key press
        cv2.waitKey(13)

    # Close the image window
    cv2.destroyAllWindows()

    # Store the pixel coordinates of the selected points
    return np.array(selected_points, dtype="double")

if __name__ == '__main__':

    # Define the dimensions of the box
    box_width = 220.0
    box_depth = 210.0
    box_height = 137.0
    input_image_path = "./data/input/1.jpg"
    num_selected_points = 6
    use_opencv_epnp = True

    # Create the box model
    box_vertices, box_edges = create_box_model(box_width, box_depth, box_height, visualize=False)

    # Define the pixel coordinates of the 6 points
    user_selected_points = get_user_selected_points(input_image_path, num_selected_points)

    print("User Selected Points: \n", user_selected_points)

    # Load camera parameters
    intrinsic_matrix = np.load("./Output/Parameters/intrinsic.npy")
    distortion_coefficients = np.load("./Output/Parameters/distortion.npy")

    distortion_coefficients = np.zeros((4, 1))

    # ePNP
    rotation_vector = None
    translation_vector = None

    _, rotation_vector, translation_vector = cv2.solvePnP(
        objectPoints=box_vertices[:num_selected_points],
        imagePoints=user_selected_points,
        cameraMatrix=intrinsic_matrix,
        distCoeffs=distortion_coefficients,
        flags=cv2.SOLVEPNP_EPNP
    )

    print("\n\nRotation vector:\n", rotation_vector)
    print("\n\nTranslation vector:\n", translation_vector)
    
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    print("\n\nRotation matrix:\n", rotation_matrix)

    # Project the 3D points to 2D
    projected_box_2D, _ = cv2.projectPoints(box_vertices, rotation_vector, translation_vector, intrinsic_matrix, distortion_coefficients)

    # Convert the points to integer coordinates
    projected_box_2D = projected_box_2D.astype(int)

    image = cv2.imread(input_image_path)

    # Draw the projected box on the image
    for i, point in enumerate(projected_box_2D):
        color = (0, 0, 255) if i < len(user_selected_points) else (255, 0, 0)
        cv2.circle(image, tuple(point[0]), 23, color, -1)

    # Add legends
    font = cv2.FONT_HERSHEY_SIMPLEX
    legend_position = (image.shape[1] - 1000, 200)

    # Legend for selected points
    cv2.putText(image, 'Selected Points', (legend_position[0] + 30, legend_position[1]), font, 3, (0, 0, 255), 7, cv2.LINE_AA)

    # Legend for other points
    cv2.putText(image, 'Predicted Points', (legend_position[0] + 30, legend_position[1] + 100), font, 3, (255, 0, 0), 7, cv2.LINE_AA)

    cv2.namedWindow("Image with Predicted Points", cv2.WINDOW_NORMAL)
    cv2.imshow("Image with Predicted Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Draw lines connecting the vertices
    for edge in box_edges:
        start_point = tuple(projected_box_2D[edge[0]][0])
        end_point = tuple(projected_box_2D[edge[1]][0])
        cv2.line(image, start_point, end_point, (0, 0, 255), 4)

    cv2.namedWindow("Image with Predicted Points", cv2.WINDOW_NORMAL)
    cv2.imshow("Image with Predicted Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

output_image_path = "./Output/Image_with_predicted_points.jpg"
cv2.imwrite(output_image_path, image)
print("Image saved successfully at:", output_image_path)
