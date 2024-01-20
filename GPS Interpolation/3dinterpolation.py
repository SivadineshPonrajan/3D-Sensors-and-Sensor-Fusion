import matplotlib.pyplot as plt
import numpy as np

# *****************************************************************************

#Hanif's code to load data from text file
def load_points_file(filename):

    points = []
    with open(filename, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            x, y, z = map(float, line.split())
            points.append((x, y, z)) # Changed to 3D

    return points

# *****************************************************************************

def orthogonalize(vectors):
    basis = []
    for v in vectors:
        for b in basis:
            v -= np.dot(v, b) * b
        if np.linalg.norm(v) > 1e-8:
            basis.append(v / np.linalg.norm(v))
    return np.array(basis)

# *****************************************************************************

def interpolate_trajectory(trajectory_positions, trajectory_angles, skip_factor=4):
    num_positions = len(trajectory_positions) - 1

    # Data for the new interpolated positions
    interpolated_positions = np.zeros((skip_factor * num_positions - skip_factor, 3))

    for index in range(0, num_positions - 1):

        pos_prev = trajectory_positions[index - 1]
        pos_curr = trajectory_positions[index]
        pos_next = trajectory_positions[index + 1]

        angle = trajectory_angles[index]
        half_angle = angle / 2.0

        vec1 = pos_curr - pos_prev
        vec2 = pos_next - pos_curr

        if abs((angle / 2.0)) > 1e-7 and abs(angle) < np.pi:

            # Calculate the normal vector to the plane formed by vec1 and vec2
            normal_vector = np.cross(vec1, vec2)
            normal_vector /= np.linalg.norm(normal_vector)

            # Calculate the center point of the circle for interpolation
            diff = pos_next - pos_curr
            center_diff = pos_curr + diff / 2.0
            L = np.linalg.norm(diff) / 2.0
            updated_normal = np.cross(normal_vector, diff)
            n = updated_normal / np.linalg.norm(updated_normal)
            a = L / np.tan(half_angle)
            center_point = center_diff + n * a

            # Calculate the radius as the distance between the center_point and pos_curr
            radius = np.linalg.norm(center_point - pos_curr)

            # Define the first axis of the new 2D coordinate system
            axis1 = pos_curr - center_point
            axis1 /= np.linalg.norm(axis1)

            # Define the second axis of the new 2D coordinate system
            basis = orthogonalize([axis1, updated_normal])
            axis2 = basis[1]

            # Calculate the angle between vec1 and vec2
            angle_between_vectors = -angle
            angle_range = (0, angle_between_vectors / (2 * np.pi))

            # Create the interpolated points along the arc of the circle and fill the array
            theta = np.linspace(angle_range[0] * 2 * np.pi, angle_range[1] * 2 * np.pi, skip_factor + 1)
            circle_2d = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
            circle_3d = center_point + np.dot(circle_2d, np.vstack((axis1, axis2)))
            interpolated_positions[skip_factor * index:skip_factor * index + skip_factor] = circle_3d[:-1]
        else:
            # If the angle is small, interpolate linearly
            for index2 in range(0, skip_factor):
                interpolated_positions[skip_factor * index + index2] = (
                        pos_curr * index2 + pos_next * (skip_factor - index2)) / skip_factor

    return interpolated_positions

# *****************************************************************************



# Main code

#4Hz/1HZ = 4
SKIP=4

#Load Data
data3D = np.array(load_points_file("Trajectory.xyz"))

pt_size=data3D.shape

#Number of points in the GPS data
NUM=pt_size[0]
print("Number of Points: ", NUM)

# Range object to iterate over the GPS data
rangeObject = range(0,NUM,SKIP)
print("Size of Range Object: ", len(rangeObject))

# Array to store the filtered GPS data
filteredPos=np.zeros((len(rangeObject),3))
print("Size of Filtered Position: ", filteredPos.shape)

# Store the filtered data (get every 4th point)
for counter, index in enumerate(rangeObject):
    posX = data3D[index, 0]
    posY = data3D[index, 1]
    posZ = data3D[index, 2]
    filteredPos[counter, 0] = posX
    filteredPos[counter, 1] = posY
    filteredPos[counter, 2] = posZ

# *****************************************************************************

# Calculate angles and positions 
    
num_points = len(filteredPos)

angles = np.zeros(num_points)
positions = np.zeros((num_points, 3))

for index in range(0, num_points-1):  # Updated range to avoid index -1
    # Get the previous, current, and next positions
    posPrev = filteredPos[index - 1]
    posCurr = filteredPos[index]
    posNext = filteredPos[index + 1]

    # Calculate vectors and angles
    vec1 = posCurr - posPrev
    vec2 = posNext - posCurr
    
    ortVec1 = -np.cross(vec1, vec2)

    newort = np.cross(ortVec1, vec1)
    newort = newort/np.linalg.norm(newort)

    dotProd = np.dot(vec1, vec2)
    dotProd /= (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(dotProd, -1.0, 1.0))

    # Determine the sign of the angle
    tmpSign = np.dot(np.cross(vec1,vec2), ortVec1)
    
    positions[index] = posCurr

    if tmpSign > 0:
        angles[index] = -angle
    else:
        angles[index] = angle

# 3D interpolation
newPos = interpolate_trajectory(positions, angles, SKIP)

# Save the original positions and new, interpolated ones
np.savetxt('newPos_3d.txt', newPos, fmt='%.5f')
np.savetxt('origpos_3d.txt', positions, fmt='%.5f')

# *****************************************************************************

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[1:-1, 0], positions[1:-1, 1], positions[1:-1, 2], c='red', marker='o', label='Skipped Trajectory')
ax.plot(newPos[:, 0], newPos[:, 1], newPos[:, 2], c='green', linestyle='-', marker='*', label='Interpolated Trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()