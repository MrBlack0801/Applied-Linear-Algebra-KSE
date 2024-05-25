import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2

### Part |
# Functions


def f1_plot_figures(figures, titles):
    fig, ax = plt.subplots()
    for figur, title in zip(figures, titles):
        ax.plot(figur[:, 0], figur[:, 1], label=title)
    ax.legend()
    plt.show()


def f2_rotate_figure(figure, angle):
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_object = np.dot(figure, rotation_matrix.transpose())
    print('Rotation Matrix:\n',
          rotation_matrix)
    return rotated_object


def f3_scale_figure(figure, coef_x, coef_y):
    scale_matrix = np.array([
        [coef_x, 0],
        [0, coef_y]])
    figure_scaled = np.dot(figure, scale_matrix)
    print('Scaling Matrix:\n',
          scale_matrix)
    return figure_scaled


def f4_reflect_figure_on_axis(figure, axis_to_reflect):
    if axis_to_reflect == 'x':
        axis_reflection_matrix = np.array([
            [1, 0],
            [0, -1]])
    elif axis_to_reflect == 'y':
        axis_reflection_matrix = np.array([
            [-1, 0],
            [0, 1]])
    else:
        print("Choose 'x' or 'y'")
        return None

    figure_reflected = figure.dot(axis_reflection_matrix)
    print('Axis Reflection Matrix:\n',
          axis_reflection_matrix)
    return figure_reflected


def f5_shear_figure(figure, axis_to_shear, angle):
    if axis_to_shear == 'x':
        shear_matrix = np.array([
            [1, np.tan(np.radians(angle))],
            [0, 1]])
    elif axis_to_shear == 'y':
        shear_matrix = np.array([
            [1, 0],

            [np.tan(np.radians(angle)), 1]])
    else:
        print("Error: Choose 'x' or 'y' axis")
        return None

    figure_sheared = np.dot(figure, shear_matrix)
    print('Shearing Matrix:\n',
          shear_matrix)
    return figure_sheared


def f6_matrix_transformation(figure, transformation_matrix_):
    figure_transformed = np.dot(figure, transformation_matrix_.transpose())
    print('Transformation Matrix:\n',
          transformation_matrix_)
    return figure_transformed


def plot_3d_figures(figures, titles):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for figur, title in zip(figures, titles):
        ax.plot(figur[:, 0], figur[:, 1], figur[:, 2], label=title)

    # ax.quiver(0, 0, 0, 0.5, 0, 0, color='r', arrow_length_ratio=0.1)
    ax.set_xlabel('X axis')
    # ax.quiver(0, 0, 0, 0, 0.5, 0, color='g', arrow_length_ratio=0.1)
    ax.set_ylabel('Y axis')
    # ax.quiver(0, 0, 0, 0, 0, 0.5, color='b', arrow_length_ratio=0.1)
    ax.set_zlabel('Z axis')
    ax.legend()
    plt.show()


def scale_figure_3d(figure, coef_x, coef_y, coef_z):
    scale_matrix = np.array([
        [coef_x, 0, 0],
        [0, coef_y, 0],
        [0, 0, coef_z]])
    figure_scaled = np.dot(figure, scale_matrix)
    print('Scaling 3d Matrix:\n',
          scale_matrix)
    return figure_scaled


def reflect_figure_on_axis_3d(figure, axis_to_reflect):
    if axis_to_reflect == 'x':
        axis_reflection_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]])
    elif axis_to_reflect == 'y':
        axis_reflection_matrix = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]])
    elif axis_to_reflect == 'z':
        axis_reflection_matrix = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]])
    else:
        print("Choose 'x', 'y', or 'z'")
        return None

    figure_reflected = figure.dot(axis_reflection_matrix)
    print('Axis Reflection Matrix:\n',
          axis_reflection_matrix)
    return figure_reflected


# Input
fig_1 = np.array([[0, 0], [1, 0], [0.5, 1], [0, 0]])
fig_2 = np.array([[-2, 0], [0, 0.5], [-0.75, 1.5], [-2, 1], [-2, 0]])
fig_3d = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1],
    [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 0]])

# Output
# f1
f1_plot_figures([fig_1, fig_2], ['Triangle', 'Figure 2'])

# f2
rotated_figure = f2_rotate_figure(fig_1, 45)
f1_plot_figures([fig_1, rotated_figure], ['Original Triangle', 'Rotated Triangle'])

# f3
scaled_figure = f3_scale_figure(fig_1, 2, 2)
f1_plot_figures([fig_1, scaled_figure], ['Original', 'Scaled'])

# f4
reflected_figure = f4_reflect_figure_on_axis(fig_1, 'x')
f1_plot_figures([fig_1, reflected_figure], ['Original', 'Reflected'])

# f5
reflected_figure = f5_shear_figure(fig_1, 'x', 15)
f1_plot_figures([fig_1, reflected_figure], ['Original', 'Sheared'])

# f6
transformation_matrix = np.array([[1, 0.5], [0, 2]])
reflected_figure = f6_matrix_transformation(fig_1, transformation_matrix)
f1_plot_figures([fig_1, reflected_figure], ['Original', 'Transformed'])

# 3d
plot_3d_figures([fig_3d], ['Figure 3d'])

scaled_3d_figure = scale_figure_3d(fig_3d, 2, 2, 2)
plot_3d_figures([fig_3d, scaled_3d_figure], ['Original', 'Scaled'])

reflected_3d_figure = reflect_figure_on_axis_3d(fig_3d, 'z')
plot_3d_figures([fig_3d, reflected_3d_figure], ['Original', 'Reflected'])

### Part ||


def plot_figure_with_opencv(image, title="Figure"):
    plt.imshow(image)
    plt.title(title)
    plt.show()


def create_image_from_points(points):
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255
    points = np.int32(points * 100 + np.array([500, 500]) // 2)
    points[:, 1] = 500 - points[:, 1]
    for i in range(len(points) - 1):
        cv2.line(image, tuple(points[i]), tuple(points[i+1]), (255, 0, 0), 2)
    return image


def rotate_image_opencv(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return rotated_image


def scale_image_opencv(image, scale_x, scale_y):
    scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y)
    return scaled_image


def reflect_image_opencv(image, axis):
    if axis == 'x':
        reflected_image = cv2.flip(image, 0)
    elif axis == 'y':
        reflected_image = cv2.flip(image, 1)
    else:
        print("Choose 'x', 'y'")
        return None
    return reflected_image


def shear_image_opencv(image, axis, angle):
    (h, w) = image.shape[:2]
    if axis == 'x':
        shear_matrix = np.array([[1, 0, 0], [np.tan(np.radians(-angle)), 1, 0]], dtype=np.float32)
    elif axis == 'y':
        shear_matrix = np.array([[1, np.tan(np.radians(-angle)), 0], [0, 1, 0]], dtype=np.float32)
    else:
        print("Choose 'x', 'y'")
        return None
    sheared_image = cv2.warpAffine(image, shear_matrix, (h, w), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return sheared_image


def matrix_transformation_opencv(image, t_matrix):
    (h, w) = image.shape[:2]
    t_matrix[0, 1] = -t_matrix[0, 1]
    t_matrix[1, 0] = -t_matrix[1, 0]
    transformed_image = cv2.warpAffine(image, t_matrix, (h, w), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return transformed_image


create_image_from_points(fig_1)

plot_figure_with_opencv(rotate_image_opencv(create_image_from_points(fig_1), 45))

plot_figure_with_opencv(scale_image_opencv(create_image_from_points(fig_1), 2, 2))

plot_figure_with_opencv(reflect_image_opencv(create_image_from_points(fig_1), 'x'))

plot_figure_with_opencv(shear_image_opencv(create_image_from_points(fig_1), 'x', 15))

transformation_matrix = np.array([[1, 0.5, 0], [0, 2, 0]])  # third column elements - shift
plot_figure_with_opencv(matrix_transformation_opencv(create_image_from_points(fig_1), transformation_matrix))

plot_figure_with_opencv(reflect_image_opencv(rotate_image_opencv(create_image_from_points(fig_1), 45), 'x'))
