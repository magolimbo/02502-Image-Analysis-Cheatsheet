import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from skimage import img_as_ubyte
import glob
import os
import pathlib
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import matrix_transform
import pandas as pd
from scipy.stats import norm
from skimage import color
from skimage.util import img_as_float, img_as_uint
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



#show two images side by side
def show_comparison(original, transformed, transformed_name = "Transformed Image", cmap = "gray"):
    _, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(transformed, cmap = cmap)
    ax2.set_title(transformed_name)
    ax2.axis('off')
    io.show()

def create_u_byte_image_from_vector(im_vec, height, width, channels):
    min_val = im_vec.min()
    max_val = im_vec.max()

    # Transform to [0, 1]
    im_vec = np.subtract(im_vec, min_val)
    im_vec = np.divide(im_vec, max_val - min_val)
    im_vec = im_vec.reshape(height, width, channels)
    im_out = img_as_ubyte(im_vec)
    return im_out

def read_landmark_file(file_name):
    f = open(file_name, 'r')
    lm_s = f.readline().strip().split(' ')
    n_lms = int(lm_s[0])
    if n_lms < 3:
        print(f"Not enough landmarks found")
        return None

    new_lms = 3
    # 3 landmarks each with (x,y)
    lm = np.zeros((new_lms, 2))
    for i in range(new_lms):
        lm[i, 0] = lm_s[1 + i * 2]
        lm[i, 1] = lm_s[2 + i * 2]
    return lm

def blob_circularity(area, perimeter):
    """
    You may get values larger than 1 because
    we are in a "discrete" (pixels) domain. Check:

    CIRCULARITY OF OBJECTS IN IMAGES, Botterma, M.J. (2000)
    https://core.ac.uk/download/pdf/14946814.pdf
    """
    f_circ = (4 * np.pi * area) / (perimeter**2)
    return f_circ



def align_and_crop_one_cat_to_destination_cat(img_src, lm_src, img_dst, lm_dst):
    """
    Landmark based alignment of one cat image to a destination
    :param img_src: Image of source cat
    :param lm_src: Landmarks for source cat
    :param lm_dst: Landmarks for destination cat
    :return: Warped and cropped source image. None if something did not work
    """
    tform = SimilarityTransform()
    tform.estimate(lm_src, lm_dst)
    warped = warp(img_src, tform.inverse, output_shape=img_dst.shape)

    # Center of crop region
    cy = 185
    cx = 210
    # half the size of the crop box
    sz = 180
    warp_crop = warped[cy - sz:cy + sz, cx - sz:cx + sz]
    shape = warp_crop.shape
    if shape[0] == sz * 2 and shape[1] == sz * 2:
        return img_as_ubyte(warp_crop)
    else:
        print(f"Could not crop image. It has shape {shape}. Probably to close to border of image")
        return None
    
def preprocess_all_cats(in_dir, out_dir):
    """
    Create aligned and cropped version of image
    :param in_dir: Where are the original photos and landmark files
    :param out_dir: Where should the preprocessed files be placed
    """
    dst = "data/ModelCat"
    dst_lm = read_landmark_file(f"{dst}.jpg.cat")
    dst_img = io.imread(f"{dst}.jpg")

    all_images = glob.glob(in_dir + "*.jpg")
    for img_idx in all_images:
        name_no_ext = os.path.splitext(img_idx)[0]
        base_name = os.path.basename(name_no_ext)
        out_name = f"{out_dir}/{base_name}_preprocessed.jpg"

        src_lm = read_landmark_file(f"{name_no_ext}.jpg.cat")
        src_img = io.imread(f"{name_no_ext}.jpg")

        proc_img = align_and_crop_one_cat_to_destination_cat(src_img, src_lm, dst_img, dst_lm)
        if proc_img is not None:
            io.imsave(out_name, proc_img)
            
def preprocess_one_cat():
    src = "data/MissingCat"
    dst = "data/ModelCat"
    out = "data/MissingCatProcessed.jpg"

    src_lm = read_landmark_file(f"{src}.jpg.cat")
    dst_lm = read_landmark_file(f"{dst}.jpg.cat")

    src_img = io.imread(f"{src}.jpg")
    dst_img = io.imread(f"{dst}.jpg")

    src_proc = align_and_crop_one_cat_to_destination_cat(src_img, src_lm, dst_img, dst_lm)
    if src_proc is None:
        return

    io.imsave(out, src_proc)

    fig, ax = plt.subplots(ncols=3, figsize=(16, 6))
    ax[0].imshow(src_img)
    ax[0].plot(src_lm[:, 0], src_lm[:, 1], '.r', markersize=12)
    ax[1].imshow(dst_img)
    ax[1].plot(dst_lm[:, 0], dst_lm[:, 1], '.r', markersize=12)
    ax[2].imshow(src_proc)
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()
    


def create_affine_matrix(transformations):
    """
    Crea una matrice di trasformazione affine combinando le trasformazioni specificate.

    Parametri:
    transformations (list): Lista di tuple contenenti le operazioni di trasformazione e i relativi valori.
                            Le operazioni supportate sono "rotation", "translation", "scaling" e "shearing".
                            La lista deve essere nel formato:
                            [
                                ('rotation', (valore_x, valore_y, valore_z)),    # Valori in gradi per x, y, z
                                ('translation', (valore_x, valore_y, valore_z)),
                                ('scaling', (valore_x, valore_y, valore_z)),
                                ('shearing', (valore_x1, valore_y1, valore_x2, valore_y2, valore_x3, valore_y3))
                                # Valori per la shearing sugli assi x e y
                            ]

    Returns:
    np.ndarray: Matrice di trasformazione affine 4x4
    """
    matrix = np.eye(4)

    for operation, values in transformations:
        if operation == "rotation":
            # Convertire i gradi in radianti
            pitch, roll, yaw = (
                np.radians(values[0]),
                np.radians(values[1]),
                np.radians(values[2]),
            )

            # Matrici di rotazione per ciascun asse
            Rx = np.array(
                [
                    [1, 0, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch), 0],
                    [0, np.sin(pitch), np.cos(pitch), 0],
                    [0, 0, 0, 1],
                ]
            )

            Ry = np.array(
                [
                    [np.cos(roll), 0, np.sin(roll), 0],
                    [0, 1, 0, 0],
                    [-np.sin(roll), 0, np.cos(roll), 0],
                    [0, 0, 0, 1],
                ]
            )

            Rz = np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), 0, 0],
                    [np.sin(yaw), np.cos(yaw), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )

            # Moltiplicazione delle matrici di rotazione per ciascun asse
            rotation_matrix = np.dot(np.dot(Rx, Ry), Rz)
            matrix = np.dot(rotation_matrix, matrix)

        elif operation == "translation":
            translation_matrix = np.array(
                [
                    [1, 0, 0, values[0]],
                    [0, 1, 0, values[1]],
                    [0, 0, 1, values[2]],
                    [0, 0, 0, 1],
                ]
            )
            matrix = np.dot(translation_matrix, matrix)

        elif operation == "scaling":
            scaling_matrix = np.array(
                [
                    [values[0], 0, 0, 0],
                    [0, values[1], 0, 0],
                    [0, 0, values[2], 0],
                    [0, 0, 0, 1],
                ]
            )
            matrix = np.dot(scaling_matrix, matrix)

        elif operation == "shearing":
            shear_matrix = np.array(
                [
                    [1, values[0], values[1], 0],
                    [values[2], 1, values[3], 0],
                    [values[4], values[5], 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            matrix = np.dot(shear_matrix, matrix)

    return matrix


# # Esempio di trasformazioni
# # L'ordine è importante
# transformations = [
#     ('rotation', (45, -30, 10)),  # Rotazione (x=pitch, y=roll, z=yaw)
#     ('translation', (10, 5, 3)),  # Traslazione
#     ('scaling', (1.5, 2, 1)),     # Scalatura
#     ('shearing', (0.2, 0.1, 0.3, 0.1, 0.4, 0.2))  # Shearing
# ]

# # Creazione della matrice di trasformazione affine
# affine_matrix = create_affine_matrix(transformations)
# # Stampare la matrice risultante
# print(affine_matrix)
        
        

def convert_xy_to_hough(x, y):
    """
    Convert xy-coordinates to Hough space parameters (rho and theta).

    Parameters:
    x (float): x-coordinate in Cartesian space.
    y (float): y-coordinate in Cartesian space.

    Returns:
    None: Prints the calculated Hough space parameters.

    This function maps xy-coordinates (x and y) to Hough space parameters (rho and theta).
    It calculates rho (the distance from the origin to the closest point on the line) using the distance formula,
    and theta (the angle between the x-axis and the normal line from the origin to the line) using the arctan2 function.
    Then, it displays the calculated Hough space parameters.
    """
    # Convert x and y to Hough space parameters (rho and theta)
    rho = np.sqrt(x**2 + y**2)  # Calculate rho using the distance formula
    theta = np.arctan2(y, x)  # Calculate theta using the arctan2 function

    # Convert theta from radians to degrees for display (optional)
    theta_degrees = theta * (180 / np.pi)

    # Display the Hough space parameters
    print("Hough space parameters:")
    print(f"Rho: {rho:.2f}")
    print(f"Theta (degrees): {theta_degrees:.2f}")
    return rho, theta_degrees
    
def convert_hough_to_xy(rho, theta_degrees, x_values=[]):
    """
    Convert Hough space parameters (rho and theta) to Cartesian space (x and y).

    Parameters:
    rho (float): The distance from the origin to the closest point on the line.
    theta_degrees (float): The angle (in degrees) between the x-axis and the normal line from the origin to the line.
    x_values (list): List of x values for which corresponding y values will be calculated. Default is an empty list.

    Returns:
    None: Prints the approximate data points in the xy-plane.
    
    """
    # Convert theta from degrees to radians
    theta_rad = theta_degrees * (np.pi / 180)

    # Function to calculate y for a given x
    def calculate_y(x):
        return (rho - x * np.cos(theta_rad)) / np.sin(theta_rad)

    # Calculate y for a range of x values
    corresponding_y_values = [calculate_y(x) for x in x_values]

    # Display the approximate data points
    print("Approximate data points in the xy-plane:")
    for i, x in enumerate(x_values):
        y = corresponding_y_values[i]
        print(f"({x}, {y:.2f})")
 
       
def haar_features(grey_box=[], white_box=[]):
    """
    Calculate Haar-like features based on the sums of pixel values in specified boxes.

    Parameters:
    grey_box (list): List of pixel values in the grey box region. Default is an empty list.
    white_box (list): List of pixel values in the white box region. Default is an empty list.

    Returns:
    None: Prints the calculated Haar feature value.
    
    """
    grey_sum = sum(grey_box)
    white_sum = sum(white_box)
    
    print(f"HAAR FEATURE = {grey_sum - white_sum}\n")


def integral_image(integral_value=[]):
    """
    Calculate the integral image value by summing up the given integral values.

    Parameters:
    integral_value (list): List of integral values for computation. Default is an empty list.

    Returns:
    None: Prints the calculated integral image value.
    
    """
    print(f"INTEGRAL IMAGE = {sum(integral_value)}")


def linear_gray_scale_transformation(image, min_val, max_val):
    """
    Performs a linear grayscale transformation on the input image
    so that the transformed image has a minimum pixel value of 0.1
    and a maximum pixel value of 0.6.

    Parameters:
    image (numpy.ndarray): Input grayscale image.

    Returns:
    numpy.ndarray: Transformed grayscale image.
    """
    current_min = np.min(image)
    current_max = np.max(image)

    # Perform linear transformation to the desired range [min_val, max_val]
    transformed_image = min_val + (image - current_min) * ((max_val - min_val) / (current_max - current_min))
    transformed_image = np.clip(transformed_image, min_val, max_val)  # Clip values to desired range
    
    return transformed_image


def var_explained(S, plot=True, show_df=True):
    """
    S = list of variances of components (can be read from the S/Sigma matrix)
    plot = to plot or not
    show_df = to show df with varaince explained or not
    """
    S = np.array(S)

    # df with variance explained
    df_var_exp = pd.DataFrame(columns=["k", "var_explained"])
    for i in range(len(S)):
        t = np.sum(S[0 : i + 1] ** 2) / np.sum(S ** 2)
        df_var_exp.loc[i] = [i + 1, t]
    if plot:
        # plot of variance explained
        plt.plot(df_var_exp["k"], df_var_exp["var_explained"])
        plt.scatter(df_var_exp["k"], df_var_exp["var_explained"])
        plt.title("Variance explained")
        plt.xlim(np.min(df_var_exp["k"]), np.max(df_var_exp["k"]))
        plt.ylim(0, 1)
        plt.show()

    if show_df:
        print(df_var_exp)
    return df_var_exp


def similarity_transformation(moving_img, fixed_img, src, dst):
    don1 = moving_img
    don2 = fixed_img

    e_x = src[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Landmark alignment error F (sum of squared differences): {f}")


    tform = SimilarityTransform()
    tform.estimate(src, dst)
    tform.scale_params = True
    src_transform = matrix_transform(src, tform.params)

    fig, ax = plt.subplots()
    io.imshow(don1)
    ax.plot(
        src_transform[:, 0],
        src_transform[:, 1],
        "-r",
        markersize=12,
        label="Source transform",
    )
    ax.plot(dst[:, 0], dst[:, 1], "-g", markersize=12, label="Destination")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Landmarks after alignment")
    plt.show()

    warped = warp(don2, tform.inverse)
    show_comparison(don1, warped, "Landmark based transformation")
    warped = img_as_ubyte(warped)
    return warped

import numpy as np
from scipy.stats import norm


def parametric_distance_classifier(data):
    """
    Computes thresholds between adjacent classes using parametric classifier.

    Args:
    - data: List of class data containing numerical values.

    Returns:
    - List of tuples with thresholds between adjacent classes sorted by threshold values.
    """

    # Ordina i dati delle classi
    sorted_data = sorted(data, key=lambda x: np.mean(x))

    # Calcola means e stds per ogni classe
    means = [np.mean(el) for el in sorted_data]
    stds = [np.std(el) for el in sorted_data]

    thresholds = []

    # Calcola le soglie tra classi adiacenti
    for i in range(len(means) - 1):
        mu_low = means[i]
        std_low = stds[i]
        mu_high = means[i + 1]
        std_high = stds[i + 1]
        thres_low_high = None

        # Calcola la soglia tra le due distribuzioni normali
        for test_value in np.linspace(mu_low, mu_high, 1000):
            if norm.pdf(test_value, mu_high, std_high) > norm.pdf(test_value, mu_low, std_low):
                thres_low_high = round(test_value, 3)
                break

        # Salva la soglia tra classi adiacenti
        if thres_low_high is not None:
            thresholds.append((f"class_{i + 1} and class_{i + 2}", thres_low_high))

    # Ritorna le soglie tra classi ordinate
    return sorted(thresholds, key=lambda x: x[1])



# # Esempio di utilizzo con un numero variabile di classi
# Grass = [68, 65, 67]
# Road = [70, 80, 75]
# Sky = [77, 92, 89]
# lista = [Grass, Road, Sky]

# result = parametric_distance_classifier(lista)
# print("\nThresholds between adjacent classes:")
# for threshold in result:
#     print(f"{threshold[0]}: {threshold[1]}")


def minimum_distance_classifier(data):
    """
    Computes midpoints between means of consecutive elements.

    Args:
    - data: List of numerical elements.

    Returns:
    - Dictionary containing midpoints between consecutive means.
    """
    means = []

    # Calculate means for each element in the list
    for i, el in enumerate(data):
        mu = round(np.mean(el), 3)
        means.append(mu)
        print(f"{el} - mean = {means[i]}")

    grey_values = {}
    sorted_indices = sorted(range(len(means)), key=lambda x: means[x])

    # Calculate midpoints between consecutive means
    for i in range(len(sorted_indices) - 1):
        idx1 = sorted_indices[i]
        idx2 = sorted_indices[i + 1]
        midpoint = round((means[idx1] + means[idx2]) / 2, 3)
        grey_values[f"({means[idx1]}, {means[idx2]})"] = midpoint

    # Print and return dictionary of midpoints
    print(f"\n{grey_values}")
    return grey_values

# Grass = [68, 65, 67]
# Road = [70, 80, 75]
# Sky = [77, 92, 89]
# lista = [Grass, Road, Sky]

# ciao = minimum_distance_classifier(lista)

#stritching
def histogram_stretch(img_in, min, max):
    """
    Stretches the histogram of an image
    :param img_in: Input image
    :return: Image, where the histogram is stretched so the min values is 0 and the maximum value 255
    """
    # img_as_float will divide all pixel values with 255.0
    img_float = img_as_float(img_in)
    min_val = img_float.min()
    max_val = img_float.max()
    min_desired = min
    max_desired = max

    # Do something here
    img_out = (
        (img_float - min_val) * (max_desired - min_desired) / (max_val - min_val)
    ) + min_desired
    # img_as_ubyte will multiply all pixel values with 255.0 before converting to unsigned byte
    return img_out



def lda(class_0_data, class_1_data, new_vector):
    """
    Performs Linear Discriminant Analysis (LDA) and predicts probabilities for a new vector.

    Args:
    - class_0_data: Data for class 0.
    - class_1_data: Data for class 1.
    - new_vector: New observation to predict probabilities for.

    Returns:
    - Predicted probabilities for each class for the new observation.
    """
    # Combine the class data and labels
    class_0_labels = [0] * len(class_0_data)
    class_1_labels = [1] * len(class_1_data)
    positions = class_0_data + class_1_data
    labels = class_0_labels + class_1_labels

    # Create and train the LDA classifier
    lda = LinearDiscriminantAnalysis()
    lda.fit(positions, labels)

    # Predict the probabilities for each class for the new observation
    predicted_probabilities = lda.predict_proba([new_vector])

    return predicted_probabilities[0]



# # Esempio di utilizzo della funzione lda()
# # Dati delle classi
# class_0_positions = [
#     [1.00, 1.00],
#     [2.20, -3.00],
#     [3.50, -1.40],
#     [3.70, -2.70],
#     [5.00, 0],
# ]

# class_1_positions = [
#     [0.10, 0.70],
#     [0.22, -2.10],
#     [0.35, -0.98],
#     [0.37, -1.89],
#     [0.50, 0],
# ]

# # Nuovo vettore
# new_vector = [1.00, 1.00]

# # Calcola le probabilità che il nuovo vettore appartenga ad entrambe le classi
# probabilities = lda(class_0_positions, class_1_positions, new_vector)
# print(f"Probability for class 0 (Mushroom Type A): {probabilities[0]}")
# print(f"Probability for class 1 (Mushroom Type B): {probabilities[1]}")



def rgb2hsi(r, g, b):
    
    import math
    
    h = 0
    if g >= b:
        h = math.acos(
            0.5 * (r - g + r - b) / math.sqrt((r - g) * (r - g) + (r - b) * (g - b))
        )
    else:
        h = 360 - math.acos(
            0.5 * (r - g + r - b) / math.sqrt((r - g) * (r - g) + (r - b) * (g - b))
        )

    s = 1 - 3 * (min(r, g, b) / (r + g + b))
    i = (r + g + b) / 3

    return np.array([h, s, i])


#Normalized Normalised Cross Correlation
# import numpy as np

# matrix = np.array([[24,12,42,155, 28],
#  [169,92,102,60,104],
#  [143 , 81,  94 ,140,196],
#  [ 68, 157 ,134 ,135 , 73],
#  [194, 163, 146 , 37, 121]])

# image_patch = matrix[1:4, 1:4]

# template = np.array([[57,129,245],
#             [192,178,140],
#             [65,227,35]])

# scalar_prod = image_patch * template
# correlation = np.sum(scalar_prod)

# # Utilizzo della list comprehension per ottenere la matrice con ogni punto al quadrato
# squared_matrix_image_patch = [[element ** 2 for element in row] for row in image_patch]
# squared_matrix_template = [[element ** 2 for element in row] for row in template]


# length_of_image_patch = np.sqrt(np.sum(squared_matrix_image_patch))
# length_of_template = np.sqrt(np.sum(squared_matrix_template))

# NCC = correlation /(length_of_image_patch * length_of_template)



# count instances true values
# # Conta le istanze di valori unici
# valori_unici, conteggi = np.unique(array, return_counts=True)

# # Stampa i risultati
# for valore, conteggio in zip(valori_unici, conteggi):
#     print(f"Valore: {valore}, Numero di istanze: {conteggio}")