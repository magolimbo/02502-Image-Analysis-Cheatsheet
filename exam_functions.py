import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from skimage import img_as_ubyte
import glob
import os
import pathlib
from skimage.transform import SimilarityTransform
from skimage.transform import warp

#show two images side by side
def show_comparison(original, transformed, transformed_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(transformed)
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
# L'ordine è importante
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



def hough_to_xy(x,y):
    # Mapping
    # Given xy-coordinates

    # Convert x and y to Hough space parameters (rho and theta)
    rho = np.sqrt(x * 2 + y * 2)  # Calculate rho using the distance formula
    theta = np.arctan2(y, x)  # Calculate theta using the arctan2 function

    # Convert theta from radians to degrees for display (optional)
    theta_degrees = theta * (180 / np.pi)

    # Display the Hough space parameters
    print("Hough space parameters:")
    print(f"Rho: {rho:.2f}")
    print(f"Theta (degrees): {theta_degrees:.2f}")
    
def xy_to_hough(rho, theta_degrees,x_values=[]):
    # Mapping from Hough Space to Cartesian Space
    # Given Hough space parameters

    # Convert theta from degrees to radians
    theta_rad = theta_degrees * (np.pi / 180)


    # Function to calculate y for a given x
    def calculate_y(x):
        return (rho - x * np.cos(theta_rad)) / np.sin(theta_rad)

    # Calculate y for a range of x values
    x_values = [7, 9, 6, 6, 3]  # Choose different x values
    corresponding_y_values = [calculate_y(x) for x in x_values]

    # Display the approximate data points
    print("Approximate data points in the xy-plane:")
    for i, x in enumerate(x_values):
        y = corresponding_y_values[i]
        print(f"({x}, {y:.2f})")