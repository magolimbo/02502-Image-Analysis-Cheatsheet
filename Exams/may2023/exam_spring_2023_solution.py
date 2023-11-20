from skimage import color, io
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from skimage.morphology import erosion, dilation, binary_erosion, binary_dilation
from skimage.morphology import disk
from skimage.morphology import square
from skimage.filters import prewitt
from skimage.filters import median
from skimage import segmentation
from skimage import measure
import math
from scipy.stats import norm
import pandas as pd
import seaborn as sns
from skimage.transform import rescale, resize
from skimage import color, data, io, morphology, measure, segmentation, img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from skimage.color import label2rgb
from scipy.spatial import distance
from skimage.transform import rotate
from skimage.transform import EuclideanTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import swirl
from skimage.transform import matrix_transform
import glob
from sklearn.decomposition import PCA
import random

# https://www.kaggle.com/datasets/uciml/glass?resource=download
def pca_on_glass_data_F2023():
    in_dir = "data/GlassPCA/"
    txt_name = "glass_data.txt"

    glass_data = np.loadtxt(in_dir + txt_name, comments="%")
    x = glass_data
    n_feat = x.shape[1]
    n_obs = x.shape[0]
    print(f"Number of features: {n_feat} and number of observations: {n_obs}")

    mn = np.mean(x, axis=0)
    data = x - mn

    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    spread = maxs - mins
    data = data / spread

    print(f"Answer: Amount of Sodium {data[0][1]:.2f}")
    c_x = np.cov(data.T)

    print(f"Answer: Covariance matrix at (0, 0): {c_x[0][0]:.3f}")

    values, vectors = np.linalg.eig(c_x)

    v_norm = values / values.sum() * 100
    plt.plot(v_norm)
    plt.xlabel('Principal component')
    plt.ylabel('Percent explained variance')
    plt.ylim([0, 100])
    plt.show()

    answer = v_norm[0] + v_norm[1] + v_norm[2]
    print(f"Answer: Variance explained by the first three PC: {answer:.2f}")

    # Project data
    pc_proj = vectors.T.dot(data.T)

    abs_pc_proj = np.abs(pc_proj)
    max_proj_val = np.max(abs_pc_proj)
    print(f"Answer: maximum absolute projected answer {max_proj_val}")

def change_detection_F2023():
    name_1 = 'data/ChangeDetection/background.png'
    name_2 = 'data/ChangeDetection/new_frame.png'

    im_1 = io.imread(name_1)
    im_2 = io.imread(name_2)
    im_1_g = color.rgb2gray(im_1)
    im_2_g = color.rgb2gray(im_2)

    average_val_org = np.average(im_1_g[150:200, 150:200])
    print(f"Average_value {average_val_org:.2f}")

    alpha = 0.90
    im_new_background = alpha * im_1_g + (1 - alpha) * im_2_g
    average_val = np.average(im_new_background[150:200, 150:200])
    print(f"Answer: average_value {average_val:.2f}")

    dif_thres = 0.1
    dif_img = np.abs(im_new_background - im_2_g)
    dif_bin = (dif_img > dif_thres)
    io.imshow(dif_bin)
    io.show()
    changed_pixels = np.sum(dif_bin)
    print(f"Answer: Changed pixels {changed_pixels:.0f}")


def system_frame_rate_F2023():
    # bytes per second
    transfer_speed = 24000000
    image_mb = 1600 * 800 * 3
    images_per_second = transfer_speed / image_mb
    print(f"Images transfered per second {images_per_second:.3f}")

    proc_time = 0.230
    proc_per_second = 1/proc_time
    print(f"Images processed per second {proc_per_second:.1f}")

    max_fps = min(proc_per_second, images_per_second)
    print(f"System framerate {max_fps:.1f}")

    img_per_sec = 6.25
    transfer_speed = img_per_sec * image_mb
    print(f"Computed transfer speed {transfer_speed}")

def nike_rgb_hsv_thresholds_F2023():
    in_dir = "data/Pixelwise/"
    im_name = "nike.png"
    im_org = io.imread(in_dir + im_name)
    hsv_img = color.rgb2hsv(im_org)
    hue_img = hsv_img[:, :, 0]

    io.imshow(hue_img)
    plt.title('Hue image')
    io.show()

    letters = (0.3 < hue_img) & (hue_img < 0.7)
    io.imshow(letters)
    plt.title('Segmented Letters')
    io.show()

    footprint = disk(8)
    dilated = dilation(letters, footprint)

    result = dilated.sum()
    print(f"Answer: Result {result}")

    io.imshow(dilated)
    plt.title('Dilated letters')
    io.show()


def filtering_F2023():
    in_dir = "data/Letters/"
    im_name = "Letters.png"
    im_org = io.imread(in_dir + im_name)
    io.imshow(im_org)
    plt.title('Letters')
    io.show()

    img_g = color.rgb2gray(im_org)

    size = 8
    footprint = np.ones([size, size])
    # med_img = median(img_g, footprint)
    med_img = median(img_g, square(size))
    io.imshow(med_img)
    plt.title('Filtered letters')
    io.show()
    print(f"Answer: value at (100,100): {med_img[100, 100]:.2f}")

    # edge_img = prewitt(img_as_ubyte(im_org))
    # min_val = edge_img.min()
    # max_val = edge_img.max()
    # io.imshow(edge_img, vmin=min_val, vmax=max_val, cmap="terrain")
    # plt.title('Prewitt filtered image')
    # io.show()
    #
    # bin_edges = edge_img > 0.06
    # io.imshow(bin_edges)
    # plt.title('Binary edges. Manual threshold')
    # io.show()
    #
    # num_pixels = bin_edges.sum()
    # print(f"Number of edge pixels {num_pixels}")


def letters_blob_analysis_F2023():
    in_dir = "data/Letters/"
    im_name = "Letters.png"
    im_org = io.imread(in_dir + im_name)
    io.imshow(im_org)
    plt.title('Letters')
    io.show()

    img_r = im_org[:, :, 0]
    img_g = im_org[:, :, 1]
    img_b = im_org[:, :, 2]

    img_letters = (img_r > 100) & (img_g < 100) & (img_b < 100)

    footprint = disk(3)
    eroded = erosion(img_letters, footprint)

    result = eroded.sum()
    print(f"Answer: eroded {result}")

    # img_g = color.rgb2gray(im_org)

    # size = 8
    # footprint = np.ones([size, size])
    # med_img = median(img_g, footprint)
    io.imshow(eroded)
    plt.title('Filtered letters')
    io.show()
    # print(f"Answer: value at (100,100): {med_img[100, 100]:.2f}")

    label_img = measure.label(eroded)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")

    region_props = measure.regionprops(label_img)

    # areas = np.array([prop.area for prop in region_props])

    min_area = 1000
    max_area = 4000
    min_perm = 300

    # Create a copy of the label_img
    label_img_filter = label_img.copy()
    for region in region_props:
        a = region.area
        p = region.perimeter

        if p < min_perm or a < min_area or a > max_area:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0

    # Create binary image from the filtered label image
    i_letter = label_img_filter > 0
    # show_comparison(img, i_area, 'Found spleen based on area')
    io.imshow(i_letter)
    io.show()


def kidney_pixel_analysis_F2023():
    in_dir = "data/abdominal/"
    im_name = "1-166.dcm"

    ct = dicom.read_file(in_dir + im_name)
    img = ct.pixel_array

    kidney_l_roi = io.imread(in_dir + 'kidneyROI_l.png')
    kidney_l_mask = kidney_l_roi > 0
    kidney_l_values = img[kidney_l_mask]
    (mu_kidney_l, std_kidney_l) = norm.fit(kidney_l_values)
    print(f"Answer: kidney_l: Average {mu_kidney_l:.0f} standard deviation {std_kidney_l:.0f}")

    kidney_r_roi = io.imread(in_dir + 'kidneyROI_r.png')
    kidney_r_mask = kidney_r_roi > 0
    kidney_r_values = img[kidney_r_mask]
    (mu_kidney_r, std_kidney_r) = norm.fit(kidney_r_values)
    print(f"Answer: kidney_r: Average {mu_kidney_r:.0f} standard deviation {std_kidney_r:.0f}")


    #
    # aorta_roi = io.imread(in_dir + 'AortaROI.png')
    # aorta_mask = aorta_roi > 0
    # aorta_values = img[aorta_mask]
    # (mu_aorta, std_aorta) = norm.fit(aorta_values)
    # print(f"Aorta: Average {mu_aorta:.0f} standard deviation {std_aorta:.0f}")
    #
    liver_roi = io.imread(in_dir + 'LiverROI.png')
    liver_mask = liver_roi > 0
    liver_values = img[liver_mask]
    (mu_liver, std_liver) = norm.fit(liver_values)
    # print(f"Answer: liver: Average {mu_liver:.0f} standard deviation {std_liver:.0f}")
    # min_hu = mu_liver - np.sqrt(3) * std_liver
    # max_hu = mu_liver + np.sqrt(3) * std_liver
    min_hu = mu_liver - std_liver
    max_hu = mu_liver + std_liver
    print(f"Answer: HU limits : {min_hu:0.2f} {max_hu:0.2f}")

    bin_img = (img > min_hu) & (img < max_hu)
    liver_label_colour = color.label2rgb(bin_img)
    io.imshow(liver_label_colour)
    plt.title("First Liver estimate")
    io.show()

    footprint = disk(3)
    dilated = dilation(bin_img, footprint)

    footprint = disk(10)
    eroded = erosion(dilated, footprint)
    io.imshow(eroded)
    plt.title("Second Liver estimate")
    io.show()

    footprint = disk(10)
    dilated = dilation(eroded, footprint)
    io.imshow(dilated)
    plt.title("Third Liver estimate")
    io.show()

    label_img = measure.label(dilated)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")

    region_props = measure.regionprops(label_img)

    min_area = 1500
    max_area = 7000
    min_perm = 300

    # Create a copy of the label_img
    label_img_filter = label_img.copy()
    for region in region_props:
        a = region.area
        p = region.perimeter

        if p < min_perm or a < min_area or a > max_area:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0

    # Create binary image from the filtered label image
    i_liver = label_img_filter > 0
    # show_comparison(img, i_area, 'Found spleen based on area')
    io.imshow(i_liver)
    io.show()

    gt_bin = liver_roi > 0
    dice_score = 1 - distance.dice(i_liver.ravel(), gt_bin.ravel())
    print(f"Answer: DICE score {dice_score:.3f}")


    #
    # min_hu = 147
    # max_hu = 155
    # hu_range = np.arange(min_hu, max_hu, 1.0)
    # pdf_aorta = norm.pdf(hu_range, mu_aorta, std_aorta)
    # pdf_liver = norm.pdf(hu_range, mu_liver, std_liver)
    # plt.plot(hu_range, pdf_aorta, 'r--', label="aorta")
    # plt.plot(hu_range, pdf_liver, 'g', label="liver")
    # plt.title("Fitted Gaussians")
    # plt.legend()
    # plt.show()
    # # Answer = 151


# https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.rotate
def otsu_rotate_image_F2023():
    in_dir = "data/GeomTrans/"
    im_name = "lights.png"
    im_org = io.imread(in_dir + im_name)

    # angle in degrees - counter clockwise
    rotation_angle = 11
    rot_center = [40, 40]
    rotated_img = rotate(im_org, rotation_angle, center=rot_center)
    # rot_byte = img_as_ubyte(rotated_img
    # print(f"Value at (200, 200) : {rot_byte[200, 200]}")
    # io.imshow(rot_byte)
    # io.show()

    img_gray = color.rgb2gray(rotated_img)

    auto_tresh = threshold_otsu(img_gray)
    print(f"Answer: Otsus treshold: {auto_tresh:.2f}")

    # auto_tresh = threshold_otsu(img_out)
    # print(f"Otsus treshold {auto_tresh:.2f}")

    img_thres = img_gray > auto_tresh
    io.imshow(img_thres)
    io.show()
    for_percent = img_thres.sum() / img_thres.size * 100

    print(f"Answer: foreground pixels percent: {for_percent:.0f}")


def landmark_based_registration_F2023():
    in_dir = "data/LMRegistration/"
    src_img = io.imread(in_dir + 'shoe_1.png')
    dst_img = io.imread(in_dir + 'shoe_2.png')

    # src = np.array([[55, 220], [675, 105], [675, 315]])
    # dst = np.array([[165, 100], [605, 200], [525, 379]])

    # src = np.array([[320, 40], [120, 425], [330, 740]])
    # dst = np.array([[320, 80], [155, 380], [300, 670]])

    src = np.array([[40, 320], [425, 120], [740, 330]])
    dst = np.array([[80, 320], [380, 155], [670, 300]])

    e_x = src[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Landmark alignment error F: {f}")

    plt.imshow(src_img)
    plt.plot(src[:, 0], src[:, 1], '.r', markersize=12)
    plt.title("Source image")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(src[:, 0], src[:, 1], '*r', markersize=12, label="Source")
    ax.plot(dst[:, 0], dst[:, 1], '*g', markersize=12, label="Destination")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Landmarks before alignment")
    plt.show()

    # plt.scatter(src[:, 0], src[:, 1])
    # plt.scatter(trg[:, 0], trg[:, 1])
    # plt.show()
    # tform = EuclideanTransform()
    tform = SimilarityTransform()
    tform.estimate(src, dst)
    print(f"Answer: scale {tform.scale:.2f}")

    src_transform = matrix_transform(src, tform.params)
    # print(src_transform)

    e_x = src_transform[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src_transform[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f_after = error_x + error_y
    print(f"Aligned landmark alignment error F: {f_after}")
    print(f"Answer: alignment error change: {f - f_after:.0f}")

    fig, ax = plt.subplots()
    ax.plot(src[:, 0], src[:, 1], '*r', markersize=12, label="Source")
    ax.plot(src_transform[:, 0], src_transform[:, 1], '*b', markersize=12, label="Source transformed")
    ax.plot(dst[:, 0], dst[:, 1], '*g', markersize=12, label="Destination")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Landmarks after alignment")
    plt.show()

    warped = warp(src_img, tform.inverse)

    val_1 = img_as_ubyte(warped)[200, 200]
    val_2 = img_as_ubyte(dst_img)[200, 200]
    print(f"Value at (200, 200) : {val_1}")
    print(f"Value at (200, 200) : {val_2}")

    val_1_b = val_1[2]
    val_2_b = val_2[2]
    print(f"Answer: b difference {np.abs(val_2_b - val_1_b)}")

    fig, ax = plt.subplots(ncols=3, figsize=(16, 6))
    ax[0].imshow(src_img)
    ax[0].plot(src[:, 0], src[:, 1], '.r', markersize=12)
    # ax[1].plot(dst[:, 0], dst[:, 1], '.r', markersize=12)
    ax[1].imshow(warped)
    ax[1].plot(src_transform[:, 0], src_transform[:, 1], '.r', markersize=12)
    ax[2].imshow(dst_img)
    ax[2].plot(dst[:, 0], dst[:, 1], '.r', markersize=12)
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()

def create_u_byte_image_from_vector(im_vec, height, width, channels):
    min_val = im_vec.min()
    max_val = im_vec.max()

    # Transform to [0, 1]
    im_vec = np.subtract(im_vec, min_val)
    im_vec = np.divide(im_vec, max_val - min_val)
    im_vec = im_vec.reshape(height, width, channels)
    im_out = img_as_ubyte(im_vec)
    return im_out

def do_pca_on_all_images_in_directory_F2023():
    # Find all image files in data dir
    in_dir = "data/PizzaPCA/training/"
    all_images = glob.glob(in_dir + "*.png")
    n_samples = len(all_images)

    # Exercise 2
    # Read first image to get image dimensions
    im_org = io.imread(all_images[0])
    im_shape = im_org.shape
    height = im_shape[0]
    width = im_shape[1]
    channels = im_shape[2]
    n_features = height * width * channels

    print(f"Found {n_samples} image files. Height {height} Width {width} Channels {channels} n_features {n_features}")

    data_matrix = np.zeros((n_samples, n_features))

    idx = 0
    for image_file in all_images:
        img = io.imread(image_file)
        flat_img = img.flatten()
        data_matrix[idx, :] = flat_img
        idx += 1

    # Exercise 3 + 4: The Average pizza
    average_pizza = np.mean(data_matrix, 0)
    io.imshow(create_u_byte_image_from_vector(average_pizza, height, width, channels))
    plt.title('The Average Pizza')
    io.show()

    # Find the missing pizza twin
    # Exercise 7 + 8
    # im_miss = io.imread("data/pizzaPCA/super_pizza.png")
    # im_miss_flat = im_miss.flatten()

    # Find pizza closest to the average pizza
    sub_data = data_matrix - average_pizza
    sub_distances = np.linalg.norm(sub_data, axis=1)

    # Exercise 9 + 10 + 11
    best_match = np.argmin(sub_distances)
    best_average_pizza = data_matrix[best_match, :]
    worst_match = np.argmax(sub_distances)
    worst_average_pizza = data_matrix[worst_match, :]

    print(f"Pizza most away from average pizza {worst_match} : {all_images[worst_match]}")

    fig, ax = plt.subplots(ncols=3, figsize=(16, 6))
    ax[0].imshow(create_u_byte_image_from_vector(average_pizza, height, width, channels))
    ax[0].set_title('The Real average pizza')
    ax[1].imshow(create_u_byte_image_from_vector(best_average_pizza, height, width, channels))
    ax[1].set_title('The Best Matching Twin pizza')
    ax[2].imshow(create_u_byte_image_from_vector(worst_average_pizza, height, width, channels))
    ax[2].set_title('The Worst Matching Twin pizza')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()

    # Exercise 12
    print("Computing PCA")
    # pizzas_pca = PCA(n_components=10)
    pizzas_pca = PCA(n_components=5)
    # pizzas_pca = PCA(n_components=0.80)
    pizzas_pca.fit(data_matrix)

    # Exercise 13
    plt.plot(pizzas_pca.explained_variance_ratio_ * 100)
    plt.xlabel('Principal component')
    plt.ylabel('Percent explained variance')
    plt.show()

    # Exercise 14
    print(f"Total variation explained by first component {pizzas_pca.explained_variance_ratio_[0] * 100}")
    print(f"Number of component to explain desired variation {pizzas_pca.n_components_}")

    # Exercise 15
    components = pizzas_pca.transform(data_matrix)

    # Exercise 16
    pc_1 = components[:, 0]
    pc_2 = components[:, 1]

    # Debug
    print(f"PC_1 : {pc_1.min()} - {pc_1.max()}")
    print(f"PC_2 : {pc_2.min()} - {pc_2.max()}")
    print(f"Explained variance: {pizzas_pca.explained_variance_[0]} {pizzas_pca.explained_variance_[1]}")
    print(f"SQRT Explained variance: {np.sqrt(pizzas_pca.explained_variance_[0])} "
          f"{np.sqrt(pizzas_pca.explained_variance_[1])}")

    plt.plot(pc_1, pc_2, '.')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

    # Exercise 17 + 18
    extreme_pc_1_pizza_m = np.argmin(pc_1)
    extreme_pc_1_pizza_p = np.argmax(pc_1)
    extreme_pc_2_pizza_m = np.argmin(pc_2)
    extreme_pc_2_pizza_p = np.argmax(pc_2)

    print(f'PC 1 extreme minus pizza: {all_images[extreme_pc_1_pizza_m]}')
    print(f'PC 1 extreme minus pizza: {all_images[extreme_pc_1_pizza_p]}')
    print(f'PC 2 extreme minus pizza: {all_images[extreme_pc_2_pizza_m]}')
    print(f'PC 2 extreme minus pizza: {all_images[extreme_pc_2_pizza_p]}')

    fig, ax = plt.subplots(ncols=4, figsize=(16, 6))
    ax[0].imshow(create_u_byte_image_from_vector(data_matrix[extreme_pc_1_pizza_m, :], height, width, channels))
    ax[0].set_title(f'PC 1 extreme minus pizza')
    ax[1].imshow(create_u_byte_image_from_vector(data_matrix[extreme_pc_1_pizza_p, :], height, width, channels))
    ax[1].set_title(f'PC 1 extreme plus pizza')
    ax[2].imshow(create_u_byte_image_from_vector(data_matrix[extreme_pc_2_pizza_m, :], height, width, channels))
    ax[2].set_title(f'PC 2 extreme minus pizza')
    ax[3].imshow(create_u_byte_image_from_vector(data_matrix[extreme_pc_2_pizza_p, :], height, width, channels))
    ax[3].set_title(f'PC 2 extreme plus pizza')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()

    plt.plot(pc_1, pc_2, '.', label="All pizzas")
    plt.plot(pc_1[extreme_pc_1_pizza_m], pc_2[extreme_pc_1_pizza_m], "*", color="green", label="Extreme pizza 1 minus")
    plt.plot(pc_1[extreme_pc_1_pizza_p], pc_2[extreme_pc_1_pizza_p], "*", color="green", label="Extreme pizza 1 plus")
    plt.plot(pc_1[extreme_pc_2_pizza_m], pc_2[extreme_pc_2_pizza_m], "*", color="green", label="Extreme pizza 2 minus")
    plt.plot(pc_1[extreme_pc_2_pizza_p], pc_2[extreme_pc_2_pizza_p], "*", color="green", label="Extreme pizza 2 plus")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("pizzas in PCA space")
    plt.legend()
    plt.show()

    # Exercise 19 + 20: First synthetic pizza
    w = 60000
    synth_pizza = average_pizza + w * pizzas_pca.components_[0, :]
    io.imshow(create_u_byte_image_from_vector(synth_pizza, height, width, channels))
    plt.title("First synthetic pizza")
    plt.show()

    # Exercise 22: Major modes of variation
    n_modes = 5
    # Create a n_modes x 3 plot to show major modes of variation
    fig, ax = plt.subplots(ncols=3, nrows=n_modes, figsize=(15, 15))
    for m in range(n_modes):
        # Average pizza in middle of all
        ax[m][1].set_title("The Mean pizza")
        ax[m][1].imshow(create_u_byte_image_from_vector(average_pizza, height, width, channels))
        # Create mode: synth_pizza = average_pizza + alpha * eigenvector[m]
        synth_pizza_plus = average_pizza + 3 * np.sqrt(pizzas_pca.explained_variance_[m]) * pizzas_pca.components_[m, :]
        synth_pizza_minus = average_pizza - 3 * np.sqrt(pizzas_pca.explained_variance_[m]) * pizzas_pca.components_[m, :]
        ax[m][0].set_title(f"Mode: {m} minus")
        ax[m][2].set_title(f"Mode: {m} plus")
        ax[m][0].imshow(create_u_byte_image_from_vector(synth_pizza_minus, height, width, channels))
        ax[m][2].imshow(create_u_byte_image_from_vector(synth_pizza_plus, height, width, channels))
        ax[m][0].axis('off')
        ax[m][1].axis('off')
        ax[m][2].axis('off')
    fig.suptitle("Major modes of pizza variations")
    plt.tight_layout()
    plt.show()

    print(f"Computing synthetic pizzas")
    n_random = 3
    n_components_to_use = 10
    n_components_to_use = min(n_components_to_use, pizzas_pca.n_components_)
    fig, ax = plt.subplots(ncols=n_random, nrows=n_random, figsize=(10, 10))
    for i in range(n_random):
        for j in range(n_random):
            synth_pizza = average_pizza
            for idx in range(n_components_to_use):
                w = random.uniform(-1, 1) * 3 * np.sqrt(pizzas_pca.explained_variance_[idx])
                synth_pizza = synth_pizza + w * pizzas_pca.components_[idx, :]
            ax[i][j].imshow(create_u_byte_image_from_vector(synth_pizza, height, width, channels))
            ax[i][j].axis('off')
    fig.suptitle("Some Random Synthetic pizzas")
    plt.tight_layout()
    plt.show()

    # Exercise 24: Find the missing pizza twin
    im_miss = io.imread("data/pizzaPCA/super_pizza.png")
    im_miss_flat = im_miss.flatten()
    im_miss_flat = im_miss_flat.reshape(1, -1)
    pca_coords = pizzas_pca.transform(im_miss_flat)
    pca_coords = pca_coords.flatten()

    # Exercise 25
    plt.plot(pc_1, pc_2, '.', label="All pizzas")
    plt.plot(pca_coords[0], pca_coords[1], "*", color="red", label="Missing pizza")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("The Missing pizza in PCA space")
    plt.legend()
    plt.show()

    # Exercise 26
    synth_pizza = average_pizza
    for idx in range(pizzas_pca.n_components_):
        synth_pizza = synth_pizza + pca_coords[idx] * pizzas_pca.components_[idx, :]

    fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
    ax[0].imshow(im_miss)
    ax[0].set_title('The Real Missing pizza')
    ax[1].imshow(create_u_byte_image_from_vector(synth_pizza, height, width, channels))
    ax[1].set_title('The Synthetic Missing pizza')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()

    # Exercise 27
    comp_sub = components - pca_coords
    pca_distances = np.linalg.norm(comp_sub, axis=1)

    best_match = np.argmin(pca_distances)
    print(f"Answer: Best matching PCA pizza {all_images[best_match]}")
    best_twin_pizza = data_matrix[best_match, :]
    worst_match = np.argmax(pca_distances)
    worst_twin_pizza = data_matrix[worst_match, :]
    fig, ax = plt.subplots(ncols=3, figsize=(16, 6))
    ax[0].imshow(im_miss)
    ax[0].set_title('The Real Missing pizza')
    ax[1].imshow(create_u_byte_image_from_vector(best_twin_pizza, height, width, channels))
    ax[1].set_title('The Best Matching Twin pizza')
    ax[2].imshow(create_u_byte_image_from_vector(worst_twin_pizza, height, width, channels))
    ax[2].set_title('The Worst Matching Twin pizza')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()

    # Exercise 28
    n_best = 5
    best = np.argpartition(pca_distances, n_best)
    fig, ax = plt.subplots(ncols=n_best, figsize=(16, 6))
    for i in range(n_best):
        candidate_twin_pizza = data_matrix[best[i], :]
        ax[i].imshow(create_u_byte_image_from_vector(candidate_twin_pizza, height, width, channels))
        ax[i].axis('off')

    fig.suptitle(f"The {n_best} most similar pizzas")
    plt.tight_layout()
    plt.show()


# https://seaborn.pydata.org/generated/seaborn.heatmap.html
def create_exam_images():
    o_name = "exam_image_1.png"
    m = 8
    n = 8
    img = np.random.randint(0, 255, size=(m, n))
    # array = [[13, 17, 190, 210, 140, 130],
    #          [23, 25, 26, 43, 56, 103],
    #          [45, 61, 245, 240, 234, 210],
    #          [47, 63, 248, 120, 110, 90],
    #          [89, 76, 203, 134, 123, 75],
    #          [98, 78, 196, 144, 134, 15]]

    df_cm = pd.DataFrame(img) #, range(6), range(6))
    # plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4)  # for label size
    # https://stackoverflow.com/questions/29647749/seaborn-showing-scientific-notation-in-heatmap-for-3-digit-numbers
    ax = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16, "color": 'cyan'},  vmin=0,
                    vmax=255, cmap="gray", fmt='g')  # font layout
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')
    plt.savefig(o_name)
    plt.show()


def car_rgb_hsv_thresholds():
    in_dir = "data/"
    im_name = "car.png"
    im_org = io.imread(in_dir + im_name)
    hsv_img = color.rgb2hsv(im_org)
    # hue_img = hsv_img[:, :, 0]
    # value_img = hsv_img[:, :, 2]
    s_img = hsv_img[:, :, 1]

    segm_car = (s_img > 0.7)
    io.imshow(segm_car)
    plt.title('Segmented car')
    io.show()

    print(f"Segm Result {segm_car.sum()}")

    footprint = disk(6)
    eroded = erosion(segm_car, footprint)

    footprint = disk(4)
    dilated = dilation(eroded, footprint)

    result = dilated.sum()
    print(f"Result {result}")

    footprint = disk(6)
    eroded = binary_erosion(segm_car, footprint)

    footprint = disk(4)
    dilated = binary_dilation(eroded, footprint)

    result = dilated.sum()
    print(f"Result 2 {result}")

    io.imshow(dilated)
    plt.title('Cleaned car')
    io.show()


def road_analysis():
    in_dir = "data/"
    im_name = "road.png"
    im_org = io.imread(in_dir + im_name)
    hsv_img = color.rgb2hsv(im_org)
    value_img = hsv_img[:, :, 2]

    segm_road = (value_img > 0.9)
    io.imshow(segm_road)
    plt.title('Segmented Road')
    io.show()

    label_img = measure.label(segm_road)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")

    region_props = measure.regionprops(label_img)

    areas = np.array([prop.area for prop in region_props])

    # Trick to sort descending
    areas_sort = -np.sort(-areas)
    print(f"Area 1 {areas_sort[0]} area 2 {areas_sort[1]}")
    answer = areas_sort[1]
    print(f"BLOBS with area less than {answer} should be removed")


def aorta_blob_analysis():
    in_dir = "data/"
    im_name = "1-442.dcm"

    ct = dicom.read_file(in_dir + im_name)
    img = ct.pixel_array

    t_1 = 90

    bin_img = (img > t_1)
    spleen_label_colour = color.label2rgb(bin_img)
    io.imshow(spleen_label_colour)
    plt.title("First aorta estimate")
    io.show()

    img_c_b = segmentation.clear_border(bin_img)
    label_img = measure.label(img_c_b)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")
    region_props = measure.regionprops(label_img)

    min_area = 200
    min_circ = 0.94

    # Create a copy of the label_img
    label_img_filter = label_img.copy()
    for region in region_props:
        a = region.area
        p = region.perimeter
        circ = 0
        if p > 0:
            circ = 4 * math.pi * a / (p * p)

        if p < 1 or a < min_area or circ < min_circ:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0

    # Create binary image from the filtered label image
    i_aorta = label_img_filter > 0
    # show_comparison(img, i_area, 'Found spleen based on area')
    io.imshow(i_aorta)
    io.show()

    i_area = label_img_filter > 0
    pix_area = i_area.sum()
    one_pix = 0.75 * 0.75
    print(f"Number of pixels {pix_area} and {pix_area * one_pix:.0f} mm2")


def aorta_pixel_values():
    in_dir = "data/"
    im_name = "1-442.dcm"

    ct = dicom.read_file(in_dir + im_name)
    img = ct.pixel_array

    aorta_roi = io.imread(in_dir + 'AortaROI.png')
    aorta_mask = aorta_roi > 0
    aorta_values = img[aorta_mask]
    (mu_aorta, std_aorta) = norm.fit(aorta_values)
    print(f"Average {mu_aorta:.0f} standard deviation {std_aorta:.0f}")

    liver_roi = io.imread(in_dir + 'LiverROI.png')
    liver_mask = liver_roi > 0
    liver_values = img[liver_mask]
    (mu_liver, std_liver) = norm.fit(liver_values)

    min_hu = 147
    max_hu = 155
    hu_range = np.arange(min_hu, max_hu, 1.0)
    pdf_aorta = norm.pdf(hu_range, mu_aorta, std_aorta)
    pdf_liver = norm.pdf(hu_range, mu_liver, std_liver)
    plt.plot(hu_range, pdf_aorta, 'r--', label="aorta")
    plt.plot(hu_range, pdf_liver, 'g', label="liver")
    plt.title("Fitted Gaussians")
    plt.legend()
    plt.show()
    # Answer = 151


def point_transformation():
    v = math.radians(20)
    a_scale = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    a_rot = np.array([[math.cos(v), -math.sin(v), 0], [math.sin(v), math.cos(v), 0],
                     [0, 0, 1]])
    a_trans = np.array([[1, 0, 3.1], [0, 1, -3.3], [0, 0, 1]])

    a_tot = np.matmul(a_trans, np.matmul(a_scale, a_rot))
    p = np.array([10, 10, 1])
    p_out = np.matmul(a_tot, p)
    print(p_out)


def haar_feature():
    feature = 178 + 60 + 155 + 252 - 168 - 217 - 159 - 223 - 97 - 136 - 32 - 108
    print(f"Haar feature: {feature}")

    integral_value = 33 + 12 + 110 + 200 + 53 + 81 + 220 + 120 + 107
    print(f"Integral value {integral_value}")


def system_frame_rate():
    # bytes per second
    transfer_speed = 30000000
    image_mb = 1024 * 768 * 3
    images_per_second = transfer_speed / image_mb
    print(f"Images transfered per second {images_per_second:.1f}")

    proc_time = 0.054
    proc_per_second = 1/proc_time
    print(f"Images processed per second {proc_per_second:.1f}")

    max_fps = min(proc_per_second, images_per_second)
    print(f"System framerate {max_fps:.1f}")


def crop_change_images():
    name = 'data\ChangeDetection\change1.jpg'
    o_name = 'data\ChangeDetection\change1.png'

    # (640, 1360) (2470, 2800)
    r_start = 1360
    r_end = 2800
    c_start = 640
    c_end = 2470
    im_org = io.imread(name)
    im_crop = im_org[r_start:r_end, c_start:c_end, :]

    image_resized = resize(im_crop, (im_crop.shape[0] // 4, im_crop.shape[1] // 4),
                           anti_aliasing=True)

    io.imsave(o_name, img_as_ubyte(image_resized))



def change_detection():
    name_1 = 'data\ChangeDetection\change1.png'
    name_2 = 'data\ChangeDetection\change2.png'

    im_1 = io.imread(name_1)
    im_2 = io.imread(name_2)
    im_1_g = color.rgb2gray(im_1)
    im_2_g = color.rgb2gray(im_2)

    print(im_1_g.shape)
    print(im_1_g.dtype)

    dif_thres = 0.3
    dif_img = np.abs(im_1_g - im_2_g)
    io.imshow(dif_img)
    io.show()
    dif_bin = (dif_img > dif_thres)
    io.imshow(dif_bin)
    io.show()
    changed_pixels = np.sum(dif_bin)
    total_pixel = im_1_g.shape[0] * im_1_g.shape[1]
    img_size = im_1_g.size
    change_per = float(changed_pixels) / float(total_pixel) * 100
    print(f"Changed percent {change_per:.0f}")


def linear_stretch_and_otsus():
    name_1 = 'data\PixelWiseOps\pixelwise.png'

    im_1 = io.imread(name_1)
    im_1_g = color.rgb2gray(im_1)

    min_val = im_1_g.min()
    max_val = im_1_g.max()

    print(f"Float image minimum pixel value: {min_val} and max value: {max_val}")

    min_desired = 0.1
    max_desired = 0.6

    img_out = (max_desired - min_desired) / (max_val - min_val) * (im_1_g - min_val) + min_desired

    min_val = img_out.min()
    max_val = img_out.max()
    print(f"Out float image minimum pixel value: {min_val} and max value: {max_val}")

    auto_tresh = threshold_otsu(im_1_g)
    print(f"Otsus treshold on original {auto_tresh:.2f}")

    auto_tresh = threshold_otsu(img_out)
    print(f"Otsus treshold {auto_tresh:.2f}")


    #img_thres = img_out > auto_tresh
    #io.imshow(img_thres)
    # plt.title('Otsus thresholded image')
    # io.show()
    #plt.savefig('otsu_1.png')
    # plt.show()

    auto_tresh = 0.50
    img_thres = img_out > auto_tresh
    io.imshow(img_thres)
    plt.savefig('otsu_2.png')

    auto_tresh = 0.45
    img_thres = img_out > auto_tresh
    io.imshow(img_thres)
    plt.savefig('otsu_3.png')

    auto_tresh = 0.20
    img_thres = img_out > auto_tresh
    io.imshow(img_thres)
    plt.savefig('otsu_4.png')

    auto_tresh = 0.25
    img_thres = img_out > auto_tresh
    io.imshow(img_thres)
    plt.savefig('otsu_5.png')


def rgb_to_hsv_threshold():
    name_1 = 'data\PixelWiseOps\pixelwise.png'

    im_1 = io.imread(name_1)

    hsv_img = color.rgb2hsv(im_1)
    # hue_img = hsv_img[:, :, 0]
    sat_img = hsv_img[:, :, 1]
    # value_img = hsv_img[:, :, 2]

    # auto_tresh = threshold_otsu(hue_img)
    # print(f"Otsus treshold {auto_tresh:.2f}")

    # segm_otsu_hue = (hue_img > auto_tresh)
    # io.imshow(segm_otsu_hue)
    # plt.title('Otsu hue')
    # io.show()

    # auto_tresh = threshold_otsu(value_img)
    # segm_otsu = (value_img > auto_tresh)
    # io.imshow(segm_otsu)
    # plt.title('Otsu value')
    # io.show()

    auto_tresh = threshold_otsu(sat_img)
    segm_otsu = (sat_img > auto_tresh)
    io.imshow(segm_otsu)
    plt.title('Otsu saturation')
    io.show()

    footprint = disk(4)
    eroded = erosion(segm_otsu, footprint)
    io.imshow(eroded)
    plt.title('eroded')
    io.show()

    num_pixel = eroded.sum()
    print(f"Foreground pixels: {num_pixel}")


def gaussian_filtering():
    in_dir = "data/Filtering/"
    im_name = "rocket.png"
    im_org = io.imread(in_dir + im_name)
    io.imshow(im_org)
    plt.title('Rocket')
    io.show()

    sigma = 3
    gauss_img = gaussian(im_org, sigma)
    io.imshow(gauss_img)
    plt.title('Gaussian filtered image')
    io.show()

    img_b = img_as_ubyte(gauss_img)
    print(f"Value at (100, 100) : {img_b[100, 100]}")


def edge_filtering():
    in_dir = "data/Filtering/"
    im_name = "rocket.png"
    im_org = io.imread(in_dir + im_name)
    # io.imshow(im_org)
    # plt.title('Rocket')
    # io.show()

    edge_img = prewitt(img_as_ubyte(im_org))
    min_val = edge_img.min()
    max_val = edge_img.max()
    io.imshow(edge_img, vmin=min_val, vmax=max_val, cmap="terrain")
    plt.title('Prewitt filtered image')
    io.show()

    bin_edges = edge_img > 0.06
    io.imshow(bin_edges)
    plt.title('Binary edges. Manual threshold')
    io.show()

    num_pixels = bin_edges.sum()
    print(f"Number of edge pixels {num_pixels}")


def blob_analysis():
    in_dir = "data/BLOBs/"
    im_name = "figures.png"
    im_org = io.imread(in_dir + im_name)
    # io.imshow(im_org)
    # plt.title('Rocket')
    # io.show()
    im_g = color.rgb2gray(im_org)

    auto_tresh = threshold_otsu(im_g)
    print(f"Otsus treshold on original {auto_tresh:.2f}")

    img_thres = im_g < auto_tresh
    io.imshow(img_thres)
    plt.title('Otsus thresholded image')
    io.show()
    bin_img = img_thres

    img_c_b = segmentation.clear_border(bin_img)

    label_img = measure.label(img_c_b)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")

    image_label_overlay = label2rgb(label_img)
    io.imshow(image_label_overlay)
    plt.title('Found blobs')
    io.show()

    # io.imshow(image_label_overlay)
    # io.show()
    region_props = measure.regionprops(label_img)
    # print(region_props)

    areas = np.array([prop.area for prop in region_props])
    areas.sort()
    print(areas)
    plt.hist(areas, bins=50)
    plt.show()

    for region in region_props:
        print(f"Area {region.area} perimeter {region.perimeter}")



def multi_organ_exploration():
    in_dir = "data/dicom/"
    ct = dicom.read_file(in_dir + '1-162.dcm')
    ground_truth_img = io.imread(in_dir + 'KidneyROI.png')

    img = ct.pixel_array

    liver_roi = io.imread(in_dir + 'LiverROI.png')
    liver_mask = liver_roi > 0
    liver_values = img[liver_mask]
    # liver_mean = np.average(liver_values)
    # liver_std = np.std(liver_values)
    (mu_liver, std_liver) = norm.fit(liver_values)

    kidney_roi = io.imread(in_dir + 'KidneyROI.png')
    kidney_mask = kidney_roi > 0
    kidney_values = img[kidney_mask]
    (mu_kidney, std_kidney) = norm.fit(kidney_values)

    aorta_roi = io.imread(in_dir + 'AortaROI.png')
    aorta_mask = aorta_roi > 0
    aorta_values = img[aorta_mask]
    (mu_aorta, std_aorta) = norm.fit(aorta_values)

    back_roi = io.imread(in_dir + 'BackROI.png')
    back_mask = back_roi > 0
    back_values = img[back_mask]
    (mu_back, std_back) = norm.fit(back_values)

    # Hounsfield unit limits of the plot
    # min_hu = -200
    # max_hu = 1000
    # hu_range = np.arange(min_hu, max_hu, 1.0)
    # pdf_spleen = norm.pdf(hu_range, mu_spleen, std_spleen)
    # pdf_aorta = norm.pdf(hu_range, mu_aorta, std_aorta)
    # plt.plot(hu_range, pdf_spleen, 'r--', label="spleen")
    # plt.plot(hu_range, pdf_aorta, 'g', label="aorta")
    # plt.title("Fitted Gaussians")
    # plt.legend()
    # plt.show()


    # Hounsfield unit limits of the plot
    min_hu = -200
    max_hu = 500
    hu_range = np.arange(min_hu, max_hu, 1.0)
    pdf_back = norm.pdf(hu_range, mu_back, std_back)
    pdf_aorta = norm.pdf(hu_range, mu_aorta, std_aorta)
    pdf_liver = norm.pdf(hu_range, mu_liver, std_liver)
    pdf_kidney = norm.pdf(hu_range, mu_kidney, std_kidney)
    # pdf_fat = norm.pdf(hu_range, mu_fat, std_fat)
    plt.plot(hu_range, pdf_back, 'r--', label="back")
    plt.plot(hu_range, pdf_aorta, 'g--', label="aorta")
    plt.plot(hu_range, pdf_liver, label="liver")
    plt.plot(hu_range, pdf_kidney, label="kidney")
    # plt.plot(hu_range, pdf_fat, label="fat")
    plt.title("Fitted Gaussians")
    plt.legend()
    plt.show()

    t_liver_kidney = (mu_liver + mu_kidney) / 2
    t_kidney_aorta = (mu_kidney + mu_aorta) / 2
    print(f"Thresholds: {t_liver_kidney:.1f}, {t_kidney_aorta:.1f}")

    # t_background = -200
    kidney_img = (img > t_liver_kidney) & (img < t_kidney_aorta)
    io.imshow(kidney_img)
    io.show()

    gt_bin = ground_truth_img > 0
    dice_score = 1 - distance.dice(kidney_img.ravel(), gt_bin.ravel())
    print(f"DICE score {dice_score:.3f}")


# https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.rotate
def rotate_image():
    in_dir = "data/GeomTrans/"
    im_name = "CPHSun.png"
    im_org = io.imread(in_dir + im_name)

    # angle in degrees - counter clockwise
    rotation_angle = 16
    rot_center = [20, 20]
    rotated_img = rotate(im_org, rotation_angle, center=rot_center)
    rot_byte = img_as_ubyte(rotated_img)
    print(f"Value at (200, 200) : {rot_byte[200, 200]}")
    io.imshow(rot_byte)
    io.show()


def shortest_path_cost():
    cost = 19 + 23 + 17 + 18 + 44
    print(f"Path cost {cost}")


def cow_sheep_classifier():
    cows = [26, 46, 33, 23, 35, 28, 21, 30, 38, 43]
    sheep = [67, 27, 40, 60, 39, 45, 27, 67, 43, 50, 37, 100]

    (mu_cows, std_cows) = norm.fit(cows)
    (mu_sheep, std_sheep) = norm.fit(sheep)

    min_dist_thres = (mu_sheep + mu_cows) / 2
    print(f"Min dist threshold {min_dist_thres}")

    min_val = 20
    max_val = 110
    val_range = np.arange(min_val, max_val, 0.2)
    pdf_cows = norm.pdf(val_range, mu_cows, std_cows)
    pdf_sheep = norm.pdf(val_range, mu_sheep, std_sheep)

    test_val = 38
    cow_prob = norm.pdf(test_val, mu_cows, std_cows)
    sheep_prob = norm.pdf(test_val, mu_sheep, std_sheep)
    print(f"Cow probability {cow_prob:.2f}")
    print(f"Sheep probability {sheep_prob:.2f}")

    plt.plot(val_range, pdf_cows, 'r--', label="cows")
    plt.plot(val_range, pdf_sheep, 'g', label="sheep")
    plt.title("Fitted Gaussians")
    plt.legend()
    plt.show()


def hough_space():
    xs = [7, 9, 6, 6, 3]
    ys = [13, 10, 10, 8, 6]

    theta = 151
    rho = 0.29

    for x in xs:
        y = -x * (math.cos(math.radians(theta)) / math.sin(math.radians(theta))) + \
            rho * 1 / math.sin(math.radians(theta))
        print(f"Found pairs (x,y) = ({x}, {y:.0f})")

    # it is found that (7, 13) and (3, 6) are the correct pairs


def landmark_based_registration():
    in_dir = "data/GeomTrans/"
    src_img = io.imread(in_dir + 'rocket.png')

    # src = np.array([[55, 220], [675, 105], [675, 315]])
    # dst = np.array([[165, 100], [605, 200], [525, 379]])

    src = np.array([[220, 55], [105, 675], [315, 675]])
    dst = np.array([[100, 165], [200, 605], [379, 525]])

    e_x = src[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Landmark alignment error F: {f}")

    plt.imshow(src_img)
    plt.plot(src[:, 0], src[:, 1], '.r', markersize=12)
    plt.title("Source image")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(src[:, 0], src[:, 1], '*r', markersize=12, label="Source")
    ax.plot(dst[:, 0], dst[:, 1], '*g', markersize=12, label="Destination")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Landmarks before alignment")
    plt.show()

    # plt.scatter(src[:, 0], src[:, 1])
    # plt.scatter(trg[:, 0], trg[:, 1])
    # plt.show()
    tform = EuclideanTransform()
    tform.estimate(src, dst)

    src_transform = matrix_transform(src, tform.params)
    # print(src_transform)

    e_x = src_transform[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src_transform[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Aligned landmark alignment error F: {f}")

    fig, ax = plt.subplots()
    ax.plot(src[:, 0], src[:, 1], '*r', markersize=12, label="Source")
    ax.plot(src_transform[:, 0], src_transform[:, 1], '*b', markersize=12, label="Source transformed")
    ax.plot(dst[:, 0], dst[:, 1], '*g', markersize=12, label="Destination")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Landmarks after alignment")
    plt.show()

    warped = warp(src_img, tform.inverse)

    print(f"Value at (150, 150) : {img_as_ubyte(warped)[150, 150]}")

    fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
    ax[0].imshow(src_img)
    ax[0].plot(src[:, 0], src[:, 1], '.r', markersize=12)
    # ax[1].plot(dst[:, 0], dst[:, 1], '.r', markersize=12)
    ax[1].imshow(warped)
    ax[1].plot(src_transform[:, 0], src_transform[:, 1], '.r', markersize=12)
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()

# The @ operator is a matrix multiplication operator
def lda_classification():
    pooled_cov = 2 * np.eye(2)
    group_mean = np.array([[24, 3], [30, 7]])
    x = np.array([[23], [5]])
    group_diff = group_mean[1, :] - group_mean[0, :]
    group_diff = group_diff[:, None]

    w = np.linalg.inv(pooled_cov) @ group_diff
    c = -0.5 * np.sum(group_mean, axis=0, keepdims=True)
    w0 = c @ w
    y = x.T @ w + w0
    print(y)

    pooled_cov = 2 * np.eye(2)  # or 4*np.eye(2)?
    group_mean = np.array([[24, 3], [30, 7]])
    prior_prob = [[0.5], [0.5]]
    m = 2  # n dimensions
    k = 2  # n classes
    W = np.zeros((k, m + 1))

    for i in range(k):
        # Intermediate calculation for efficiency
        temp = group_mean[i, :][np.newaxis] @ np.linalg.inv(pooled_cov)
        # Constant
        W[i, 0] = -0.5 * temp @ group_mean[i, :].T + np.log(prior_prob[i])
        # Linear
        W[i, 1:] = temp

    Y = np.array([[1, 23, 5]]) @ W.T
    posteriorProb = np.clip(np.exp(Y) / np.sum(np.exp(Y), 1)[:, np.newaxis], 0, 1)

    print(Y)
    print(posteriorProb)


if __name__ == '__main__':
    # pca_on_glass_data_F2023()
    # change_detection_F2023()
    # system_frame_rate_F2023()
    # nike_rgb_hsv_thresholds_F2023()
    # filtering_F2023()
    # letters_blob_analysis_F2023()
    # kidney_pixel_analysis_F2023()
    otsu_rotate_image_F2023()
    # landmark_based_registration_F2023()
    # do_pca_on_all_images_in_directory_F2023()
