import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from skimage.morphology import erosion
from skimage.morphology import disk
from skimage.filters import prewitt
import math
from scipy.stats import norm
import pandas as pd
import seaborn as sns
from skimage import color, io, measure, segmentation, img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from skimage.color import label2rgb
from scipy.spatial import distance
from skimage.transform import rotate
from skimage.transform import EuclideanTransform
from skimage.transform import warp
from skimage.transform import matrix_transform


# E2022
def pca_on_car_data():
    in_dir = "data/CarPCA/"
    txt_name = "car_data.txt"

    car_data = np.loadtxt(in_dir + txt_name, comments="%")
    x = car_data
    n_feat = x.shape[1]
    n_obs = x.shape[0]
    print(f"Number of features: {n_feat} and number of observations: {n_obs}")

    # plt.figure()
    # Transform the data into a Pandas dataframe
    # d = pd.DataFrame(x)
    # sns.pairplot(d)
    # plt.show()

    mn = np.mean(x, axis=0)
    data = x - mn
    data = data / data.std(axis=0)
    c_x = np.cov(data.T)

    print(f"First answer at top of matrix {data[0][0]:.2f}")

    values, vectors = np.linalg.eig(c_x)

    v_norm = values / values.sum() * 100
    plt.plot(v_norm)
    plt.xlabel('Principal component')
    plt.ylabel('Percent explained variance')
    plt.ylim([0, 100])
    plt.show()

    answer = v_norm[0] + v_norm[1]
    print(f"Answer: Variance explained by the first two PC: {answer:.2f}")

    # Project data
    pc_proj = vectors.T.dot(data.T)
    print(f"Projected answer: {pc_proj[0][0]:.2f}")

    pc_proj_red = pc_proj[0:3, :]
    # pc_proj_red = pc_proj[5:8, :]

    # Answer 3
    plt.figure()
    # Transform the data into a Pandas dataframe
    d = pd.DataFrame(pc_proj_red.T)
    sns.pairplot(d)
    # plt.savefig('pairplot_5.png')
    plt.show()


# E2022
def haar_feature():
    feature = 178 + 60 + 155 + 252 - 168 - 217 - 159 - 223 - 97 - 136 - 32 - 108
    print(f"Haar feature: {feature}")

    integral_value = 33 + 12 + 110 + 200 + 53 + 81 + 220 + 120 + 107
    print(f"Integral value {integral_value}")


# E2022
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


# E2022
def we_cu_change_detection():
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


# E2022
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

    # auto_tresh = threshold_otsu(im_1_g)
    # print(f"Otsus treshold on original {auto_tresh:.2f}")

    auto_tresh = threshold_otsu(img_out)
    print(f"Otsus treshold {auto_tresh:.2f}")

    img_thres = img_out > auto_tresh
    io.imshow(img_thres)
    plt.title('Otsus thresholded image')
    io.show()


# E2022
def car_tracking_rgb_to_hsv_threshold():
    name_1 = 'data\PixelWiseOps\pixelwise.png'

    im_1 = io.imread(name_1)

    hsv_img = color.rgb2hsv(im_1)
    # hue_img = hsv_img[:, :, 0]
    sat_img = hsv_img[:, :, 1]
    # value_img = hsv_img[:, :, 2]

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


# E2022
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


# E2022
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


# E2022
def blob_analysis_mini_figures():
    in_dir = "data/BLOBs/"
    im_name = "figures.png"
    im_org = io.imread(in_dir + im_name)
    im_g = color.rgb2gray(im_org)

    auto_tresh = threshold_otsu(im_g)
    # print(f"Otsus treshold on original {auto_tresh:.2f}")

    img_thres = im_g < auto_tresh
    io.imshow(img_thres)
    plt.title('Otsus thresholded image')
    io.show()
    bin_img = img_thres

    img_c_b = segmentation.clear_border(bin_img)

    label_img = measure.label(img_c_b)
    n_labels = label_img.max()
    # print(f"Number of labels: {n_labels}")

    image_label_overlay = label2rgb(label_img)
    io.imshow(image_label_overlay)
    plt.title('Found blobs')
    io.show()

    region_props = measure.regionprops(label_img)

    areas = np.array([prop.area for prop in region_props])
    areas.sort()
    # print(areas)
    # plt.hist(areas, bins=50)
    # plt.show()
    count_big = areas > 13000
    print(f"Number of BLOBs with an area larger than 13000 : {sum(count_big)}")
    largest_area = areas[len(areas) - 1]

    for region in region_props:
        if region.area == largest_area:
            print(f"Area {region.area} perimeter {region.perimeter}")


# E2022
def abdominal_analysis():
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
    min_hu = -200
    max_hu = 500
    hu_range = np.arange(min_hu, max_hu, 1.0)
    pdf_back = norm.pdf(hu_range, mu_back, std_back)
    pdf_aorta = norm.pdf(hu_range, mu_aorta, std_aorta)
    pdf_liver = norm.pdf(hu_range, mu_liver, std_liver)
    pdf_kidney = norm.pdf(hu_range, mu_kidney, std_kidney)
    plt.plot(hu_range, pdf_back, 'r--', label="back")
    plt.plot(hu_range, pdf_aorta, 'g--', label="aorta")
    plt.plot(hu_range, pdf_liver, label="liver")
    plt.plot(hu_range, pdf_kidney, label="kidney")
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


# E2022
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


# E2022
def shortest_path_cost():
    cost = 19 + 23 + 17 + 18 + 44
    print(f"Path cost {cost}")


# E2022
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


# E2022
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


# E2022
def landmark_based_registration():
    in_dir = "data/GeomTrans/"
    src_img = io.imread(in_dir + 'rocket.png')

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


# E2022
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
    abdominal_analysis()
    hough_space()
    shortest_path_cost()
    haar_feature()
    linear_stretch_and_otsus()
    edge_filtering()
    lda_classification()
    cow_sheep_classifier()
    blob_analysis_mini_figures()
    pca_on_car_data()
    landmark_based_registration()
    gaussian_filtering()
    rotate_image()
    car_tracking_rgb_to_hsv_threshold()
    we_cu_change_detection()
    system_frame_rate()
