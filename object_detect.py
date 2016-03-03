from __future__ import print_function
from liblas import file
from pylab import *
import matplotlib.pyplot as plt
import gdal
from osgeo import osr
import os
from skimage import transform
from skimage.feature import peak_local_max, match_template
from scipy import ndimage
from visual import shapes
import sys
import numpy as np
import numba
import masks

__author__ = 'mkatic'

# output_dir = os.path.join(os.getcwd(), 'output')
output_dir = os.path.join('D:', 'SRP', 'object_detect_output')


# # @numba.njit
# def rasterize_point(point, color, w, h, max_, min_, counts, maxes, mins, colors):
#     point = (point - min_) / (max_ - min_)
#     col = int(point[0] * (w - 1))
#     row = int(point[1] * (h - 1))
#     if point[2] < mins[col, row]:
#         mins[col, row] = min(mins[col, row], point[2])
#     if point[2] > maxes[col, row]:
#         maxes[col, row] = point[2]
#         # colors[col, row, :] = color
#     counts[col, row] += 1
#
#
# # @numba.jit
# def rasterize_las(las_file, w, h, min_x, max_x, min_y, max_y):
#     max_ = np.array(las_file.header.max)
#     min_ = np.array(las_file.header.min)
#     mins = np.ones((w, h)) * max_[2]
#     maxes = np.zeros((w, h))
#     counts = np.zeros((w, h))
#     n = las_file.header.count
#     percent = 0
#     for i, point in enumerate(las_file[:500000]):
#         if min_x < point.x < max_x and min_y < point.y < max_y:
#             color = (0, 0, 0)
#             rasterize_point(np.array((point.x, point.y, point.z)), color,
#                             w, h, max_, min_, counts, maxes, mins, colors)
#         if round((i * 100.0) / n, 2) != percent:
#             percent = round((i * 100.0) / n, 2)
#             print('\r{}  %'.format(percent), end='')
#     return counts, maxes, mins
#
#
# # @numba.jit
# def rasterize_sample():
#     las_file = file.File('srp_scans/sample.las', mode='r')
#
#     w = 512
#     h = 512
#
#     min_x = 150
#     max_x = 250
#     min_y = 200
#     max_y = 300
#
#     counts, maxes, mins = rasterize_las(las_file, w, h, min_x, max_x, min_y, max_y)
#
#     np.save('counts', counts)
#     np.save('maxes', maxes)
#     np.save('mins', mins)


def visualize(image, title='', min_row=0, max_row=0, min_col=0, max_col=0):
    if max_row == 0:
        max_row = image.shape[0]
    if max_col == 0:
        max_col = image.shape[1]

    figure()
    plt.imshow(image[min_row:max_row,min_col:max_col], interpolation='none')
    plt.title(title)
    gray()
    colorbar()
    show()

def las_to_np_array(folder_name, file_name, min_x=0, max_x=0, min_y=0, max_y=0, ratio=1, visualize=False, rasterize=False, tile_size=512):
    las_file = file.File(os.path.join(folder_name, file_name), mode='r')
    las_maxs = np.array(las_file.header.max)
    las_mins = np.array(las_file.header.min)
    las_ranges = las_maxs - las_mins + [1, 1, 1]
    point_count = las_file.header.count

    if max_x == 0:
        max_x = las_ranges[0]
    if max_y == 0:
        max_y = las_ranges[1]
    range_x = max_x - min_x
    range_y = max_y - min_y

    min_array = np.zeros([range_y * ratio, range_x * ratio])
    max_array = np.zeros([range_y * ratio, range_x * ratio])
    mean_array = np.zeros([range_y * ratio, range_x * ratio])
    count_array = np.zeros([range_y * ratio, range_x * ratio])
    min_distance_array = np.zeros([range_y * ratio, range_x * ratio])

    percent = 0
    for i, point in enumerate(las_file):
        x = point.x - las_mins[0]
        y = point.y - las_mins[1]
        z = point.z - las_mins[2]

        if min_x < x < max_x and min_y < y < max_y:
            x = floor((x - min_x) * ratio)
            y = floor((y - min_y) * ratio)

            count_array[y][x] += 1

            if min_array[y][x] == 0 or z < min_array[y][x]:
                min_array[y][x] = z

            if max_array[y][x] == 0 or z > max_array[y][x]:
                max_array[y][x] = z

            mean_array[y][x] = z * (1 / count_array[y][x]) + mean_array[y][x] * (1 - 1 / count_array[y][x])

        if round((i * 100.0) / point_count, 2) != percent:
            percent = round((i * 100.0) / point_count, 2)
            print('\r{}  %'.format(percent), end='')

    difference_array = max_array - min_array

    # # Threshold adjust
    # difference_array_adjusted = np.array(difference_array)
    # count_high_cutoff = 2.5
    # count_low_cutoff = .5
    # for i, value in ndenumerate(difference_array):
    #     if value < count_low_cutoff or value > count_high_cutoff:
    #         difference_array_adjusted[i] = 0
    # np.save('output/sample_diff_adjusted', difference_array_adjusted)

    # np.save(os.path.join(output_dir, os.path.splitext(file_name)[0] + '_min'), min_array)
    # np.save(os.path.join(output_dir, os.path.splitext(file_name)[0] + '_max'), max_array)
    # np.save(os.path.join(output_dir, os.path.splitext(file_name)[0] + '_mean'), mean_array)
    # # np.save(os.path.join(output_dir, os.path.splitext(file_name)[0] + '_variance'), variance_array)
    # np.save(os.path.join(output_dir, os.path.splitext(file_name)[0] + '_count'), count_array)
    # np.save(os.path.join(output_dir, os.path.splitext(file_name)[0] + '_min_distance'), min_distance_array)
    np.save(os.path.join(output_dir, os.path.splitext(file_name)[0] + '_difference'), difference_array)

    if rasterize:
        las_x_scale = las_ranges[1] / difference_array.shape[0]
        las_y_scale = las_ranges[0] / difference_array.shape[1]
        np_array_to_geoimage(
            difference_array, las_mins[0], las_mins[1], las_y_scale, las_x_scale, 26949,
            os.path.join(output_dir, os.path.splitext(file_name)[0] + '_difference.tiff'))

    if visualize:
        visualize(difference_array, "Data Written to GeoTiff")

def np_array_to_geoimage(np_array, x, y, x_scale, y_scale, espg, file_name):
    driver = gdal.GetDriverByName("GTiff")
    assert isinstance(driver, gdal.Driver)

    data_set = driver.Create(file_name, np_array.shape[1], np_array.shape[0], 1, gdal.GDT_Float32)
    assert isinstance(data_set, gdal.Dataset)

    spatial_reference = osr.SpatialReference()
    assert isinstance(spatial_reference, osr.SpatialReference)
    spatial_reference.ImportFromEPSG(espg)

    data_set.SetProjection(spatial_reference.ExportToWkt())
    data_set.SetGeoTransform([x, x_scale, 0,
                              y, 0, y_scale])
    band = data_set.GetRasterBand(1)
    assert isinstance(band, gdal.Band)

    band.WriteArray(np_array)

def process_lidar(folder_name, file_name, visualize=False, rasterize=False, overwrite=False):
    # print('\nProcessing file:', file_name)

    file_exists = False
    for output_file in os.listdir(output_dir):
        if os.path.splitext(file_name)[0] in output_file:
            file_exists = True

    if not file_exists or overwrite:
        las_to_np_array(folder_name, file_name, ratio=4, visualize=visualize, rasterize=rasterize)
    else:
        print('Existing file found and not overwritten: ', file_name)

    # print('\nFinished processing file:', file_name)


def process_all_lidar(dir):
    folder_name = os.path.basename(dir)
    file_names = os.listdir(dir)
    for file_name in file_names:
        file_type = os.path.splitext(file_name)[1]
        if file_type == '.las':
            try:
                process_lidar(dir, file_name, False, True)
            except MemoryError:
                print('Memory error while processing file: ', file_name)

def find_matching_shapes():
    input_image = np.load('D:\SRP\object_detect_output\Lt._20130815(0)_LAS1_2_difference.npy')

    mask = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    input_image = input_image[500:1000, 1000:1500][100:200, 250:350]

    input_image *= (1/input_image.max())

    visualize(input_image, 'input_image')

    rand_width = 10
    rand_height = 10
    rand_x = 0
    rand_y = 0
    rand_rotation = 0
    # mask = np.zeros([rand_width, rand_height])
    # while np.sum(mask) < rand_width * rand_height / 10:
    #     rand_rotation = 0 #np.random.rand() * 360
    #     rand_x = round(np.random.rand() * (input_image.shape[0] - rand_width)) - 1
    #     rand_y = round(np.random.rand() * (input_image.shape[1] - rand_height)) - 1
    #     mask = transform.rotate(input_image, rand_rotation, order=0, resize=False, preserve_range=True)[rand_x:rand_x + rand_width, rand_y:rand_y + rand_height]

    mask[mask > 0] = [1 for i in range(0, mask[mask > 0].size)]

    visualize(mask, 'mask')

    mask_negatives = mask[mask == 0].size
    mask_positives = mask[mask == 1].size

    best_match = []
    best_match_rating = -1

    top_ten = np.ndarray(shape=(10, 5), dtype=float)

    max_r = 12
    for r in range(0, max_r):
        rotation = 360 / max_r * r
        rotated_mask = transform.rotate(mask, rotation, order=0, resize=False, preserve_range=True)

        max_x = input_image.shape[0] - rotated_mask.shape[0] + 1
        max_y = input_image.shape[1] - rotated_mask.shape[1] + 1

        for x in range(0, max_x):
            for y in range(0, max_y):
                cropped_input = input_image[x:x+rotated_mask.shape[0], y:y+rotated_mask.shape[1]]

                if np.sum(cropped_input) > 0:
                    positive_match = np.copy(cropped_input)
                    negative_match = np.copy(cropped_input)
                    positive_match[rotated_mask == 0] = rotated_mask[rotated_mask == 0]
                    negative_match[rotated_mask == 1] = [0 for i in range(0, len(cropped_input[rotated_mask == 1]))]

                    rating = (np.sum(positive_match) / mask_positives - np.sum(negative_match) / mask_negatives) / cropped_input.size
                    if rating > .0005:
                        print('\nMatch: ', x, y, rotation, round(rating * 100, 2), '%')
                        visualize(rotated_mask, 'rotated_mask')
                        visualize(positive_match, 'positive_match')
                        visualize(negative_match, 'negative_match')
                        visualize(cropped_input, 'cropped_input')

                    if rating > best_match_rating:
                        best_match = [x, y, rotation, cropped_input]
                        best_match_rating = rating

                    # if top_ten.size < 10:
                    #     np.append(top_ten, [x, y, rotation, cropped_input, rating])
                    # else:
                    #     print(10)

            iteration = r * (max_x * max_y) + ((x + 1) * max_x + (y + 1))
            max_iterations = (max_r + 1) * max_x * max_y
            percent = round(float(iteration) / max_iterations * 100.0, 2)
            print('\r{}  %'.format(percent), end='')

    best_x, best_y, best_rotation, best_match_image = best_match

    # input_image[best_x:best_match_image.shape[0], best_y:best_match_image.shape[1]]

    print('\nBest Match: ', best_x, best_y, best_rotation, round(best_match_rating * 100, 2), '%')

    print('\nGoal Match:', rand_x, rand_y, rand_rotation)

    visualize(best_match_image, 'Best Match')

    visualize(mask, 'Goal Match')


def convolve():
    input_image = np.load('D:\SRP\object_detect_output\Lt._20130815(0)_LAS1_2_difference.npy')

    blend_rotations = 0
    mask = masks.balance(masks.blend_rotation(masks.line, blend_rotations))

    input_image = input_image[500:1000, 1000:1500][100:200, 250:350]  # [20:60, 25:65]
    mask_part = transform.rotate(masks.square_boarder_half, randint(0, 360), order=0, resize=False, preserve_range=True)
    mask_part *= np.max(input_image) / 2
    rand_index = [randint(0, input_image.shape[0] - mask_part.shape[0]),
                  randint(0, input_image.shape[1] - mask_part.shape[1])]
    input_image[rand_index[0]:rand_index[0] + mask_part.shape[0],
                rand_index[1]:rand_index[1] + mask_part.shape[1]] += mask_part

    input_image *= (1/input_image.max())

    max_r = 360

    convolutions = ndarray((max_r, input_image.shape[0], input_image.shape[1]))
    for r in range(max_r):
        rotation = 360 / max_r * r
        rotated_mask = transform.rotate(mask, rotation, order=0, resize=False, preserve_range=True)

        convolution = match_template(input_image, rotated_mask, pad_input=True)
        convolutions[r, :, :] = convolution

        percent = round(float(r) / (max_r - 1) * 100.0, 2)
        print('\r{}  %'.format(percent), end='')

    angles = convolutions.argmax(0)
    values = convolutions.max(0)
    points = peak_local_max(values, min_distance=mask.shape[0] / 2, exclude_border=False)

    figure()

    subplot(321)
    title('input')
    imshow(input_image, interpolation='none')
    colorbar()
    gray()


    subplot(322)
    title('mask')
    imshow(mask, interpolation='none')
    colorbar()
    gray()

    subplot(323)
    title('angles')
    imshow(angles, interpolation='none')
    colorbar()
    jet()

    subplot(324)
    title('maxima')
    imshow(values, interpolation='none')
    colorbar()
    gray()
    scatter(points[:, 1], points[:, 0], c='blue')

    print(values[points[:,0], points[:,1]])

    subplot(325)
    title('maxima hist')
    hist(values.flatten(), bins=100)

    lines = np.array(values)
    lines[lines < np.max(lines) / 2] = 0

    subplot(326)
    title('lines')
    imshow(lines, interpolation='none')
    colorbar()
    gray()

    show()

    print('\nFinished!')


if __name__ == "__main__":
    # process_all_lidar(os.path.join(os.getcwd(), 'srp_scans'))
    # process_all_lidar('H:\SRP Scan data\AZ_SRP_Section11_Seiler\Export\Section11_LAS12_AZC_m')
    # find_matching_shapes()
    convolve()


    # # np_array = las_to_np_array('srp_scans/sample.las', 150, 250, 200, 300, 4)
    # # np_array = las_to_np_array('srp_scans/sample.las', ratio=4)
    # # np.save('sample', np_array)
    #
    # # sample = np.load('sample' + '.npy')
    # # max_value = sample.max()
    # # max_cutoff_ratio = 50
    # # sample_size = sample.size
    # # for i, value in ndenumerate(sample):
    # #     if value > max_value / max_cutoff_ratio:
    # #         sample[i] = max_value / max_cutoff_ratio
    # # np.save('sample_max_adjust', sample)
    # # las_image = np.load('sample_max_adjust' + '.npy')
    # # visualize(las_image)
    #
    # data_min = np.load('output/sample_min' + '.npy')
    # data_max = np.load('output/sample_max' + '.npy')
    # data_mean = np.load('output/sample_mean' + '.npy')
    # data_variance = np.load('output/sample_variance' + '.npy')
    # data_count = np.load('output/sample_count' + '.npy')
    #
    # # Threshold adjust
    # data_diff = data_max - data_min
    # data_diff_adjusted = np.array(data_diff)
    # count_high_cutoff = 2.5
    # count_low_cutoff = .5
    # for i, value in ndenumerate(data_diff):
    #     if value < count_low_cutoff or value > count_high_cutoff:
    #         data_diff_adjusted[i] = 0
    #
    # np.save('output/sample_diff_adjusted', data_diff_adjusted)
    #
    # visualize(data_min, "min", 650, 900, 1700, 2000)
    # visualize(data_max, "max", 650, 900, 1700, 2000)
    # visualize(data_mean, "mean", 650, 900, 1700, 2000)
    # visualize(data_variance, "variance", 650, 900, 1700, 2000)
    # visualize(data_count, "count", 650, 900, 1700, 2000)
    # visualize(data_diff, "max - min", 650, 900, 1700, 2000)
    # visualize(data_diff_adjusted, "max - min adjusted", 650, 900, 1700, 2000)
    #
    # # las_image = np.load('output/sample_diff' + '.npy')
    # # rotated_las_image = np.zeros((las_image.shape[1] / 4, las_image.shape[0] / 4))
    # # for x, row in enumerate(las_image):
    # #     if x % 4 == 0:
    # #         for y, cell in enumerate(row):
    # #             if y % 4 == 0:
    # #                 rotated_las_image[y / 4][x / 4] = cell
    # #
    # # np.save('output/temp', rotated_las_image)
    #
    # las_file = file.File('srp_scans/sample.las', mode='r')
    # las_image = data_diff_adjusted
    # las_maxs = np.array(las_file.header.max)
    # las_mins = np.array(las_file.header.min)
    # las_ranges = las_maxs - las_mins + [1, 1, 1]
    # las_x_scale = las_ranges[1] / las_image.shape[0]
    # las_y_scale = las_ranges[0] / las_image.shape[1]
    # np_array_to_geoimage(las_image, las_mins[0], las_mins[1], las_y_scale, las_x_scale, 26949, "output/sample_diff_adjusted.tiff")
    # visualize(las_image, "Data Written to GeoTiff")