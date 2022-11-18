import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import cv2

import time
import os
from geoservices_scraping import ScrapeCoordinates, ScrapePictures
from coordinate_operations import scale_coordinate

if 'Unpacking' not in os.listdir():
    os.mkdir('Unpacking')


def crop_polygon_from_image(img_name: str, img_polygon: Polygon, airport_polygon: Polygon) -> np.array:
    """This method extracts the airport from an image. It is of paramount importance that the polygons were transformed
    to the same coordinate reference system."""
    # OpenCV loads the images as numpy arrays
    img = cv2.imread(img_name)
    # By default they are in the BGR format, which is not the standard anymore we therefore convert it to the nowadays
    # more popular RGB format
    # img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Take the intersection between the img_polygon and the airport_polygon
    intersection = img_polygon.intersection(airport_polygon)

    # Getting the bounds of the img_polygon such that we can shift the intersection such that would we apply the
    # same shift to the img_polygon its lowest x and its highest y values would be at 0.
    img_bounds = img_polygon.bounds
    x_shift, y_shift = - img_bounds[0], - img_bounds[3]
    # Shift the intersection
    intersection_shifted = np.array(intersection.exterior.coords) + (x_shift, y_shift)
    # Since the image loads from top to bottom, we have to flip it
    intersection_flipped = intersection_shifted * (1, -1)

    # Scale the x and y values of the shifted intersection polygon based on the bounds of the img_polygon such that its
    # x_max - x_min and y_max - y_min values span the same number as the ortho image pixel dimensions
    x_multiplier = img.shape[0] / (img_bounds[2] - img_bounds[0])
    y_multiplier = img.shape[1] / (img_bounds[3] - img_bounds[1])
    intersection_scaled = intersection_flipped * (x_multiplier, y_multiplier)

    # Use cv2.fillPoly(orth array, pts=[shifted and scaled intersection polygon array], color=(255, 255, 255)) to create
    # the image
    mask = np.zeros((img.shape[0], img.shape[1]))
    cv2.fillPoly(mask, pts=[np.ceil(intersection_scaled).astype(int)], color=1)
    # Cropping the image to a rectangle that only contains the airport. We have to crop the x and y axis separately,
    # otherwise numpy is overwhelmed. While we have to crop two arrays here, it is still faster than doing it only at
    # the final image, since we are saving time when applying the mask
    row_clipping = mask.sum(axis=1) > 0
    col_clipping = mask.sum(axis=0) > 0
    cropped_image = img[row_clipping, :, :]
    cropped_image = cropped_image[:, col_clipping, :]
    cropped_mask = mask[row_clipping, :]
    cropped_mask = cropped_mask[:, col_clipping]
    # Since the original image has three dimensions (width, height and color) we expand the mask here as well
    mask_3d = np.expand_dims(cropped_mask, axis=2)
    # Apply the mask
    new_image = np.where(mask_3d == 0, 0, cropped_image)
    # Save the resulting array
    cv2.imwrite('new_image0.jp2', new_image)
    return new_image


def mask_image(img_name: str, img_polygon: Polygon, airport_polygon: Polygon, dest_name: str = '',
               file_extension: str = '.jpg') -> np.array:
    if not dest_name:
        dest_name = f"{'_'.join(format(bound, '.2f') for bound in img_polygon.bounds)}&"\
                    f"{img_name.split('.')[0].split('/')[-1]}"
    # OpenCV loads the images as numpy arrays
    img = cv2.imread(img_name)

    # Take the intersection between the img_polygon and the airport_polygon
    intersection = img_polygon.intersection(airport_polygon)

    # Getting the bounds of the img_polygon such that we can shift the intersection such that would we apply the
    # same shift to the img_polygon its lowest x and its highest y xvalues would be at 0.
    img_bounds = img_polygon.bounds
    x_shift, y_shift = - img_bounds[0], - img_bounds[3]
    # Shift the intersection
    intersection_shifted = np.array(intersection.exterior.coords) + (x_shift, y_shift)
    # Since the image loads from top to bottom, we have to flip it
    intersection_flipped = intersection_shifted * (1, -1)

    # Scale the x and y values of the shifted intersection polygon based on the bounds of the img_polygon such that its
    # x_max - x_min and y_max - y_min values span the same number as the ortho image pixel dimensions
    x_multiplier = img.shape[0] / (img_bounds[2] - img_bounds[0])
    y_multiplier = img.shape[1] / (img_bounds[3] - img_bounds[1])
    intersection_scaled = intersection_flipped * (x_multiplier, y_multiplier)

    # Use cv2.fillPoly(orth array, pts=[shifted and scaled intersection polygon array], color=(255, 255, 255)) to create
    # the image
    mask = np.zeros((img.shape[0], img.shape[1]))
    cv2.fillPoly(mask, pts=[np.ceil(intersection_scaled).astype(int)], color=1)
    # Since the original image has three dimensions (width, height and color) we expand the mask here as well
    mask_3d = np.expand_dims(mask, axis=2)
    # Apply the mask
    new_image = np.where(mask_3d == 0, 0, img)
    # Save the resulting array
    cv2.imwrite(f'{dest_name}{file_extension}', new_image)
    return new_image


def crop_image(img_name: str = '', img: np.ndarray = None) -> None:
    """This function crops out all the rows/columns with only 0 value pixels."""
    if img is None:
        img = cv2.imread(img_name)
    # Cropping the image to a rectangle that only contains non-zero rows/columns. We have to crop the x and y axis
    # separately, otherwise numpy is overwhelmed
    row_clipping = img.max(axis=2).max(axis=1) > 0
    col_clipping = img.max(axis=2).max(axis=0) > 0
    cropped_image = img[row_clipping, :, :]
    cropped_image = cropped_image[:, col_clipping, :]
    cv2.imwrite(f'{img_name.split(".")[0]}_cropped.jpg', cropped_image)


