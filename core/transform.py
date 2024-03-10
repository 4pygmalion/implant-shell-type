from typing import Tuple, List, Tuple

import cv2
import numpy as np
from PIL import Image

import torch


def normalize(x: np.ndarray) -> np.ndarray:
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))


def find_largest_image(
    image: np.ndarray, contours: Tuple[np.ndarray]
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """이미지와 윤곽선 정보가 주어졌을 때, 가장 큰 영역을 갖는 이미지를 잘라서 반환

    매개변수:
        image (np.ndarray): 이미지 배열. OpenCV 또는 NumPy로 표현된 이미지.
        contours (Tuple[np.ndarray]): 윤곽선 정보를 담은 튜플. 각 윤곽선은 NumPy 배열로 표현

    반환값:
        np.ndarray: 가장 큰 영역을 잘라낸 이미지 배열.

    예시:
        >>> image = cv2.imread('path_to_image.jpg')
        >>> gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        >>> _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        >>> contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        >>> largest_image = find_largest_image(image, contours)
    """

    x_min = 0
    y_min = 0

    max_size = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        size = w * h

        if max_size <= size:
            max_size = size
            max_image = image[
                y : y + h,
                x : x + w,
            ]

            x_min = x
            y_min = y

    return max_image, (y_min, x_min)


def crop_usimage(
    image: np.ndarray, size: tuple = tuple(), margin_px: int = 10
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    이미지를 전처리하는 함수
    이 함수는 주어진 이미지를 이진화하여 윤곽선을 찾고, 가장 큰 영역을 갖는 이미지를 잘라내어 반환

    매개변수:
        image (np.ndarray): 전처리할 이미지 배열. OpenCV 또는 NumPy로 표현된 이미지.
        size (tuple, optional): 잘라낸 이미지를 리사이징할 크기.
            (가로, 세로) 형식의 튜플. 기본값은 리사이징을 수행하지 않음
        margin_px (int, optional): 잘라낸 이미지의 가장자리 여백 크기. 기본값은 10px

    반환값:
        Tuple
            1) np.ndarray: 전처리된 이미지 배열. 잘라낸 이미지 또는 리사이징된 이미지가 반환
            2) Tuple[int, int]: 오리지널 이미지의 최대크기의 이미지의 좌상단

    예시:
        >>> image = cv2.imread('path_to_image.jpg')
        >>> preprocessed_image = crop_usimage(image)
        >>> cv2.imshow('Preprocessed Image', preprocessed_image)
        >>> cv2.waitKey(0)
        >>> cv2.destroyAllWindows()

    """
    ret, binary_image = cv2.threshold(
        image, thresh=1, maxval=255, type=cv2.THRESH_BINARY
    )
    contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # contours: Tuple[np.ndarray]

    crop_image, (y_min, x_min) = find_largest_image(image, contours)
    crop_image = crop_image[margin_px:-margin_px, margin_px:-margin_px]
    if not size:
        return crop_image, (y_min, x_min)

    return np.array(Image.fromarray(crop_image).resize(size)), (y_min, x_min)


def crop_ratio(
    image_array, upper_margin: float = 0.12, lower_margin: float = 0.1, is_st=False
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    이미지의 상하의 일정 비율을 잘라냅니다.

    Note:
        09.08에 배포했던 모델은 0.15, 0.1 (%)의 마진을 두고 크로핑했었음.

        10.07에 위쪽 비율을 0.12로 수정. ST의 경우, 좌우 비율은 0.7(%)로 롭롭

    Args:
        image_array (numpy.ndarray): 처리할 넘파이 배열

    Returns:
        numpy.ndarray: 잘라낸 넘파이 배열
    """

    height, width, c = image_array.shape

    upper_margin_idx = int(height * upper_margin)
    lower_margin_idx = int(height * lower_margin)

    if is_st:
        left_margin = 0.07
        right_margin = 0.07
        left_interval = int(width * left_margin)
        right_interval = int(width * right_margin)

        cropped_np_array = image_array[
            upper_margin_idx:-lower_margin_idx, left_interval:-right_interval, :
        ]

        return cropped_np_array, (upper_margin_idx, left_interval)

    else:
        cropped_np_array = image_array[upper_margin_idx:-lower_margin_idx, :]

        return cropped_np_array, (upper_margin_idx, 0)


def find_margin(
    image_array: np.ndarray, threshold_value: int, return_ratio=False
) -> (int, int):
    if image_array.ndim != 2:
        ValueError("image array must be 2 dim array")

    height, width = image_array.shape
    mid_x = int(width / 2)

    ret, binary_image = cv2.threshold(
        image_array, threshold_value, 255, cv2.THRESH_BINARY
    )

    for i, pix in enumerate(binary_image[:, mid_x]):
        if pix == 0:
            break

    for j, pix in enumerate(binary_image[i:, mid_x]):
        if pix == 0:
            continue
        break

    upper_indice = i + j

    for i, pix in enumerate(binary_image[:, mid_x][::-1]):
        if pix == 0:
            continue

        break

    lower_indice = height - i

    if return_ratio:
        return upper_indice / height, lower_indice / height

    return upper_indice, lower_indice


def transform_image(crop_image):
    crop_image_tensor = np.expand_dims(crop_image, axis=0)
    crop_image_tensor = np.repeat(crop_image_tensor, 3, axis=0)
    crop_image_tensor = torch.from_numpy(crop_image_tensor)

    if crop_image_tensor.ndim == 3:
        crop_image_tensor = crop_image_tensor.unsqueeze(dim=0)

    crop_image_tensor = crop_image_tensor.to("cpu").float()

    return crop_image_tensor
