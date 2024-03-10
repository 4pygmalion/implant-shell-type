from typing import Tuple

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from .base_cam import min_max_scaling


def grad_cam(
    model: torch.nn.Module,
    image: np.ndarray,
    target_layer: torch.nn.Module,
) -> Tuple[np.ndarray, float]:
    """
    Args:
        model (torch.nn.Module): Grad-CAM을 적용할 딥러닝 모델.
        image (np.ndarray): Grad-CAM을 계산할 입력 이미지.
        target_layer (Type[torch.nn.Module]): Grad-CAM을 계산할 대상 레이어.

    Returns:
        Tuple[np.ndarray, float]: Grad-CAM 시각화 결과, model confidence
    """

    def forward_hook(module, input, output):
        grad_cam_data["feature_map"] = output

    def backward_hook(module, grad_input, grad_output):
        grad_cam_data["grad_output"] = grad_output[0]

    grad_cam_data = {}
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(image)  # 모델의 출력값을 계산합니다. y_c에 해당
    model.zero_grad()

    # 가장 예측값이 높은 그레디언트를 계산합니다. output[0,]은 차원을 하나 제거
    output[0, output.argmax()].backward()

    feature_map = grad_cam_data["feature_map"]
    grad_output = grad_cam_data["grad_output"]
    weights = grad_output.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feature_map).sum(1, keepdim=True).squeeze()
    cam = cam.detach().cpu().numpy()

    return cam, torch.sigmoid(output.squeeze()).item()


def postprocess_gcam(gcam: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Grad-CAM 출력을 후처리하여 이미지와 같은 크기로 조절하고 강조된 부분을 더 강하게 표현

    Args:
        gcam (np.ndarray): Grad-CAM 출력 배열.
        size (Tuple[int, int]): 원본 이미지 크기 (높이, 너비).

    Returns:
        np.ndarray: 후처리된 Grad-CAM 이미지.
    """
    gcam = np.clip(gcam, a_min=0, a_max=gcam.max())

    # gcam = gcam / np.max(gcam)  # 정규화합니다.

    # gcam[gcam <= gcam.mean() + gcam.std()] = 0
    gcam = cv2.resize(
        gcam, size, interpolation=cv2.INTER_LINEAR
    )  # 원본 이미지 크기로 조절합니다.

    return min_max_scaling(gcam)


def plot_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    image: np.ndarray,
    target_layer: torch.nn.Module,
    alpha: float = 0.4,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    주어진 모델과 입력에 대해 Grad-CAM을 계산하고 시각화

    Args:
        model (torch.nn.Module): Grad-CAM을 계산할 딥러닝 모델.
        input_tensor (torch.Tensor): 모델의 입력 텐서.
        image (np.ndarray): 원본 이미지.
        target_layer (torch.nn.Module): Grad-CAM을 계산할 대상 레이어.
        alpha (float, optional): 원본 이미지와 Grad-CAM 오버레이의 가중치. 기본값은 0.8.

    Returns:
        Tuple[plt.Figure, np.ndarray]: 시각화된 결과의 Matplotlib Figure 객체와 Grad-CAM 이미지.

    """
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(dim=0)

    gcam, confidence = grad_cam(model, input_tensor, target_layer=target_layer)

    if image.ndim == 2:
        w, h = image.shape
    elif image.ndim == 3:
        w, h, c = image.shape

    heatmap = postprocess_gcam(gcam, (w, h))

    fig, axes = plt.subplots(1, 1, figsize=(7, 7))
    axes.imshow(image)
    axes.imshow(heatmap, cmap="jet", alpha=alpha)
    axes.set_title("Grad-CAM")
    return fig, axes
