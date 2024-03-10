import os
from typing import Tuple

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from PIL import Image

from .posthoc import grad_cam, postprocess_gcam
from .dataset import ImageDataset
from .transform import crop_usimage


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(model, dataset: ImageDataset, return_probs=True) -> Tuple[np.ndarray]:
    y_trues = list()
    y_preds = list()

    for batch in tqdm(dataset):
        x, y = batch

        with torch.no_grad():
            logit = model(x.unsqueeze(0))

        y_trues.append(y.item())
        logit = logit.item()

        if return_probs:
            y_preds.append(sigmoid(logit))
        else:
            y_preds.append(logit)

    return np.array(y_trues), np.array(y_preds)


def plot_confusion_matrix(y_true, y_pred, label_names: list, title=str()):
    """_summary_

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        label_names (list): _description_
        title (_type_, optional): _description_. Defaults to str().

    Example:
        >>> from core.evaluation import forward, plot_confusion_matrix
        >>> y_trues, y_probs = forward(model, test_dataset, return_probs=True)
        >>> y_preds = y_probs >= 0.5
        >>> plot_confusion_matrix(y_trues, y_preds, label_names=["Normal", "Rupture"])
    """
    # 계산할 성능 메트릭을 계산
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # 오차 행렬 생성
    cm = confusion_matrix(y_true, y_pred)

    # 그래프 크기 설정
    plt.figure(figsize=(10, 8))

    # 오차 행렬을 히트맵으로 시각화
    sns.set(font_scale=1.4)  # 폰트 크기 조정
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=label_names,
        yticklabels=label_names,
    )

    # 성능 메트릭 출력
    plt.text(
        -0.1,
        1.05,
        f"Accuracy: {accuracy:.2f}",
        ha="left",
        va="center",
        transform=plt.gca().transAxes,
        fontsize=14,  # 폰트 크기 조정
    )
    plt.text(
        -0.1,
        1.02,
        f"Precision: {precision:.2f}",
        ha="left",
        va="center",
        transform=plt.gca().transAxes,
        fontsize=14,  # 폰트 크기 조정
    )
    plt.text(
        -0.1,
        0.99,
        f"Recall: {recall:.2f}",
        ha="left",
        va="center",
        transform=plt.gca().transAxes,
        fontsize=14,  # 폰트 크기 조정
    )
    plt.text(
        -0.1,
        0.96,
        f"F1 Score: {f1:.2f}",
        ha="left",
        va="center",
        transform=plt.gca().transAxes,
        fontsize=14,  # 폰트 크기 조정
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    if title:
        plt.title(title, fontsize=20, y=1.10)  # 제목
    else:
        plt.title("Confusion matrix", fontsize=14, y=1.05)  # 제목


def plot_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    image: np.ndarray,
    target_layer: torch.nn.Module,
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
    post_processgcam = postprocess_gcam(gcam, image.shape[::-1])

    fig, axes = plt.subplots(1, 2, figsize=(24, 12))

    axes[0].grid(False)
    axes[0].imshow(post_processgcam)
    axes[0].set_title(
        f"Grad-CAM (Posthoc): confidence({round(confidence, 4)})", fontsize=25
    )

    axes[1].grid(False)
    axes[1].imshow(image, cmap="gray")
    axes[1].set_title("Original image", fontsize=25)

    return fig, axes


def save_gradcam_by_dataset(
    model, image_paths, dataset, crop: callable = None, save_dir=None, threshold=0.5
):
    if save_dir:
        os.makedirs(os.path.join(save_dir, "false_positive"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "false_negative"), exist_ok=True)

    for image_path, (x, y) in tqdm(zip(image_paths, dataset)):
        image_name = os.path.basename(image_path)
        image = np.array(Image.open(image_path).convert("L"))

        if crop:
            image = crop(image)
            if isinstance(image, tuple):
                image, (_, _) = image

        fig, axes = plot_gradcam(
            model,
            x,
            image,
            target_layer=model._blocks[-1],
        )

        with torch.no_grad():
            logit = model(x.unsqueeze(0))

        if save_dir:
            proba = torch.sigmoid(logit).item()
            if proba < threshold and y == True:
                plt.savefig(os.path.join(save_dir, "false_negative", f"{image_name}"))

            elif proba >= threshold and y == False:
                plt.savefig(os.path.join(save_dir, "false_positive", f"{image_name}"))

            plt.close(fig)
