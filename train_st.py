"""
$ python3 experiments/st/train_st.py \
    --batch_size 1 \
    --num_epochs 5 \
    --run_name score_cam_debug \
    --device cpu 

"""

import os
import sys
import copy
import glob
import math
import argparse
from functools import partial

import torch
import mlflow
import numpy as np
from torchvision.transforms import PILToTensor, ConvertImageDtype, Compose, Resize
from torchvision.models import (
    ResNet50_Weights,
    resnet50,
    efficientnet_b4,
    vit_b_16,
    ViT_B_16_Weights,
    EfficientNet_B4_Weights,
)
from sklearn.model_selection import train_test_split, StratifiedKFold

ST_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(ST_DIR)
ROOT_DIR = os.path.dirname(EXP_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")

sys.path.insert(0, ROOT_DIR)
from experiments.st._paths import DataPaths
from experiments.st.data_model import STImages
from experiments.st.datasets import STDataSet
from core.trainer import BinaryClassifierTrainer
from core.metics import (
    plot_cv_auroc,
    plot_cv_prauc,
    plot_auroc,
    plot_prauc,
    plot_confusion_matrix,
)
from mlflow_settings import get_expid_client, TRACKING_URI, EXP_S_T, save_plot_and_clear
from utils.log_ops import get_logger
from utils.io_ops import save_pickle


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument(
        "--model_name", type=str, choices=["vit-b-16", "resnet50"], required=True
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def model_build(model_name) -> torch.nn.Module:
    if model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=1)

    elif model_name == "vit-b-16":
        model = vit_b_16(ViT_B_16_Weights.DEFAULT)
        model.heads = torch.nn.Linear(in_features=768, out_features=1)
    # model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    # model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=1)
    # model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
    # model.classifier[1] = torch.nn.Linear(in_features=1792, out_features=1)

    return model


def cal_classweight(train_st_image: STImages):
    count = len(train_st_image.labels)
    n_texture = sum(train_st_image.labels)
    n_smooth = count - n_texture
    return torch.tensor(n_texture / n_smooth)


if __name__ == "__main__":
    ARGS = get_args()
    LOGGER = get_logger("train_st")
    for arg_name, arg_value in vars(ARGS).items():
        LOGGER.info(f"{arg_name}: {arg_value}")
    LOGGER.info(f"ROOT_DIR({ROOT_DIR})")

    torch.manual_seed(ARGS.seed)
    torch.multiprocessing.set_start_method("spawn")

    data_path = DataPaths(DATA_DIR)
    s_image_paths = glob.glob(str(data_path.canon.s / "*"))
    t_image_paths = glob.glob(str(data_path.canon.t / "*"))
    LOGGER.info(f"S Image (n={len(s_image_paths)}) load")
    LOGGER.info(f"T Image (n={len(t_image_paths)}) load")
    st_images = STImages(
        s_image_paths=s_image_paths,
        t_image_paths=t_image_paths,
    )
    transform = Compose(
        [PILToTensor(), Resize((512, 512)), ConvertImageDtype(torch.float)]
    )

    exp_id, mlflow_client = get_expid_client(TRACKING_URI, EXP_S_T)

    with mlflow.start_run(experiment_id=exp_id, run_name=ARGS.run_name):
        mlflow.log_params(vars(ARGS))
        mlflow.log_artifact(os.path.abspath(__file__))

        stratified_kfold = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=ARGS.seed
        )

        cv_y_trues = list()
        cv_y_probs = list()
        for fold, (train_indices, test_indices) in enumerate(
            stratified_kfold.split(st_images.image_paths, st_images.labels), start=1
        ):
            LOGGER.info(f"Start fold {fold}")
            with mlflow.start_run(
                experiment_id=exp_id, run_name=ARGS.run_name, nested=True
            ):
                interim_train_images = st_images[train_indices]
                LOGGER.info(
                    f"Train+Val S(n={interim_train_images.s_image_count}), "
                    f"T(n={interim_train_images.t_image_count})"
                )

                (
                    train_image_paths,
                    val_image_paths,
                    train_image_labels,
                    val_image_labels,
                ) = train_test_split(
                    interim_train_images.image_paths,
                    interim_train_images.labels,
                    stratify=interim_train_images.labels,
                )
                train_images = STImages(
                    image_paths=train_image_paths, labels=train_image_labels
                )
                val_images = STImages(
                    image_paths=val_image_paths, labels=val_image_labels
                )
                test_images = st_images[test_indices]
                LOGGER.info(
                    f"Train image number: {len(train_images)}, S(n={train_images.s_image_count}), T(n={train_images.t_image_count})"
                )
                LOGGER.info(
                    f"Val image number: {len(val_images)}, S(n={val_images.s_image_count}), T(n={val_images.t_image_count})"
                )
                LOGGER.info(
                    f"Test image number: {len(test_images)}, S(n={test_images.s_image_count}), T(n={test_images.t_image_count})"
                )

                train_st_dataset = STDataSet(
                    train_images.image_paths,
                    train_images.labels,
                    transform,
                    device=ARGS.device,
                )
                train_loader = torch.utils.data.DataLoader(
                    train_st_dataset,
                    batch_size=ARGS.batch_size,
                    shuffle=True,
                    num_workers=ARGS.num_workers,
                )
                val_loader = torch.utils.data.DataLoader(
                    STDataSet(
                        val_images.image_paths,
                        val_images.labels,
                        transform,
                        device=ARGS.device,
                    ),
                    batch_size=ARGS.batch_size,
                    shuffle=True,
                    num_workers=ARGS.num_workers,
                )
                test_loader = torch.utils.data.DataLoader(
                    STDataSet(
                        test_images.image_paths,
                        test_images.labels,
                        transform,
                        device=ARGS.device,
                    ),
                    batch_size=ARGS.batch_size,
                    shuffle=True,
                    num_workers=ARGS.num_workers,
                )

                model = model_build(ARGS.model_name)
                weighted_bce = partial(
                    torch.nn.functional.binary_cross_entropy_with_logits,
                    weight=torch.tensor(
                        train_images.s_image_count / train_images.t_image_count
                    ),
                )
                dd_model = torch.nn.DataParallel(model).to(ARGS.device)
                trainer = BinaryClassifierTrainer(
                    model=dd_model,
                    loss=weighted_bce,
                    optimizer=torch.optim.Adam(dd_model.parameters()),
                    logger=LOGGER,
                )

                best_loss = math.inf
                n_paitence = 0
                for epoch in range(1, ARGS.num_epochs + 1):
                    train_loss_meter, train_metrics_meter = trainer.run_epoch(
                        "train", epoch, train_loader
                    )
                    mlflow.log_metric("train_loss", train_loss_meter.avg, step=epoch)
                    mlflow.log_metrics(train_metrics_meter.to_dict(), step=epoch)
                    val_loss_meter, val_metrics_meter = trainer.run_epoch(
                        "val", epoch, train_loader
                    )
                    mlflow.log_metric("val_loss", val_loss_meter.avg, step=epoch)
                    mlflow.log_metrics(val_metrics_meter.to_dict(), step=epoch)

                    if best_loss > val_loss_meter.avg:
                        best_loss = val_loss_meter.avg
                        best_weight = copy.deepcopy(model.state_dict())
                        LOGGER.info(f"epoch:{epoch}, Best loss: {best_loss}")

                    else:
                        if n_paitence == ARGS.max_patience:
                            break

                        n_paitence += 1

                model.load_state_dict(best_weight)
                test_loss_meter, test_metrics_meter = trainer.run_epoch(
                    "test", epoch, test_loader
                )
                fold_test_labels = np.array(test_metrics_meter.labels)
                fold_test_probs = np.array(test_metrics_meter.probs)
                cv_y_trues.append(fold_test_labels)
                cv_y_probs.append(fold_test_probs)
                mlflow.log_metric("test_loss", test_loss_meter.avg, step=epoch)
                mlflow.log_metrics(test_metrics_meter.to_dict(), step=epoch)
                mlflow.pytorch.log_model(model, "model")

                save_pickle(test_metrics_meter, "test_metrics_meter.pkl")
                mlflow.log_artifact("test_metrics_meter.pkl")
                os.remove("test_metrics_meter.pkl")

                fig, axes = plot_auroc(fold_test_labels, fold_test_probs)
                save_plot_and_clear("auroc.png")
                fig, axes = plot_prauc(fold_test_labels, fold_test_probs)
                save_plot_and_clear("prauc.png")
                fig, axes = plot_confusion_matrix(
                    fold_test_labels,
                    fold_test_probs >= 0.5,
                    labels=["smooth", "texture"],
                )
                save_plot_and_clear("confusion_matrix.png")

        plot_cv_auroc(cv_y_trues, cv_y_probs)
        save_plot_and_clear("cv_auroc.png")
        plot_cv_prauc(cv_y_trues, cv_y_probs)
        save_plot_and_clear("cv_prauc.png")
