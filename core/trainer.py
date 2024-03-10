import os
import sys
import math
import logging
from typing import Tuple, Dict
from abc import ABC, abstractmethod

import numpy as np
import torch
from progress.bar import Bar
from sklearn.metrics import roc_auc_score

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CORE_DIR)
sys.path.append(ROOT_DIR)
from core.metics import AverageMeter, Metrics, calculate_metrics, MetricsMeter


class BaseTrainer(ABC):
    @abstractmethod
    def make_bar_sentence(self):
        pass

    @abstractmethod
    def run_epoch(self):
        pass

    def get_accuracy(
        self, logit: torch.Tensor, labels: torch.Tensor, threshold: int = 0.5
    ) -> float:
        confidence = torch.sigmoid(logit).flatten()
        pred_labels = (confidence > threshold).float().flatten()
        return (pred_labels == labels).sum().item() / len(labels)

    def get_auroc(self, logit: torch.Tensor, labels: torch.Tensor) -> float:
        confidence = torch.sigmoid(logit).flatten()
        return roc_auc_score(labels.flatten(), confidence)


class BinaryClassifierTrainer(ABC):
    def __init__(
        self,
        model: torch.nn.modules.Module,
        loss: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer = None,
        logger: logging.Logger = None,
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.logger = (
            logging.Logger("BinaryClassifierTrainer") if logger is None else logger
        )

    def make_bar_sentence(
        self,
        phase: str,
        epoch: int,
        total_step: int,
        step: int,
        eta: str,
        total_loss: float,
        accuracy: float,
        auroc: float,
        prauc: float,
    ) -> str:
        """ProgressBar의 stdout의 string을 생성하여 반환

        Args:
            phase (str): Epoch의 phase
            epoch (int): epoch
            total_step (int): total steps for one epoch
            step (int): Step (in a epoch)
            eta (str): Estimated Time of Arrival
            loss (float): loss
            accuracy (float): accuracy
            auroc (float): auroc
            prauc (float): prauc

        Returns:
            str: progressbar senetence

        """
        total_loss = round(total_loss, 5)
        accuracy = round(accuracy, 5)
        auroc = round(auroc, 5)
        prauc = round(prauc, 5)

        return (
            f"{phase} | EPOCH {epoch}: [{step}/{total_step}] | "
            f"eta:{eta} | total_loss: {total_loss} | "
            f"accuracy: {accuracy} | auroc: {auroc} | prauc: {prauc}"
        )

    def run_epoch(
        self,
        phase: str,
        epoch: int,
        dataloader: torch.utils.data.DataLoader,
        threshold: float = 0.5,
    ) -> Tuple[float, float]:
        """1회 Epoch을 각 페이즈(train, validation)에 따라서 학습하거나 손실값을
        반환함.

        Note:
            - 1 epoch = Dataset의 전체를 학습한경우
            - 1 step = epoch을 하기위해 더 작은 단위(batch)로 학습할 떄의 단위

        Args:
            phase (str): training or validation
            epoch (int): epoch
            dataloader (torch.utils.data.DataLoader): dataset (train or validation)

        Returns:
            Tuple: loss, accuracy, top_k_recall
        """
        total_step = len(dataloader)
        bar = Bar(max=total_step, check_tty=False)

        loss_meter = AverageMeter("loss")
        metrics_meter = MetricsMeter(name=phase, accuracy_threshold=threshold)
        for step, batch in enumerate(dataloader):
            xs, ys = batch

            if phase == "train":
                self.model.train()
                logits = self.model(xs)
            else:
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(xs)

            loss = self.loss(logits, ys)

            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # metric
            loss_meter.update(loss.item(), len(ys))
            flatten_ys = ys.flatten()
            model_confidence = torch.sigmoid(logits).flatten()
            metrics_meter.update(
                model_confidence.detach().cpu().numpy().tolist(),
                flatten_ys.cpu().numpy().tolist(),
            )
            self.logger.debug(
                f"Step({step}): \n"
                f"logits({str(logits)}) \n"
                f"labels({str(flatten_ys)})"
            )
            self.logger.debug(
                f"Step ({step}): Accuracy({metrics_meter.accuracy}), "
                f"AUROC({metrics_meter.auroc}), PRAUC({metrics_meter.prauc})"
            )

            bar.suffix = self.make_bar_sentence(
                phase=phase,
                epoch=epoch,
                step=step,
                total_step=total_step,
                eta=bar.eta,
                total_loss=loss_meter.avg,
                accuracy=metrics_meter.accuracy,
                auroc=metrics_meter.auroc,
                prauc=metrics_meter.prauc,
            )
            bar.next()

        bar.finish()

        return (loss_meter, metrics_meter)


class Trainer:
    def __init__(
        self,
        model: torch.nn.modules.Module,
        loss: callable,
        optimizer: torch.optim.Optimizer,
        device: str,
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.y_tures = list()
        self.y_probs = list()

    def get_best_weight(self):
        return (
            self.best_weight
            if hasattr(self, "best_weight")
            else self.model.state_dict()
        )

    def make_bar_sentence(
        self,
        phase: str,
        epoch: int,
        total_step: int,
        step: int,
        eta: str,
        total_loss: float,
        metrics: Dict[str, AverageMeter],
    ) -> str:
        """ProgressBar의 stdout의 string을 생성하여 반환

        Args:
            phase (str): Epoch의 phase
            epoch (int): epoch
            total_step (int): total steps for one epoch
            step (int): Step (in a epoch)
            eta (str): Estimated Time of Arrival
            loss (float): loss
            acc (float): accuracy
            metrics (dict): 그외 metrics

        Returns:
            str: progressbar senetence

        """

        total_loss = round(total_loss, 5)

        additional_measures = list()
        return_string = (
            f"{phase} | EPOCH {epoch}: [{step}/{total_step}] | "
            f"eta:{eta} | total_loss: {total_loss} "
        )
        if not metrics:
            return return_string

        for name, value in metrics.items():
            additional_measures.append(f"{name}: {value}")

        return_string += "| ".join(additional_measures)

        return return_string

    def run_epoch(
        self,
        phase: str,
        epoch: int,
        dataloader: torch.utils.data.DataLoader,
    ) -> Tuple[float, float, float]:
        """1회 Epoch을 각 페이즈(train, validation)에 따라서 학습하거나 손실값을
        반환함.

        Note:
            - 1 epoch = Dataset의 전체를 학습한경우
            - 1 step = epoch을 하기위해 더 작은 단위(batch)로 학습할 떄의 단위

        Args:
            phase (str): training or validation
            epoch (int): epoch
            dataloader (torch.utils.data.DataLoader): dataset (train or validation)

        Returns:
            Tuple: loss, accuracy, AUC
        """
        total_step = len(dataloader)
        bar = Bar(max=total_step)

        self.y_tures.clear()
        self.y_probs.clear()

        metrics = Metrics()
        for step, batch in enumerate(dataloader):
            images, labels = batch
            labels = torch.ravel(labels)
            if phase == "train":
                self.model.train()
            else:
                self.model.eval()

            logits = self.model(images).ravel()  # (N, )
            empirical_loss = self.loss(logits, labels)
            y_prob = torch.sigmoid(logits)

            self.y_tures.extend(labels.ravel().tolist())
            self.y_probs.extend(y_prob.ravel().tolist())

            if phase == "train":
                self.optimizer.zero_grad()
                empirical_loss.backward()
                self.optimizer.step()

            metric_values: dict = calculate_metrics(
                np.array(self.y_tures), np.array(self.y_probs)
            )
            metric_values.update({"loss": empirical_loss.item() * len(images)})
            metrics.update(len(images), metric_values)
            bar.suffix = self.make_bar_sentence(
                phase=phase,
                epoch=epoch,
                step=step,
                total_step=total_step,
                eta=bar.eta,
                total_loss=metrics.loss.avg,
                metrics=metrics.to_dict(),
            )
            bar.next()

        bar.finish()

        return metrics

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        n_epochs: int = 100,
        patience_limit: int = 10,
        run_id: str = str(),
        mlflow_client=None,
        checkpoint_dir=None,
    ):
        n_patience = 0

        os.makedirs(checkpoint_dir, exist_ok=True)

        min_loss = math.inf
        for epoch in range(1, n_epochs):
            train_metrics = self.run_epoch(
                "train",
                epoch=epoch,
                dataloader=train_loader,
            )

            if mlflow_client:
                for k, v in train_metrics.to_dict(prefix="train_").items():
                    mlflow_client.log_metric(run_id=run_id, key=k, value=v, step=epoch)

            with torch.no_grad():
                val_metrics = self.run_epoch(
                    "val",
                    epoch=epoch,
                    dataloader=val_loader,
                )
                if mlflow_client:
                    for k, v in val_metrics.to_dict(prefix="val_").items():
                        mlflow_client.log_metric(
                            run_id=run_id, key=k, value=v, step=epoch
                        )

            if min_loss > val_metrics.loss.avg:
                min_loss = val_metrics.loss.avg
                n_patience = 0
                if checkpoint_dir is None:
                    torch.save(self.model.state_dict(), "best_weight.pt")
                else:
                    fname = os.path.join(
                        checkpoint_dir,
                        f"epoch_{epoch}.pt",
                    )

                    torch.save(
                        self.model.state_dict(),
                        fname,
                    )
                    print(f"{fname} saved")

                    torch.save(
                        self.model.state_dict(),
                        os.path.join(checkpoint_dir, "best_weight.pt"),
                    )

            else:
                n_patience += 1

            if n_patience == patience_limit:
                break
