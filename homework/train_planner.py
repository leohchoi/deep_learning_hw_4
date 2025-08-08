"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

print("Time to train")

import argparse
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .datasets import road_dataset
from .metrics import PlannerMetric
from .models import MODEL_FACTORY, save_model


def _choose_transform(model_name: str) -> str:
    """Pick the correct transform pipeline for the chosen model."""
    if model_name in ("mlp_planner", "transformer_planner"):
        # Only need track geometry (no images)
        return "state_only"
    if model_name == "cnn_planner":
        return "default"
    raise ValueError(f"Unknown model '{model_name}' - cannot decide transform pipeline")


def _forward(model_name: str, model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run a forward pass with the right inputs for the selected model."""
    if model_name == "cnn_planner":
        return model(image=batch["image"])
    return model(track_left=batch["track_left"], track_right=batch["track_right"])


def _masked_l1_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    l1 = (pred - gt).abs() * mask[..., None]
    return l1.sum() / mask.sum()

def train(
    model_name: str = "mlp_planner",
    exp_dir: str = "logs",
    num_epoch: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 2025,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    transform_pipeline = _choose_transform(model_name)

    train_loader = road_dataset.load_data(
        "drive_data/train",
        transform_pipeline=transform_pipeline,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = road_dataset.load_data(
        "drive_data/val",
        transform_pipeline=transform_pipeline,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    model = MODEL_FACTORY[model_name]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_metric = PlannerMetric()
    val_metric = PlannerMetric()

    global_step = 0

    for epoch in range(num_epoch):
        model.train()
        train_metric.reset()

        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            optimizer.zero_grad()
            preds = _forward(model_name, model, batch)

            loss = _masked_l1_loss(preds, batch["waypoints"], batch["waypoints_mask"])
            loss.backward()
            optimizer.step()

            train_metric.add(preds.detach(), batch["waypoints"], batch["waypoints_mask"])
            logger.add_scalar("loss/train", loss.item(), global_step)
            global_step += 1

        train_stats = train_metric.compute()

        # --------------------------- Validation ------------------------------
        model.eval()
        val_metric.reset()
        with torch.inference_mode():
            for batch in val_loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                preds = _forward(model_name, model, batch)
                val_metric.add(preds, batch["waypoints"], batch["waypoints_mask"])

        val_stats = val_metric.compute()

        # --------------------------- Logging ---------------------------------
        for k in ("l1_error", "longitudinal_error", "lateral_error"):
            logger.add_scalar(f"{k}/train", train_stats[k], epoch)
            logger.add_scalar(f"{k}/val", val_stats[k], epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:>3}/{num_epoch}: "
                f"train L1={train_stats['l1_error']:.3f} "
                f"val L1={val_stats['l1_error']:.3f} "
                f"val long={val_stats['longitudinal_error']:.3f} "
                f"val lat={val_stats['lateral_error']:.3f}"
            )

    # -------------------------------------------------------------------------
    save_path = save_model(model)
    print(f"Finished training! Weights saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlp_planner", choices=list(MODEL_FACTORY.keys()))
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)

    train(**vars(parser.parse_args()))
