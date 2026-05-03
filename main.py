"""
main.py — IITF End-to-End Training Entry Point
================================================
Wires together the full Phase-1 pipeline:

    IITFConfig → get_dataloaders() → DualTraceLSTM
    → AdamW + ReduceLROnPlateau → Trainer.fit()
    → evaluate_model() on held-out test set

Usage:
    uv run python main.py
    uv run python main.py --config checkpoints/config.json   # resume from saved config
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make src/ importable regardless of how the script is invoked
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import IITFConfig
from data_loader import get_dataloaders
from evaluate import evaluate_model, load_best_model
from model import DualTraceLSTM, get_loss_fn
from trainer import Trainer

# ---------------------------------------------------------------------------
# Logging — configure once at the entry point
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("iitf.main")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IITF Phase-1 Training Job")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a saved config.json to resume from. Omit to use defaults.",
    )
    p.add_argument(
        "--root-dir",
        type=str,
        default=".",
        metavar="DIR",
        help="Project root containing data/raw/. Default: '.'",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ── 1. Configuration ──────────────────────────────────────────────────
    if args.config:
        cfg = IITFConfig.from_json(args.config)
        logger.info("Loaded config from %s", args.config)
    else:
        cfg = IITFConfig()
        logger.info("Using default IITFConfig")

    logger.info("\n%s", cfg)

    # ── 2. Device selection ───────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        logger.info(
            "CUDA device: %s  (%d MiB)",
            props.name,
            props.total_memory // (1024 ** 2),
        )
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available — training on CPU (will be slow).")

    # ── 3. Data loaders ───────────────────────────────────────────────────
    logger.info("Building DataLoaders …")
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=args.root_dir,
        seq_len=cfg.seq_len,
        window_stride=cfg.window_stride,
        batch_size=cfg.batch_size,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        scaler_cache_dir=cfg.scaler_cache_dir,
    )

    # Infer num_features from the first batch to guard against config mismatch
    try:
        sample_x, _, _ = next(iter(train_loader))
        inferred_features = sample_x.shape[-1]
        if cfg.num_features != inferred_features:
            logger.warning(
                "Config num_features=%d but data has %d — overriding.",
                cfg.num_features,
                inferred_features,
            )
            cfg.num_features = inferred_features
    except StopIteration:
        logger.error("Training loader is empty — check data/raw/ directories.")
        sys.exit(1)

    # ── 4. Model ──────────────────────────────────────────────────────────
    logger.info("Instantiating DualTraceLSTM …")
    model = DualTraceLSTM(
        num_features=cfg.num_features,
        hidden_size=cfg.hidden_size,
        num_lstm_layers=cfg.num_lstm_layers,
        embedding_dim=cfg.embedding_dim,
        dropout_rate=cfg.dropout_rate,
    ).to(device)

    # ── 5. Loss, optimizer, scheduler ─────────────────────────────────────
    loss_fn = get_loss_fn(delta=1.0)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        min_lr=cfg.scheduler_min_lr,
    )

    # ── 6. Trainer ────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        save_dir=cfg.save_dir,
        grad_clip_norm=cfg.grad_clip_norm,
        lr_scheduler=scheduler,
    )

    # ── 7. Persist config alongside checkpoints ───────────────────────────
    config_path = Path(cfg.save_dir) / "config.json"
    cfg.to_json(config_path)
    logger.info("Config saved → %s", config_path)

    # ── 8. Training ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting training for %d epochs", cfg.epochs)
    logger.info("=" * 60)

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=cfg.epochs,
    )

    # ── 9. Final evaluation on test set ───────────────────────────────────
    best_ckpt = Path(cfg.save_dir) / "best_model.pth"
    if best_ckpt.exists():
        model, best_epoch, best_val_loss = load_best_model(
            model, str(best_ckpt), device
        )
        logger.info(
            "Loaded best checkpoint (epoch=%d, val_loss=%.4f)",
            best_epoch,
            best_val_loss,
        )
    else:
        logger.warning("best_model.pth not found — evaluating last weights.")

    logger.info("=" * 60)
    logger.info("Test-set evaluation (per-trace MAE / RMSE)")
    logger.info("=" * 60)
    evaluate_model(model, test_loader, device, verbose=True)

    logger.info("IITF Phase-1 run complete.")


if __name__ == "__main__":
    main()
