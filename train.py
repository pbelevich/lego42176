#!/usr/bin/env python3
"""
Train an ACT policy on the LEGO 42176 garage parking dataset using LeRobot.

Uses PyTorch MPS backend (Apple Silicon).

Usage:
    source venv_record/bin/activate
    python train.py

Prerequisites:
    - A valid dataset at ./lego_garage_dataset/ recorded with record_dataset.py
    - The venv_record virtualenv with lerobot installed

The trained model is saved to ./outputs/train/<run_name>/checkpoints/.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_REPO_ID = "lego42176/garage_parking"
DATASET_ROOT = "./lego_garage_dataset"
OUTPUT_DIR = "./outputs/train"

# Training hyperparameters (tuned for small dataset on MPS)
BATCH_SIZE = 8
TRAINING_STEPS = 50_000
SAVE_FREQ = 10_000
LOG_FREQ = 100
EVAL_FREQ = 0          # no sim env, skip eval
NUM_WORKERS = 0         # 0 avoids macOS multiprocessing issues
SEED = 42

# Weights & Biases
WANDB_ENABLE = True
WANDB_PROJECT = "lego42176-garage"
WANDB_ENTITY = None     # set to your wandb username/org, or None for default

# ACT policy hyperparameters
CHUNK_SIZE = 20         # predict 20 future actions (~1.3s at 15fps)
N_ACTION_STEPS = 20
LEARNING_RATE = 1e-4
VISION_BACKBONE = "resnet18"
DIM_MODEL = 256         # smaller transformer for small dataset
N_HEADS = 4
DIM_FEEDFORWARD = 1024
N_ENCODER_LAYERS = 2
N_DECODER_LAYERS = 1
LATENT_DIM = 16
KL_WEIGHT = 10.0


def main():
    # ---- Imports (deferred so --help is fast) ----
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.scripts.lerobot_train import train

    # ---- Verify dataset exists ----
    dataset_root = os.path.abspath(DATASET_ROOT)
    info_path = os.path.join(dataset_root, "meta", "info.json")
    if not os.path.exists(info_path):
        print(f"ERROR: Dataset not found at {dataset_root}")
        print("Record a dataset first with: python record_dataset.py")
        return

    import json
    with open(info_path) as f:
        info = json.load(f)
    print(f"Dataset: {info['total_episodes']} episodes, {info['total_frames']} frames, {info['fps']} fps")

    # ---- Build configs ----
    dataset_cfg = DatasetConfig(
        repo_id=DATASET_REPO_ID,
        root=dataset_root,
        use_imagenet_stats=True,    # normalize images with ImageNet stats
    )

    policy_cfg = ACTConfig(
        device="mps",
        use_amp=False,              # AMP not supported on MPS
        push_to_hub=False,          # don't push model to HuggingFace Hub

        # Action chunking
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,

        # Vision encoder
        vision_backbone=VISION_BACKBONE,
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",

        # Transformer
        dim_model=DIM_MODEL,
        n_heads=N_HEADS,
        dim_feedforward=DIM_FEEDFORWARD,
        n_encoder_layers=N_ENCODER_LAYERS,
        n_decoder_layers=N_DECODER_LAYERS,

        # VAE
        use_vae=True,
        latent_dim=LATENT_DIM,
        kl_weight=KL_WEIGHT,

        # Optimizer (used by get_optimizer_preset)
        optimizer_lr=LEARNING_RATE,
        optimizer_weight_decay=1e-4,
    )

    train_cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        env=None,                   # no simulation environment

        output_dir=Path(OUTPUT_DIR),
        batch_size=BATCH_SIZE,
        steps=TRAINING_STEPS,
        num_workers=NUM_WORKERS,
        seed=SEED,

        eval_freq=EVAL_FREQ,
        log_freq=LOG_FREQ,
        save_freq=SAVE_FREQ,
        save_checkpoint=True,

        use_policy_training_preset=True,    # use ACT's AdamW preset

        wandb=WandBConfig(
            enable=WANDB_ENABLE,
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
        ),
    )

    print(f"Policy:  ACT (dim={DIM_MODEL}, heads={N_HEADS}, chunk={CHUNK_SIZE})")
    print(f"Device:  {policy_cfg.device}")
    print(f"Steps:   {TRAINING_STEPS}, batch_size={BATCH_SIZE}")
    print(f"WandB:   {'enabled -> ' + WANDB_PROJECT if WANDB_ENABLE else 'disabled'}")
    print(f"Output:  {OUTPUT_DIR}")
    print()

    # ---- Launch training ----
    train(train_cfg)


if __name__ == "__main__":
    main()
