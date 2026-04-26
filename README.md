# LEGO 42176 Autonomous Garage Parking

Train a physical AI model to drive a LEGO Technic 42176 car into a garage using imitation learning.

Uses [LeRobot](https://github.com/huggingface/lerobot) framework with an ACT (Action Chunking with Transformers) policy, trained on PyTorch MPS backend (Apple Silicon).

## Demo

Trained model autonomously driving the car into the garage:

https://github.com/user-attachments/assets/example1

https://github.com/user-attachments/assets/example2

Screen recording of the inference UI:

https://github.com/user-attachments/assets/example3

> Replace the placeholder video URLs above: edit this README on GitHub, drag & drop the files from [`example_videos/`](example_videos/) into the editor, and GitHub will generate the embed URLs.

## Dataset

The recorded dataset (100 episodes, 12k frames, 13 min of driving) is published on HuggingFace:

**[pbelevich/lego42176_garage_parking](https://huggingface.co/datasets/pbelevich/lego42176_garage_parking)**

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("pbelevich/lego42176_garage_parking")
```

## Hardware

- **LEGO Technic 42176** — Porsche GT4 e-Performance with Bluetooth (BLE) control
- **IP camera** — any RTSP-capable camera (640x480 @ 15fps)
- **Mac with Apple Silicon** — M1/M2/M3 for MPS-accelerated training

## Architecture

The system uses behavior cloning: a human drives the car into the garage while the camera records observations. The ACT policy learns to map camera images + car state (speed, steering) to actions.

```
Camera (RTSP) ──► ACT Policy ──► speed, steering ──► LEGO Hub (BLE)
                    │
              ResNet18 encoder
              + Transformer
              + VAE
```

## Setup

```bash
# Create Python 3.12 virtualenv
python3.12 -m venv venv_record
source venv_record/bin/activate

# Install dependencies
pip install lerobot pygame bleak opencv-python-headless

# Login to wandb (optional, for training metrics)
wandb login
```

## Usage

### 1. Manual driving

Drive the car from the keyboard (no recording, no model):

```bash
python main.py
```

### 2. Record dataset

Drive the car while recording camera frames and controller state into a LeRobot dataset:

```bash
python record_dataset.py
```

| Key | Action |
|-----|--------|
| Arrows / WASD | Drive |
| Space | Emergency stop |
| R | Start / stop recording an episode |
| N | Discard current episode |
| Q / Esc | Quit and save |

Press **R**, drive to the garage, press **R** again to save the episode. Repeat. Each episode is finalized to disk immediately so no data is lost if the process exits.

The dataset is saved to `./lego_garage_dataset/` in LeRobot v3.0 format.

### 3. Train

Train an ACT policy on the recorded dataset:

```bash
python train.py
```

Training runs on **MPS** (Apple Silicon GPU). Checkpoints are saved to `./outputs/train/checkpoints/` every 10k steps.

The trained model is published on HuggingFace: **[pbelevich/lego42176_garage_parking_act](https://huggingface.co/pbelevich/lego42176_garage_parking_act)**

Key hyperparameters (edit at the top of `train.py`):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `TRAINING_STEPS` | 50,000 | ~6h on M1 Pro |
| `BATCH_SIZE` | 8 | |
| `LEARNING_RATE` | 1e-4 | |
| `CHUNK_SIZE` | 20 | Action horizon (~1.3s at 15fps) |
| `DIM_MODEL` | 256 | Transformer hidden dim |
| `VISION_BACKBONE` | resnet18 | ImageNet-pretrained |

### 4. Run inference

Let the trained model drive the car:

```bash
python inference.py outputs/train/checkpoints/050000/pretrained_model
```

| Key | Action |
|-----|--------|
| Enter | Start / stop autonomous driving |
| Space | Emergency stop (back to manual) |
| Arrows / WASD | Manual drive |
| Q / Esc | Quit |

Press **Enter** to hand control to the model. Press **Space** at any time to stop.

## Project structure

```
├── main.py              # Manual keyboard driving (BLE only)
├── record_dataset.py    # Record dataset while driving
├── train.py             # Train ACT policy on recorded dataset
├── inference.py         # Run trained policy on the car
├── example_videos/      # Demo videos of the model in action
├── lego_garage_dataset/ # Recorded dataset (not in git)
├── outputs/             # Training checkpoints and logs (not in git)
└── venv_record/         # Python 3.12 virtualenv (not in git)
```

## Configuration

Set the RTSP camera URL via environment variable:

```bash
export RTSP_URL="rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=1"
```

Or copy `.env.example` to `.env` and edit it (`.env` is gitignored).

The LEGO hub is discovered by Bluetooth name (`Technic Move`). Make sure the LEGO Powered Up app is closed before connecting.

## References

- [LeRobot](https://github.com/huggingface/lerobot) — Hugging Face framework for real-world robotics
- [ACT policy](https://arxiv.org/abs/2304.13705) — Action Chunking with Transformers (Zhao et al., 2023)
- [LEGO Powered Up protocol](https://lego.github.io/lego-ble-wireless-protocol-docs/) — BLE communication spec
