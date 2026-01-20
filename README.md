# Vehicle Type Classifier
## 🟢 Project is live on HuggingFace-->FastAPI https://botinfinity-vehicle-classifier.hf.space/docs

This project is a small, practical vehicle type classifier. The model is a ConvNeXt-Tiny fine-tuned on a Roboflow vehicle dataset, and you can use it in three ways:

- As a FastAPI REST API
- As a Docker container
- As a simple local CLI script (no API, no Docker)

The goal is to be easy to run and easy to understand.

---

## Features

- ConvNeXt-Tiny backbone fine-tuned for vehicle type classification.
- REST API using FastAPI (`/predict`) with automatic docs.
- Docker image for easy deployment.
- Local CLI script for quick one-off predictions.
- Training script that downloads data from Roboflow and does staged fine-tuning.

---

## Project Structure

- `app/`
  - `main.py` – FastAPI app exposing `/` and `/predict`.
- `scripts/`
  - `predict_local.py` – Simple CLI for local prediction on a single image.
  - `trainModel.py` – Training pipeline using the Roboflow dataset and ConvNeXt-Tiny.
- `models/`
  - `vehicle_classifier_best.pth` – Trained model checkpoint used for inference.
  - `best_vehicle_model.pth` – Best checkpoint saved during training (for fine-tuning).
- `requirements.txt` – Python dependencies.
- `Dockerfile` – Docker image definition for serving the API.

---
## Clone the project
```bash
   git clone https://github.com/GaneshNeupane01/Vehicle_Classifier.git
   cd Vehicle_Classifier
```

## Quickstart (Docker)

If you are comfortable with Docker, this is the fastest way to try it.

### 1. Build the image

```bash
docker build -t vehicle-type-classifier .
```

### 2. Run the container

```bash
docker run --rm -p 7860:7860 vehicle-type-classifier
```

The API will be available at `http://localhost:7860` with docs at `http://localhost:7860/docs`.

You can hit it with the same `curl` command as in the local section below.

---

## API Endpoints

### `GET /`

- Simple health check.
- Returns a JSON message indicating the API is running.

### `POST /predict`

- Accepts an image file (`multipart/form-data`, field name: `file`).
- Returns:
  - `label`: predicted vehicle class.
  - `confidence`: softmax probability for the predicted class.

Example request/response are shown in the Quickstart section.

---

## Quickstart (Local, No Docker)

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\\Scripts\\activate  # Windows
```

### 2. Install PyTorch (short guide)

If you already have PyTorch installed, you can skip this step.

```bash
# macOS (CPU)
pip install torch torchvision

# Windows/Linux (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Windows/Linux (GPU, CUDA 11.8) – optional
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For more detailed – and always up to date – instructions, see the official docs: https://pytorch.org/get-started/locally/

### 3. Install project dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Make sure `vehicle_classifier_best.pth` is present in the project root (it is already included in this repo).

### 4. Run the FastAPI app locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

Then open:

- Interactive docs: http://localhost:7860/docs
- Health check: http://localhost:7860/

### 5. Test the API with `curl`

```bash
curl -X POST "http://localhost:7860/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg" 
```

You should get back a JSON response like:

```json
{
  "label": "car",
  "confidence": 0.9876
}
```

---

## Quickstart (Local CLI only, No FastAPI)

If you just want to test a single image without running an API or Docker container, use the CLI script:

```bash
python scripts/predict_local.py path/to/your/image.jpg
```

Output example:

```text
Using device: cpu
Prediction: car
Confidence: 0.9876
```

This uses the same model and preprocessing pipeline as the API.

---

## Dataset

The model is trained on a Roboflow classification dataset.

- Workspace: `paul-guerrie-tang1`
- Project: `vehicle-classification-eapcd`
- Version: `19`
- Downloaded in the training script as a folder dataset.

In code (from `scripts/trainModel.py`):

```python
project = rf.workspace("paul-guerrie-tang1").project("vehicle-classification-eapcd")
dataset = project.version(19).download("folder")
```

The script then uses the `train` and `valid` folders from that download.

---

## Training (Optional)

The training script [scripts/trainModel.py](scripts/trainModel.py) will:

- Download the dataset from Roboflow.
- Build a custom `VehicleDataset` with train/validation splits.
- Perform staged training:
  - Stage 1: Train the classification head only.
  - Stage 2: Fine-tune the last ConvNeXt stage.
  - Stage 3 (optional): Fine-tune more stages.
- Save the best checkpoint as `best_vehicle_model.pth`.

To use your own Roboflow account and dataset:

1. Replace `YOUR_API_KEY_HERE` in `trainModel.py` with your Roboflow API key.
2. Optionally change the workspace, project, and version IDs.
3. Run:

```bash
python scripts/trainModel.py
```

After training, the script will store `best_vehicle_model.pth` in the `models/` folder. You can then rename or copy it to `vehicle_classifier_best.pth` (also in `models/`) so the API and CLI use your new model.

> Note: For public repositories, **do not commit private API keys**. Use environment variables or local config files that are ignored by git.

---

## Results

Final metrics from training (Stage 3 – Full Fine-Tune, last epoch):

- Loss: `0.0045`
- Accuracy: `87.78%`
- F1-score (weighted): `87.67%`

These numbers come from the last logged epoch:

```text
[Full Fine-Tune] Epoch 8/8
Loss: 0.0045 | Acc: 87.78% | F1: 87.67%
```

---

## Model Details

- Backbone: `convnext_tiny` (via `timm`).
- Input size: 224x224 RGB.
- Normalization: ImageNet mean/std.
- Output: one of the vehicle classes defined in the dataset (see `class_names` in the checkpoint).

The inference scripts (`app/main.py` and `scripts/predict_local.py`) load the model on `cuda` if available, otherwise CPU.

---

## 👤 Author
**Ganesh Neupane**
Computer Engineering | AI / ML
- GitHub: [@GaneshNeupane01](https://github.com/GaneshNeupane01)
- Detail: [@portfolio](https://ganesh-neupane.com.np)
