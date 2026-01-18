import io
import argparse
from pathlib import Path

import torch
import timm
from PIL import Image
from torchvision import transforms


# Resolve model path relative to project root: <repo_root>/models/vehicle_classifier_best.pth
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "vehicle_classifier_best.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    checkpoint = torch.load(str(MODEL_PATH), map_location=device)

    # Handle both possible checkpoint formats
    class_names = checkpoint.get("class_names") or checkpoint.get("classes")
    if class_names is None:
        raise RuntimeError("Checkpoint must contain 'class_names' or 'classes'.")

    num_classes = len(class_names)

    model = timm.create_model(
        "convnext_tiny",
        pretrained=False,
        num_classes=num_classes,
    )

    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model")
    if state_dict is None:
        raise RuntimeError("Checkpoint must contain 'model_state_dict' or 'model'.")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, class_names


def predict_image(model, class_names, image_path: str):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    label = class_names[pred_idx]
    confidence = probs[0][pred_idx].item()
    return label, confidence


def main():
    parser = argparse.ArgumentParser(description="Vehicle type prediction (CLI)")
    parser.add_argument("image", help="Path to input image")
    args = parser.parse_args()

    model, class_names = load_model()
    label, confidence = predict_image(model, class_names, args.image)

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
