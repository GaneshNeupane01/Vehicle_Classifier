import io
from pathlib import Path

import torch
import timm
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from torchvision import transforms


app = FastAPI(title="Vehicle Type Classifier API")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Resolve model path relative to project root: <repo_root>/models/vehicle_classifier_best.pth
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "vehicle_classifier_best.pth"
model = None
class_names = None



class PredictionResponse(BaseModel):
    label: str
    confidence: float



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



@app.on_event("startup")
def load_model():
    global model, class_names

    checkpoint = torch.load(str(MODEL_PATH), map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = timm.create_model(
        "convnext_tiny",
        pretrained=False,
        num_classes=num_classes
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("✅ Model loaded successfully")



@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()

        label = class_names[pred_idx]
        confidence = probs[0][pred_idx].item()

        return PredictionResponse(
            label=label,
            confidence=round(confidence, 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health Check
@app.get("/")
def root():
    return {"status": "Vehicle Classifier API is running 🚗🏍️"}
