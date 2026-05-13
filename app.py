import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import glob
import io
import functools

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from typing import Optional, Tuple, Dict, Any


class HybridCNNViT(nn.Module):
    def __init__(self, cnn, vit, num_classes):
        super().__init__()
        self.cnn = cnn
        self.vit = vit
        
        # Features from EfficientNet-B3 (1536) + ViT-B/16 (768)
        cnn_out_features = 1536
        vit_out_features = 768
        
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_features + vit_out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        combined = torch.cat((cnn_feat, vit_feat), dim=1)
        return self.fc(combined)

def get_model(device):
    # CNN Backbone: EfficientNet-B3
    cnn = models.efficientnet_b3(weights=None) # We will load weights later, or rely on state_dict
    cnn.classifier = nn.Identity()

    # ViT Backbone: ViT-B/16
    vit = models.vit_b_16(weights=None)
    vit.heads = nn.Identity()

    # Initialize Hybrid Model
    return HybridCNNViT(cnn, vit, num_classes=15)


class DiseaseClassifier(nn.Module):
    """
    ConvNeXt-Tiny based disease classifier.
    Matches the training-time architecture from disease.ipynb:
    - backbone: torchvision.models.convnext_tiny
    - classifier[2]: Dropout -> Linear(768, 256) -> GELU -> Dropout -> Linear(256, num_classes)
    """

    def __init__(self, num_classes):
        super().__init__()
        # Pretrained weights are not required at inference since we load the
        # fine-tuned state_dict; using weights=None avoids any download.
        self.backbone = models.convnext_tiny(weights=None)
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def get_disease_model(device, num_classes):
    disease_model = DiseaseClassifier(num_classes=num_classes)
    disease_model.to(device)
    disease_model.eval()
    return disease_model


torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import zipfile
import shutil

#
# Lightweight "is this cattle?" gate (no training)
#
IMAGENET_GATE_MODEL: Optional[nn.Module] = None
IMAGENET_GATE_PREPROCESS = None
IMAGENET_GATE_CATEGORIES = None
IMAGENET_GATE_ENABLED = os.environ.get("ENABLE_IMAGENET_GATE", "0") == "1"

_CATTLE_KEYWORDS = (
    "cow",
    "ox",
    "bull",
    "water buffalo",
    "bison",
    "yak",
    "zebu",
    "cattle",
)


def _init_imagenet_gate(device: torch.device) -> None:
    """
    Best-effort init of a pre-trained ImageNet classifier used only to reject
    obvious non-cattle uploads (e.g., dog/cat). If weights can't be loaded
    (e.g., no internet), we keep the gate disabled and fall back to the
    breed-model confidence heuristic.
    """
    global IMAGENET_GATE_MODEL, IMAGENET_GATE_PREPROCESS, IMAGENET_GATE_CATEGORIES
    if not IMAGENET_GATE_ENABLED:
        IMAGENET_GATE_MODEL = None
        IMAGENET_GATE_PREPROCESS = None
        IMAGENET_GATE_CATEGORIES = None
        print("ImageNet gate disabled (ENABLE_IMAGENET_GATE=0).")
        return
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        gate = models.resnet50(weights=weights)
        gate.to(device)
        gate.eval()

        IMAGENET_GATE_MODEL = gate
        IMAGENET_GATE_PREPROCESS = weights.transforms()
        IMAGENET_GATE_CATEGORIES = weights.meta.get("categories")
        print("ImageNet gate enabled (ResNet50).")
    except Exception as e:
        IMAGENET_GATE_MODEL = None
        IMAGENET_GATE_PREPROCESS = None
        IMAGENET_GATE_CATEGORIES = None
        print(f"WARNING: ImageNet gate disabled: {e}")


def _is_cattle_by_imagenet(image: Image.Image, threshold: float = 0.20) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns (is_cattle, debug_info). Uses a pre-trained ImageNet model if available.
    """
    # Lazy-init so Render can bind the port quickly.
    if IMAGENET_GATE_MODEL is None and IMAGENET_GATE_ENABLED:
        _init_imagenet_gate(device)

    if IMAGENET_GATE_MODEL is None or IMAGENET_GATE_PREPROCESS is None or not IMAGENET_GATE_CATEGORIES:
        return True, {"gate": "disabled"}

    x = IMAGENET_GATE_PREPROCESS(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = IMAGENET_GATE_MODEL(x)[0]
        probs = torch.nn.functional.softmax(logits, dim=0)
        topk_prob, topk_idx = torch.topk(probs, 5)

    top = []
    cattle_score = 0.0
    for p, idx in zip(topk_prob.tolist(), topk_idx.tolist()):
        label = IMAGENET_GATE_CATEGORIES[idx]
        top.append({"label": label, "score": float(p)})
        if any(k in label.lower() for k in _CATTLE_KEYWORDS):
            cattle_score = max(cattle_score, float(p))

    return cattle_score >= threshold, {"gate": "imagenet", "cattle_score": cattle_score, "top5": top, "threshold": threshold}


def _is_cattle_by_breed_confidence(breed_probabilities: torch.Tensor, threshold: float = 0.65) -> Tuple[bool, Dict[str, Any]]:
    """
    Fallback: if the breed model isn't confident, treat it as 'unknown / not cattle'.
    This is weaker than ImageNet gating but requires no extra weights.
    """
    top1 = float(torch.max(breed_probabilities).item())
    return top1 >= threshold, {"gate": "breed_confidence", "top1": top1, "threshold": threshold}


def repack_model(folder_path, output_path):
    print(f"Repacking model from {folder_path} to {output_path}...")
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_STORED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            # Exclude dot-directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if file.startswith('.'):
                    continue
                file_path = os.path.join(root, file)
                # Prepend 'archive/' to make it compatible with torch.load
                rel_path = os.path.relpath(file_path, folder_path)
                arcname = os.path.join('archive', rel_path).replace(os.path.sep, '/')
                zipf.write(file_path, arcname)
    print("Repacking complete.")


def load_disease_model_weights(disease_model, device):
    """
    Load trained weights for the disease classifier from the `disease` folder.
    Looks for any `.pt` or `.pth` file inside that folder.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    disease_dir = os.path.join(current_dir, "disease")

    if not os.path.isdir(disease_dir):
        raise FileNotFoundError(
            f"Disease model directory not found at {disease_dir}. "
            f"Please place your disease model weights there."
        )

    candidates = glob.glob(os.path.join(disease_dir, "*.pt")) + glob.glob(
        os.path.join(disease_dir, "*.pth")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .pt or .pth files found in {disease_dir}. "
            f"Please add your trained disease model weights (e.g., disease_model.pt)."
        )

    disease_weights_path = candidates[0]
    print(f"Loading disease model from {disease_weights_path}...")

    obj = torch.load(disease_weights_path, map_location=device)

    # Common cases:
    # 1) state_dict directly
    # 2) dict with key "state_dict"
    # 3) full nn.Module (from which we can extract state_dict)
    if isinstance(obj, nn.Module):
        raw_state_dict = obj.state_dict()
    elif isinstance(obj, dict):
        raw_state_dict = obj.get("state_dict", obj)
    else:
        raise TypeError(
            f"Unexpected object type loaded from {disease_weights_path}: {type(obj)}"
        )

    # Filter to only keys that both exist in the current model and have
    # matching tensor shapes. This avoids size-mismatch runtime errors
    # when the checkpoint was trained with a slightly different backbone.
    model_state = disease_model.state_dict()
    filtered_state = {}
    matched_keys = 0
    skipped_keys = 0

    for key, value in raw_state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered_state[key] = value
            matched_keys += 1
        else:
            skipped_keys += 1

    # If nothing matches, the checkpoint is effectively incompatible.
    # In that case we signal failure so the app can treat the disease
    # model as unavailable rather than giving random predictions.
    if matched_keys == 0:
        print(
            f"No matching parameters between checkpoint and DiseaseNet "
            f"({skipped_keys} keys skipped)."
        )
        raise RuntimeError("Disease checkpoint incompatible with DiseaseNet architecture.")

    disease_model.load_state_dict(filtered_state, strict=False)
    print(
        f"Disease model loaded successfully with {matched_keys} matched parameters "
        f"and {skipped_keys} skipped from checkpoint."
    )

MODEL_LOADED = False
DISEASE_MODEL_LOADED = False
model: Optional[nn.Module] = None
disease_model: Optional[nn.Module] = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "best_hybrid_model")
REPACKED_PATH = os.path.join(CURRENT_DIR, "repacked_model.pt")

def _load_breed_model() -> nn.Module:
    """
    Lazy-load the breed model so the server can bind $PORT quickly on Render.
    """
    global MODEL_LOADED, model
    if MODEL_LOADED and model is not None:
        return model

    m = get_model(device)

    if os.path.exists(MODEL_DIR):
        # Check if we need to repack
        if not os.path.exists(REPACKED_PATH):
            try:
                repack_model(MODEL_DIR, REPACKED_PATH)
            except Exception as e:
                print(f"Error repacking model: {e}")

        if os.path.exists(REPACKED_PATH):
            print(f"Loading model from {REPACKED_PATH}...")
            state_dict = torch.load(REPACKED_PATH, map_location=device)
            m.load_state_dict(state_dict)
            print("Model loaded successfully!")
        else:
            raise FileNotFoundError(f"Could not create or find {REPACKED_PATH}")
    else:
        # Fallback search for any .pt file
        print(f"Directory {MODEL_DIR} not found. Searching for .pt files...")
        candidates = glob.glob(os.path.join(CURRENT_DIR, "*.pt"))
        candidates = [c for c in candidates if "repacked_model.pt" not in c and os.path.isfile(c)]

        if candidates:
            print(f"Attempting to load: {candidates[0]}")
            state_dict = torch.load(candidates[0], map_location=device)
            m.load_state_dict(state_dict)
        else:
            raise FileNotFoundError("Could not find model weights file or directory.")

    m.to(device)
    m.eval()
    model = m
    MODEL_LOADED = True
    return m


DISEASE_CLASSES = [
    "foot-and-mouth",
    "healthy",
    "lumpy",
]

DISEASE_INFO = {
    "foot-and-mouth": (
        "Foot-and-mouth disease is a severe, highly contagious viral disease that affects cattle, "
        "causing fever, blisters in the mouth and on the feet, and excessive salivation. "
        "It requires immediate veterinary attention and strict isolation."
    ),
    "healthy": (
        "The animal appears healthy with no visible signs of the target diseases detected by this model. "
        "Always confirm with a veterinarian if you suspect any issues."
    ),
    "lumpy": (
        "Lumpy skin disease is a viral infection causing nodules on the skin, fever, and enlarged lymph nodes. "
        "It spreads through biting insects and needs prompt veterinary diagnosis and treatment."
    ),
}

def _load_disease_model() -> Optional[nn.Module]:
    global DISEASE_MODEL_LOADED, disease_model
    if DISEASE_MODEL_LOADED:
        return disease_model

    try:
        dm = get_disease_model(device, num_classes=len(DISEASE_CLASSES))
        load_disease_model_weights(dm, device)
        dm.to(device)
        dm.eval()
        disease_model = dm
    except Exception as e:
        print(f"WARNING: Disease model is not available: {e}")
        disease_model = None

    DISEASE_MODEL_LOADED = True
    return disease_model


CLASSES = [
    "Ayrshire", "Brown_Swiss", "Gir", "Hallikar", "Holstein_Friesian", 
    "Jersey", "Kankrej", "Murrah", "Nagpuri", "Ongole", 
    "Rathi", "Red_Dane", "Red_Sindhi", "Sahiwal", "Tharparkar"
]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


BREED_INFO = {
    "Ayrshire": "A dairy breed from Scotland known for hardiness and efficient milk production. Red and white markings.",
    "Brown_Swiss": "One of the oldest dairy breeds, from Switzerland. Large, robust, and docile with high-protein milk suitable for cheese.",
    "Gir": "A famous Indian dairy breed from Gujarat. Distinctive rounded forehead and pendulous ears. Highly heat tolerant.",
    "Hallikar": "Draft breed from Karnataka, India. Known for vertical horns and endurance. Progenitor of many South Indian breeds.",
    "Holstein_Friesian": "World's highest-production dairy breed. Originating from Europe, known for distinctive black and white markings.",
    "Jersey": "Small dairy breed from the Channel Islands. Produces milk with very high butterfat content. Known for a gentle temperament.",
    "Kankrej": "Dual-purpose breed from Gujarat/Rajasthan. Large, with lyre-shaped horns. Used for draft and milk (Guzerat in Brazil).",
    "Murrah": "The premier water buffalo breed from India (Haryana/Punjab). Jet black with tightly curled horns. High milk yield.",
    "Nagpuri": "Riverine buffalo breed from Maharashtra. Black with white patches on face/legs. Long sword-like horns.",
    "Ongole": "Large, muscular draft/dual-purpose breed from Andhra Pradesh. White/grey coat. source of the American Brahman breed.",
    "Rathi": "Dual-purpose breed from the Thar Desert (Rajasthan). Brown/red with white patches. Adapted to extreme heat.",
    "Red_Dane": "Dairy breed from Denmark. Solid red color. Good milk production and fertility.",
    "Red_Sindhi": "Zebu dairy breed from Pakistan/India. Deep red color. Compact body, heat tolerant, and disease resistant.",
    "Sahiwal": "Top Zebu dairy breed from Punjab. Reddish-brown. Tick-resistant and heat-tolerant. High milk fat content.",
    "Tharparkar": "Dual-purpose breed from the Thar Desert. White/grey coat. Highly drought and disease resistant."
}

def predict_cattle(image):
    if image is None:
        return {}, "", {}, ""

    breed_model = _load_breed_model()
    dm = _load_disease_model()

    # Preprocess once and use for both models
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        # Breed prediction
        breed_output = breed_model(img_tensor)
        breed_probabilities = torch.nn.functional.softmax(breed_output[0], dim=0)

        # Disease prediction (if model is available)
        disease_probabilities = None
        if dm is not None:
            disease_output = dm(img_tensor)
            disease_probabilities = torch.nn.functional.softmax(disease_output[0], dim=0)

    # Breed: top 5 results
    top5_prob, top5_catid = torch.topk(breed_probabilities, 5)

    breed_result = {}
    for i in range(top5_prob.size(0)):
        class_name = CLASSES[top5_catid[i]]  # Use the sorted list
        score = float(top5_prob[i])
        breed_result[class_name] = score

    top_breed_class = CLASSES[top5_catid[0]]
    breed_info_text = (
        f"**Top Breed Prediction: {top_breed_class}**\n\n"
        f"{BREED_INFO.get(top_breed_class, 'No info available.')}"
    )

    # Disease: top 3 results (or fewer if fewer classes)
    disease_result = {}
    disease_info_text = ""

    if disease_probabilities is not None:
        k = min(3, len(DISEASE_CLASSES))
        topd_prob, topd_catid = torch.topk(disease_probabilities, k)

        for i in range(topd_prob.size(0)):
            class_name = DISEASE_CLASSES[topd_catid[i]]
            score = float(topd_prob[i])
            disease_result[class_name] = score

        top_disease_class = DISEASE_CLASSES[topd_catid[0]]
        disease_info_text = (
            f"**Top Disease Prediction: {top_disease_class}**\n\n"
            f"{DISEASE_INFO.get(top_disease_class, 'No info available.')}"
        )
    else:
        disease_info_text = (
            "Disease model is not available. Please check the server logs and "
            "ensure disease model weights are placed in the `disease` folder."
        )

    return breed_result, breed_info_text, disease_result, disease_info_text


def format_all_breeds_info():
    info = ""
    for breed, desc in sorted(BREED_INFO.items()):
        info += f"### {breed}\n{desc}\n\n"
    return info

app = FastAPI(title="Indian Cattle Breed & Disease Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


FRONTEND_DIR = os.path.join(CURRENT_DIR, "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", response_class=FileResponse)
async def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "device": str(device),
        "breed_model_loaded": MODEL_LOADED,
        "disease_model_loaded": DISEASE_MODEL_LOADED,
        "disease_model_available": disease_model is not None if DISEASE_MODEL_LOADED else False,
        "imagenet_gate_enabled": IMAGENET_GATE_ENABLED,
        "num_breeds": len(CLASSES),
        "num_disease_classes": len(DISEASE_CLASSES),
    }


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    # 1) First: block obvious non-cattle (dog/cat/etc.) without training anything.
    is_cattle, gate_debug = _is_cattle_by_imagenet(image, threshold=0.20)

    # 2) If ImageNet gate is disabled, fallback to a simple confidence heuristic.
    #    (We keep this simple and fast; no extra models.)
    if gate_debug.get("gate") == "disabled":
        img_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            breed_output = _load_breed_model()(img_tensor)
            breed_probabilities = torch.nn.functional.softmax(breed_output[0], dim=0)
        is_cattle, gate_debug = _is_cattle_by_breed_confidence(breed_probabilities, threshold=0.65)

    if not is_cattle:
        detected = None
        if gate_debug.get("gate") == "imagenet":
            top5 = gate_debug.get("top5") or []
            if isinstance(top5, list) and len(top5) > 0:
                detected = top5[0].get("label")

        if detected:
            breed_msg = f"This image looks like {detected}. This classifier is only for cattle breeds."
        else:
            breed_msg = "This image doesn't look like cattle. This classifier is only for cattle breeds."

        return JSONResponse(
            status_code=200,
            content={
                "breed_predictions": [],
                "breed_info": breed_msg,
                "disease_predictions": [],
                "disease_info": "",
                "rejected": True,
                "rejection_reason": "non_cattle",
                "gate_debug": gate_debug,
            },
        )

    breed_result, breed_info_text, disease_result, disease_info_text = predict_cattle(image)

    breed_predictions = [
        {"label": label, "score": float(score)}
        for label, score in breed_result.items()
    ]
    disease_predictions = [
        {"label": label, "score": float(score)}
        for label, score in disease_result.items()
    ]

    return JSONResponse(
        {
            "breed_predictions": breed_predictions,
            "breed_info": breed_info_text,
            "disease_predictions": disease_predictions,
            "disease_info": disease_info_text,
        }
    )


if __name__ == "__main__":
    import uvicorn
    import os
    
    # Render sets a PORT environment variable. If not found, use 8000.
    port = int(os.environ.get("PORT", 8000))
    
    # Note: 'reload=True' should typically be False in production
    uvicorn.run("app:app", host="0.0.0.0", port=port)
