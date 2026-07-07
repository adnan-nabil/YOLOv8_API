import os
# FORCE DISABLE strict weights_only check for PyTorch 2.6+ 
# This must be set BEFORE torch or ultralytics is imported
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

import io
from pathlib import Path
import collections
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
from ultralytics import YOLO

app = FastAPI(
    title="YOLO Custom Inference API",
    description="Production-ready Instance Segmentation microservice optimized for containerized deployments.",
    version="1.0.0"
)

# Load model into memory once at application startup to prevent performance bottlenecks
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# 2. Backup security allowlist just in case
try:
    import ultralytics.nn.tasks
    torch.serialization.add_safe_globals([
        ultralytics.nn.tasks.DetectionModel,
        ultralytics.nn.tasks.SegmentationModel,
        torch.nn.modules.container.Sequential,
        collections.OrderedDict
    ])
except Exception:
    pass 

# 3. Safe initialization sequence
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to initialize YOLO model weights at {MODEL_PATH}: {e}")

async def validate_image(file: UploadFile) -> bytes:
    """
    Validates both the HTTP header and the internal file structure of the upload.
    Returns the raw file bytes if valid, preventing multiple disk/memory read streams.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="Security Error: Invalid Content-Type header."
        )

    try:
        header_bytes = await file.read(2048)
        await file.seek(0)
        
        image = Image.open(io.BytesIO(header_bytes))
        image.verify()  
        
        full_bytes = await file.read()
        await file.seek(0) 
        return full_bytes
        
    except Exception:
        raise HTTPException(
            status_code=400, 
            detail="Corrupted Payload: Uploaded file is not a valid image structure."
        )


@app.get("/", tags=["Infrastructure"])
def health_check():
    return {
        "status": "healthy",
        "service": "YOLO Instance Segmentation Engine",
        "model_loaded": True
    }


@app.post("/predict", tags=["Inference"])
async def predict_image(file: UploadFile = File(...)):
    """
    Ingests an image stream, executes custom YOLO instance segmentation, 
    renders polygon mask overlays, and streams the compiled JPEG binary back to the client.
    """
    image_bytes = await validate_image(file)
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Execute forward-pass inference with 50% confidence threshold and retina masks enabled
        results = model(image, conf=0.50, retina_masks=True)[0]
        
        # Extract the annotated matrix array using the natively filtered detections
        annotated_img_array = results.plot(masks=True, boxes=True, line_width=2)
        
        # Reverse color channels slice to fix BGR -> RGB translation before serialization
        annotated_image = Image.fromarray(annotated_img_array[..., ::-1]) 
        
        img_buffer = io.BytesIO()
        annotated_image.save(img_buffer, format="JPEG", quality=90)
        img_buffer.seek(0)
        
        return StreamingResponse(img_buffer, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Inference Engine Error: {str(e)}"
        )