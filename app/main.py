import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from ultralytics import YOLO
import os
from pathlib import Path
import torch

app = FastAPI(
    title="YOLO Custom Inference API",
    description="Production-ready Object Detection microservice optimized for containerized deployments.",
    version="1.0.0"
)

# Load model into memory once at application startup to prevent performance bottlenecks
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# 2. Add Ultralytics structure mappings to PyTorch's secure serialization allowlist
try:
    import ultralytics.nn.tasks
    torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
except Exception:
    pass # Safe fallback for older local PyTorch environments

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
    # 1. Quick, low-cost header check
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="Security Error: Invalid Content-Type header."
        )

    try:
        # 2. Extract leading byte chunk to verify magic number signatures without loading full pixel matrices
        header_bytes = await file.read(2048) 
        await file.seek(0)
        
        image = Image.open(io.BytesIO(header_bytes))
        image.verify()  # Validates image integrity and format structure
        
        # 3. Read full payload into memory for downstream processing
        full_bytes = await file.read()
        await file.seek(0) # Reset pointer back to zero to follow standard stream lifecycle conventions
        return full_bytes
        
    except Exception:
        raise HTTPException(
            status_code=400, 
            detail="Corrupted Payload: Uploaded file is not a valid image structure."
        )


@app.get("/", tags=["Infrastructure"])
def health_check():
    """
    Heuristics health probe endpoint for cluster orchestrators or uptime monitors.
    """
    return {
        "status": "healthy",
        "service": "YOLO Object Detection Engine",
        "model_loaded": True
    }


@app.post("/predict", tags=["Inference"])
async def predict_image(file: UploadFile = File(...)):
    """
    Ingests an image stream, executes custom YOLO object detection, 
    renders bounding box overlays, and streams the compiled JPEG binary back to the client.
    """
    # Validate payload integrity using the security middleware wrapper
    image_bytes = await validate_image(file)
    
    try:
        # Load safe bytes directly into a Pillow context converted to standard RGB space
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Execute forward-pass inference
        results = model(image)
        
        # Extract the annotated matrix array (Default output format from Ultralytics is BGR)
        annotated_img_array = results[0].plot()
        
        # Reverse color channels slice to fix BGR -> RGB translation before serialization
        annotated_image = Image.fromarray(annotated_img_array[..., ::-1]) 
        
        # Serialize the processed image into an in-memory byte stream buffer
        img_buffer = io.BytesIO()
        annotated_image.save(img_buffer, format="JPEG", quality=90)
        img_buffer.seek(0)
        
        return StreamingResponse(img_buffer, media_type="image/jpeg")

    except Exception as e:
        # Catch and isolate internal inference anomalies without breaking the main runtime worker
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Inference Engine Error: {str(e)}"
        )
    