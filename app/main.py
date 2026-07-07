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
    description="Production-ready Instance Segmentation microservice optimized for containerized deployments.",
    version="1.0.0"
)

# Load model into memory once at application startup to prevent performance bottlenecks
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# 2. Add Ultralytics structure mappings to PyTorch's secure serialization allowlist
try:
    import ultralytics.nn.tasks
    # Registered both DetectionModel and SegmentationModel to ensure instance masks load correctly[cite: 2]
    torch.serialization.add_safe_globals([
        ultralytics.nn.tasks.DetectionModel,
        ultralytics.nn.tasks.SegmentationModel
    ])
except Exception:
    pass # Safe fallback for older local PyTorch environments[cite: 2]

# 3. Safe initialization sequence
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to initialize YOLO model weights at {MODEL_PATH}: {e}")[cite: 2]

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
        )[cite: 2]

    try:
        # 2. Extract leading byte chunk to verify magic number signatures without loading full pixel matrices
        header_bytes = await file.read(2048)[cite: 2]
        await file.seek(0)[cite: 2]
        
        image = Image.open(io.BytesIO(header_bytes))[cite: 2]
        image.verify()  # Validates image integrity and format structure[cite: 2]
        
        # 3. Read full payload into memory for downstream processing
        full_bytes = await file.read()[cite: 2]
        await file.seek(0) # Reset pointer back to zero to follow standard stream lifecycle conventions[cite: 2]
        return full_bytes
        
    except Exception:
        raise HTTPException(
            status_code=400, 
            detail="Corrupted Payload: Uploaded file is not a valid image structure."
        )[cite: 2]


@app.get("/", tags=["Infrastructure"])
def health_check():
    """
    Heuristics health probe endpoint for cluster orchestrators or uptime monitors.
    """
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
    # Validate payload integrity using the security middleware wrapper
    image_bytes = await validate_image(file)[cite: 2]
    
    try:
        # Load safe bytes directly into a Pillow context converted to standard RGB space[cite: 2]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")[cite: 2]
        
        # Execute forward-pass inference with high-resolution segmentation masks enabled
        results = model(image, retina_masks=True)
        
        # Extract the annotated matrix array. 
        # Setting masks=True forces the custom polygon paths to draw cleanly.
        annotated_img_array = results[0].plot(masks=True, boxes=True, line_width=2)
        
        # Reverse color channels slice to fix BGR -> RGB translation before serialization[cite: 2]
        annotated_image = Image.fromarray(annotated_img_array[..., ::-1])[cite: 2]
        
        # Serialize the processed image into an in-memory byte stream buffer[cite: 2]
        img_buffer = io.BytesIO()[cite: 2]
        annotated_image.save(img_buffer, format="JPEG", quality=90)[cite: 2]
        img_buffer.seek(0)[cite: 2]
        
        return StreamingResponse(img_buffer, media_type="image/jpeg")[cite: 2]

    except Exception as e:
        # Catch and isolate internal inference anomalies without breaking the main runtime worker
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Inference Engine Error: {str(e)}"
        )[cite: 2]