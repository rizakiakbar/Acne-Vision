"""
FastAPI — acne Level Classification
Menggunakan model CNN yang sudah ditraining dari notebook cnn_acne_level.ipynb
Python  : 3.10.9

Endpoint:
  GET  /                  → Info API
  GET  /health            → Health check
  POST /predict           → Prediksi satu gambar
  POST /predict/batch     → Prediksi banyak gambar sekaligus
  GET  /model/info        → Info model yang diload
"""

import os
import io
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import tensorflow as tf
from PIL import Image

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "output", "best_model.h5")
IMG_SIZE     = (128, 128)
CLASS_NAMES  = ["Level 0", "Level 1", "Level 2", "Level 3"]
CLASS_DESC   = {
    "Level 0": "Tingkat keparahan Sangat Rendah",
    "Level 1": "Tingakt Keparahan rendah",
    "Level 2": "Tingakat Keparahan sedang",
    "Level 3": "Tingakat Keparahan tinggi",
}
MAX_FILE_SIZE = 10 * 1024 * 1024   # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model = None

def load_model():
    global model
    if not Path(MODEL_PATH).exists():
        logger.warning(f"Model tidak ditemukan di '{MODEL_PATH}'. API berjalan tanpa model.")
        return
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Model berhasil dimuat dari '{MODEL_PATH}'")
        logger.info(f"Input shape  : {model.input_shape}")
        logger.info(f"Output shape : {model.output_shape}")
    except Exception as e:
        logger.error(f"Gagal memuat model: {e}")

load_model()

# ─────────────────────────────────────────────
# INISIALISASI FASTAPI
# ─────────────────────────────────────────────
app = FastAPI(
    title="🧠 Acne Level Classification API",
    description=(
        "API untuk mengklasifikasikan tingkat Keparahan Jerawat dari gambar. "
        "Menggunakan model CNN.\n\n"
        "**Kelas:**\n"
        "- `Level 0` → Sangat Rendah\n"
        "- `Level 1` → Rendah\n"
        "- `Level 2` → Sedang\n"
        "- `Level 3` → Tinggi\n"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────
class PredictionResult(BaseModel):
    filename       : str
    predicted_class: str
    class_index    : int
    confidence     : float
    confidence_pct : str
    description    : str
    probabilities  : dict
    inference_time_ms: float

class BatchPredictionResponse(BaseModel):
    total_images: int
    results     : List[PredictionResult]
    total_time_ms: float

class ModelInfo(BaseModel):
    model_loaded : bool
    model_path   : str
    input_shape  : Optional[str]
    output_shape : Optional[str]
    num_classes  : int
    class_names  : List[str]

class HealthResponse(BaseModel):
    status      : str
    model_loaded: bool
    tensorflow  : str

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Konversi bytes gambar → tensor numpy siap untuk inferensi.
    
    Steps:
    1. Decode bytes → PIL Image
    2. Convert ke RGB (handle PNG dengan alpha channel)
    3. Resize ke IMG_SIZE
    4. Normalisasi pixel ke [0, 1]
    5. Tambah batch dimension
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 128, 128, 3)


def run_inference(image_tensor: np.ndarray) -> tuple:
    """
    Jalankan prediksi dan kembalikan (class_idx, probabilities, time_ms).
    """
    t0    = time.perf_counter()
    probs = model.predict(image_tensor, verbose=0)[0]
    ms    = (time.perf_counter() - t0) * 1000
    return int(np.argmax(probs)), probs, round(ms, 2)


def validate_file(file: UploadFile):
    """Validasi tipe dan ukuran file."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Tipe file tidak didukung: '{file.content_type}'. Gunakan JPEG atau PNG."
        )

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    """Informasi umum API."""
    return {
        "api"        : "Acne Level Classification API",
        "version"    : "1.0.0",
        "status"     : "running",
        "model_ready": model is not None,
        "endpoints"  : {
            "docs"          : "/docs",
            "redoc"         : "/redoc",
            "health"        : "/health",
            "predict"       : "/predict",
            "predict_batch" : "/predict/batch",
            "model_info"    : "/model/info",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health_check():
    """Cek status API dan model."""
    return HealthResponse(
        status       = "ok",
        model_loaded = model is not None,
        tensorflow   = tf.__version__,
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
def model_info():
    """Informasi detail tentang model yang digunakan."""
    return ModelInfo(
        model_loaded = model is not None,
        model_path   = MODEL_PATH,
        input_shape  = str(model.input_shape)  if model else None,
        output_shape = str(model.output_shape) if model else None,
        num_classes  = len(CLASS_NAMES),
        class_names  = CLASS_NAMES,
    )


@app.post("/predict", response_model=PredictionResult, tags=["Prediksi"])
async def predict(file: UploadFile = File(..., description="File gambar (JPEG/PNG)")):
    """
    Prediksi tingkat Keparahan dari **satu gambar**.
    
    - Upload gambar dalam format JPEG atau PNG
    - Maksimal ukuran file: 10 MB
    - Mengembalikan kelas prediksi beserta probabilitas tiap kelas
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model belum dimuat. Pastikan file model tersedia di path yang dikonfigurasi."
        )

    validate_file(file)

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Ukuran file melebihi batas {MAX_FILE_SIZE // (1024*1024)} MB."
        )

    try:
        tensor        = preprocess_image(image_bytes)
        idx, probs, ms = run_inference(tensor)
    except Exception as e:
        logger.error(f"Error saat memproses gambar '{file.filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Gagal memproses gambar: {str(e)}"
        )

    logger.info(f"Prediksi '{file.filename}' → {CLASS_NAMES[idx]} ({probs[idx]*100:.1f}%) [{ms}ms]")

    return PredictionResult(
        filename        = file.filename,
        predicted_class = CLASS_NAMES[idx],
        class_index     = idx,
        confidence      = round(float(probs[idx]), 4),
        confidence_pct  = f"{probs[idx]*100:.2f}%",
        description     = CLASS_DESC[CLASS_NAMES[idx]],
        probabilities   = {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)},
        inference_time_ms = ms,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediksi"])
async def predict_batch(files: List[UploadFile] = File(..., description="Beberapa file gambar")):
    """
    Prediksi tingkat Keparahan  dari **banyak gambar sekaligus**.
    
    - Upload hingga banyak gambar dalam satu request
    - Format: JPEG atau PNG
    - Mengembalikan hasil prediksi untuk setiap gambar
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model belum dimuat."
        )

    if len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tidak ada file yang dikirim."
        )

    t_start = time.perf_counter()
    results = []

    for file in files:
        validate_file(file)
        image_bytes = await file.read()

        try:
            tensor        = preprocess_image(image_bytes)
            idx, probs, ms = run_inference(tensor)
            results.append(PredictionResult(
                filename        = file.filename,
                predicted_class = CLASS_NAMES[idx],
                class_index     = idx,
                confidence      = round(float(probs[idx]), 4),
                confidence_pct  = f"{probs[idx]*100:.2f}%",
                description     = CLASS_DESC[CLASS_NAMES[idx]],
                probabilities   = {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)},
                inference_time_ms = ms,
            ))
        except Exception as e:
            logger.warning(f"Gagal memproses '{file.filename}': {e}")
            results.append(PredictionResult(
                filename        = file.filename,
                predicted_class = "ERROR",
                class_index     = -1,
                confidence      = 0.0,
                confidence_pct  = "0%",
                description     = f"Error: {str(e)}",
                probabilities   = {},
                inference_time_ms = 0.0,
            ))

    total_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(f"Batch prediksi: {len(files)} gambar selesai dalam {total_ms}ms")

    return BatchPredictionResponse(
        total_images  = len(files),
        results       = results,
        total_time_ms = total_ms,
    )
