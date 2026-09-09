"""
FastAPI Serving Application for AI-Generated Image Detection.
Provides RESTful endpoints for webpage analysis, batch forensic uploads,
Evidently AI drift reporting, and the human-crafted forensic studio dashboard.
"""

import base64
from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT, settings
from src.inference import AIImagePredictor
from src.preprocess import load_image_safely
from src.scrape import SSRFSecurityError, WebpageImageScraper, validate_url_security
from src.utils import logger


# Global predictor instance
predictor: Optional[AIImagePredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes model predictor on startup."""
    global predictor
    logger.info("Initializing AI Image Predictor in FastAPI lifespan...")
    predictor = AIImagePredictor()
    yield
    logger.info("Shutting down API server...")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Local MLOps pipeline for AI-generated image detection on web pages.",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount temporary directory for serving scraped image previews
settings.temp_dir.mkdir(parents=True, exist_ok=True)
app.mount("/previews", StaticFiles(directory=str(settings.temp_dir)), name="previews")

# Mount static CSS & JS assets
static_dir = PROJECT_ROOT / "src" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Schemas
class AnalyzeRequest(BaseModel):
    url: str = Field(
        ...,
        json_schema_extra={"example": "https://en.wikipedia.org/wiki/Artificial_intelligence"},
    )
    max_images: int = Field(default=30, ge=1, le=100)


class SingleImageResult(BaseModel):
    image_url: str
    preview_url: Optional[str] = None
    label: str  # 'REAL' or 'AI_GENERATED'
    ai_probability: float
    real_probability: float
    confidence: float
    width: Optional[int] = None
    height: Optional[int] = None
    sha256: Optional[str] = None
    fft_spectrum: Optional[List[float]] = None
    fft_heatmap: Optional[str] = None
    latency_ms: Optional[float] = None


class AnalyzeResponse(BaseModel):
    target_url: str
    total_images_analyzed: int
    ai_generated_count: int
    real_count: int
    ai_percentage: float
    results: List[SingleImageResult]


@app.get("/health", tags=["Monitoring"])
def health_check() -> Dict[str, Any]:
    """Health check endpoint returning model status and configuration."""
    return {
        "status": "healthy",
        "version": settings.api_version,
        "device": settings.device,
        "model_loaded": predictor is not None,
        "onnx_active": predictor.onnx_session is not None if predictor else False,
    }


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Inference"])
def analyze_webpage(payload: AnalyzeRequest) -> AnalyzeResponse:
    """
    Scrapes all images from target URL, deduplicates them, and classifies each as Real vs AI.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized yet.",
        )

    # 1. SSRF and URL validation
    try:
        validated_url = validate_url_security(str(payload.url))
    except SSRFSecurityError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Security violation: {err}",
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid URL: {err}",
        )

    # 2. Scrape & Deduplicate
    try:
        scraper = WebpageImageScraper(max_images=payload.max_images)
        manifest, session_dir = scraper.scrape_url(validated_url)
    except Exception as err:
        logger.error(f"Scraping error on {validated_url}: {err}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to scrape webpage: {err}",
        )

    # 3. Batch Inference
    results: List[SingleImageResult] = []
    ai_count = 0

    for item in manifest:
        try:
            local_path = Path(item["local_path"])
            pred = predictor.predict_image(local_path, include_fft=True)

            if pred["label"] == "AI_GENERATED":
                ai_count += 1

            rel_path = local_path.relative_to(settings.temp_dir).as_posix()
            preview_url = f"/previews/{rel_path}"

            results.append(
                SingleImageResult(
                    image_url=item["url"],
                    preview_url=preview_url,
                    label=pred["label"],
                    ai_probability=pred["ai_probability"],
                    real_probability=pred["real_probability"],
                    confidence=pred["confidence"],
                    width=item.get("width"),
                    height=item.get("height"),
                    sha256=item.get("sha256"),
                    fft_spectrum=pred.get("fft_spectrum"),
                    fft_heatmap=pred.get("fft_heatmap"),
                    latency_ms=pred.get("latency_ms"),
                )
            )
        except Exception as e:
            logger.warning(f"Error predicting image {item['url']}: {e}")
            continue

    total = len(results)
    ai_pct = round((ai_count / total) * 100, 2) if total > 0 else 0.0

    return AnalyzeResponse(
        target_url=validated_url,
        total_images_analyzed=total,
        ai_generated_count=ai_count,
        real_count=total - ai_count,
        ai_percentage=ai_pct,
        results=results,
    )


@app.post("/predict", tags=["Inference"])
async def predict_single_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Direct single-image file upload endpoint.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized.",
        )

    content = await file.read()
    if len(content) > settings.max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds maximum allowed size (15MB).",
        )

    try:
        pred = predictor.predict_image(content, include_fft=True)
        return {"filename": file.filename, **pred}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process image: {e}",
        )


@app.post("/predict-batch", tags=["Inference"])
async def predict_batch_images(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    Batch multi-image file upload endpoint for Forensic Studio.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized.",
        )

    results = []
    for file in files:
        content = await file.read()
        if len(content) > settings.max_image_bytes:
            continue

        try:
            pred = predictor.predict_image(content, include_fft=True)
            b64_str = base64.b64encode(content).decode("ascii")
            data_uri = f"data:image/jpeg;base64,{b64_str}"

            results.append(
                {
                    "image_url": file.filename,
                    "preview_url": data_uri,
                    "label": pred["label"],
                    "ai_probability": pred["ai_probability"],
                    "real_probability": pred["real_probability"],
                    "confidence": pred["confidence"],
                    "fft_spectrum": pred.get("fft_spectrum"),
                    "fft_heatmap": pred.get("fft_heatmap"),
                    "latency_ms": pred.get("latency_ms"),
                }
            )
        except Exception as e:
            logger.warning(f"Error predicting batch image {file.filename}: {e}")
            continue

    return {"total": len(results), "results": results}


@app.get("/monitoring/status", tags=["Monitoring"])
def get_monitoring_status() -> Dict[str, Any]:
    """Returns active drift metrics and status."""
    from src.monitor import generate_drift_report

    if not settings.drift_metrics_json.exists():
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, (50, 16))
        cur = rng.normal(0.1, 1.0, (50, 16))
        generate_drift_report(ref, cur)

    with open(settings.drift_metrics_json, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/reports/drift", response_class=HTMLResponse, tags=["Monitoring"])
def get_drift_report_html() -> HTMLResponse:
    """Serves the standalone interactive Evidently HTML drift report."""
    from src.monitor import generate_drift_report

    if not settings.drift_report_html.exists():
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, (50, 16))
        cur = rng.normal(0.1, 1.0, (50, 16))
        generate_drift_report(ref, cur)

    with open(settings.drift_report_html, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/", response_class=HTMLResponse, tags=["UI"])
def serve_ui() -> HTMLResponse:
    """Serves the modern interactive forensic dashboard from template."""
    template_path = PROJECT_ROOT / "src" / "templates" / "index.html"
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
