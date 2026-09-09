"""
FastAPI Serving Application for AI-Generated Image Detection.
Provides RESTful endpoints for webpage analysis, single-image prediction, and an interactive UI.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl

from src.config import settings
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


@app.get("/", response_class=HTMLResponse, tags=["UI"])
def serve_ui() -> HTMLResponse:
    """Serves the interactive web interface dashboard."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI-Generated Image Detector | Local MLOps</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0b0f19;
      --card-bg: rgba(18, 24, 40, 0.85);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-purple: #8b5cf6;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-rose: #f43f5e;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Outfit', sans-serif;
      background: radial-gradient(circle at 10% 20%, #171d33 0%, var(--bg-dark) 90%);
      color: var(--text-main);
      min-height: 100vh;
      padding: 2.5rem 1.5rem;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      text-align: center;
      margin-bottom: 2.5rem;
    }
    .badge {
      display: inline-block;
      padding: 0.35rem 0.9rem;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(6, 182, 212, 0.2));
      border: 1px solid rgba(139, 92, 246, 0.4);
      border-radius: 9999px;
      color: #c4b5fd;
      margin-bottom: 0.8rem;
    }
    h1 {
      font-size: 2.5rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #ffffff 40%, #a5b4fc 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.6rem;
    }
    p.subtitle { color: var(--text-muted); font-size: 1.05rem; }

    .input-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 1rem;
      padding: 1.75rem;
      backdrop-filter: blur(12px);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
      margin-bottom: 2rem;
    }
    .input-group {
      display: flex;
      gap: 1rem;
      flex-wrap: wrap;
    }
    input[type="url"] {
      flex: 1;
      min-width: 280px;
      background: rgba(11, 15, 25, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 0.6rem;
      padding: 0.85rem 1.25rem;
      color: #fff;
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.2s;
    }
    input[type="url"]:focus {
      border-color: var(--accent-cyan);
    }
    button.btn-primary {
      background: linear-gradient(135deg, var(--accent-purple), #6366f1);
      color: #fff;
      border: none;
      border-radius: 0.6rem;
      padding: 0.85rem 1.8rem;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.2s, transform 0.1s;
    }
    button.btn-primary:hover { opacity: 0.92; transform: translateY(-1px); }
    button.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

    .stats-bar {
      display: none;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }
    .stat-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 0.75rem;
      padding: 1.2rem;
      text-align: center;
    }
    .stat-value {
      font-size: 1.8rem;
      font-weight: 700;
      margin-top: 0.25rem;
    }
    .stat-label { font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; }

    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 1.5rem;
    }
    .image-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 0.8rem;
      overflow: hidden;
      transition: transform 0.2s, border-color 0.2s;
    }
    .image-card:hover {
      transform: translateY(-3px);
    }
    .image-card.ai { border-color: rgba(244, 63, 94, 0.5); }
    .image-card.real { border-color: rgba(16, 185, 129, 0.5); }
    .image-preview {
      width: 100%;
      height: 200px;
      object-fit: cover;
      background: #05070d;
      display: block;
    }
    .card-body { padding: 1rem; }
    .card-label {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.6rem;
    }
    .tag-ai {
      background: rgba(244, 63, 94, 0.2);
      color: #fda4af;
      border: 1px solid rgba(244, 63, 94, 0.3);
      padding: 0.2rem 0.6rem;
      border-radius: 0.4rem;
      font-size: 0.75rem;
      font-weight: 700;
    }
    .tag-real {
      background: rgba(16, 185, 129, 0.2);
      color: #6ee7b7;
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 0.2rem 0.6rem;
      border-radius: 0.4rem;
      font-size: 0.75rem;
      font-weight: 700;
    }
    .meter {
      height: 6px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 3px;
      overflow: hidden;
      margin-top: 0.4rem;
    }
    .meter-fill { height: 100%; border-radius: 3px; }
    .meta-text {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 0.5rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-family: 'JetBrains Mono', monospace;
    }
    .spinner {
      display: none;
      text-align: center;
      margin: 2rem 0;
      color: var(--accent-cyan);
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge">Local MLOps Pipeline</div>
      <h1>AI-Generated Image Detector</h1>
      <p class="subtitle">Extracts web images, inspects CLIP embeddings & FFT artifacts, classifies Real vs Synthetic</p>
    </header>

    <div class="input-card">
      <div class="input-group">
        <input type="url" id="targetUrl" placeholder="Enter webpage URL (e.g., https://unsplash.com or https://lexica.art)" required />
        <button class="btn-primary" id="analyzeBtn" onclick="runAnalysis()">Analyze Webpage</button>
      </div>
    </div>

    <div class="spinner" id="loadingSpinner">
      <p>Scraping webpage, deduplicating images, and running CLIP inference...</p>
    </div>

    <div class="stats-bar" id="statsBar">
      <div class="stat-card">
        <div class="stat-label">Total Images</div>
        <div class="stat-value" id="statTotal">0</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">AI-Generated</div>
        <div class="stat-value" id="statAI" style="color: var(--accent-rose)">0</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Authentic (Real)</div>
        <div class="stat-value" id="statReal" style="color: var(--accent-green)">0</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">AI Ratio</div>
        <div class="stat-value" id="statRatio" style="color: var(--accent-cyan)">0%</div>
      </div>
    </div>

    <div class="gallery" id="gallery"></div>
  </div>

  <script>
    async function runAnalysis() {
      const urlInput = document.getElementById("targetUrl");
      const btn = document.getElementById("analyzeBtn");
      const spinner = document.getElementById("loadingSpinner");
      const statsBar = document.getElementById("statsBar");
      const gallery = document.getElementById("gallery");

      const url = urlInput.value.trim();
      if (!url) return alert("Please enter a valid webpage URL.");

      btn.disabled = true;
      spinner.style.display = "block";
      gallery.innerHTML = "";
      statsBar.style.display = "none";

      try {
        const resp = await fetch("/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: url, max_images: 30 })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || "Analysis failed");

        // Populate stats
        document.getElementById("statTotal").textContent = data.total_images_analyzed;
        document.getElementById("statAI").textContent = data.ai_generated_count;
        document.getElementById("statReal").textContent = data.real_count;
        document.getElementById("statRatio").textContent = data.ai_percentage + "%";
        statsBar.style.display = "grid";

        // Render cards
        data.results.forEach(item => {
          const isAI = item.label === "AI_GENERATED";
          const card = document.createElement("div");
          card.className = `image-card ${isAI ? 'ai' : 'real'}`;
          
          const meterColor = isAI ? 'var(--accent-rose)' : 'var(--accent-green)';
          const pct = Math.round(item.confidence * 100);

          card.innerHTML = `
            <img class="image-preview" src="${item.preview_url || item.image_url}" alt="Preview" loading="lazy" />
            <div class="card-body">
              <div class="card-label">
                <span class="${isAI ? 'tag-ai' : 'tag-real'}">${isAI ? 'AI GENERATED' : 'AUTHENTIC REAL'}</span>
                <span style="font-weight: 600; font-size: 0.85rem;">${pct}%</span>
              </div>
              <div class="meter">
                <div class="meter-fill" style="width: ${pct}%; background: ${meterColor};"></div>
              </div>
              <div class="meta-text" title="${item.image_url}">${item.image_url}</div>
            </div>
          `;
          gallery.appendChild(card);
        });

      } catch (err) {
        alert("Error: " + err.message);
      } finally {
        btn.disabled = false;
        spinner.style.display = "none";
      }
    }
  </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)
