"""
AINA Translator Microservice
=============================
Real-time translation from Catalan to multiple languages using
Projecte AINA models from Hugging Face.

Supported targets: en, es, fr, pt, it, de
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("aina-translator")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, str] = {
    "en": "projecte-aina/aina-translator-ca-en",
    "es": "projecte-aina/aina-translator-ca-es",
    "fr": "projecte-aina/aina-translator-ca-fr",
    "pt": "projecte-aina/aina-translator-ca-pt",
    "it": "projecte-aina/aina-translator-ca-it",
    "de": "projecte-aina/aina-translator-ca-de",
}

# Runtime stores (populated on startup)
tokenizers: dict[str, AutoTokenizer] = {}
models: dict[str, AutoModelForSeq2SeqLM] = {}

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
HAS_GPU = torch.cuda.is_available()
DEVICE = "cuda" if HAS_GPU else "cpu"
DTYPE = torch.float16 if HAS_GPU else torch.float32

logger.info(f"Device: {DEVICE} | dtype: {DTYPE} | CUDA available: {HAS_GPU}")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models() -> None:
    """Download and load all translation models into memory."""
    logger.info("Starting model loading …")
    t0 = time.time()

    for lang, model_name in MODEL_REGISTRY.items():
        logger.info(f"  Loading {lang} ← {model_name}")
        t1 = time.time()

        tok = AutoTokenizer.from_pretrained(model_name)

        if HAS_GPU:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=DTYPE,
                device_map="auto",
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model = model.to("cpu")

        model.eval()

        tokenizers[lang] = tok
        models[lang] = model

        logger.info(f"  ✓ {lang} loaded in {time.time() - t1:.1f}s")

    logger.info(f"All models loaded in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# FastAPI lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    # Cleanup (optional: free GPU memory)
    models.clear()
    tokenizers.clear()
    if HAS_GPU:
        torch.cuda.empty_cache()


app = FastAPI(
    title="AINA Translator",
    description="Catalan → multi-language translation using Projecte AINA models",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Catalan source text")
    targets: list[str] = Field(
        ...,
        description="List of target language codes (en, es, fr, pt, it, de)",
    )
    max_new_tokens: int = Field(
        default=256,
        ge=1,
        le=1024,
        description="Maximum tokens to generate per translation",
    )
    num_beams: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Beam search width (1 = greedy, higher = better quality but slower)",
    )


class TranslateResponse(BaseModel):
    """Keys are language codes; values are translated strings or null."""
    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Translation logic
# ---------------------------------------------------------------------------

def translate_single(
    text: str,
    lang: str,
    max_new_tokens: int = 256,
    num_beams: int = 1,
) -> Optional[str]:
    """Translate *text* from Catalan to *lang*. Returns None if lang unknown."""
    if lang not in models:
        return None

    tok = tokenizers[lang]
    model = models[lang]

    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    translation = tok.decode(outputs[0], skip_special_tokens=True)
    return translation


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    """Translate Catalan text to one or more target languages."""
    results: dict[str, Optional[str]] = {}

    for lang in req.targets:
        lang_lower = lang.strip().lower()
        results[lang_lower] = translate_single(
            req.text,
            lang_lower,
            max_new_tokens=req.max_new_tokens,
            num_beams=req.num_beams,
        )

    return results


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "models_loaded": list(models.keys()),
    }


@app.get("/")
async def root():
    return {
        "service": "AINA Translator",
        "docs": "/docs",
        "health": "/health",
    }
