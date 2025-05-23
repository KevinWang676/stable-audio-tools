from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import tempfile
import os
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request model
class AudioGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text description of the audio to generate")
    seconds_total: float = Field(..., gt=0, le=120, description="Duration of the audio in seconds (max 120s)")
    steps: Optional[int] = Field(100, ge=10, le=200, description="Number of diffusion steps (default: 100)")
    cfg_scale: Optional[float] = Field(7.0, ge=1.0, le=20.0, description="Classifier-free guidance scale (default: 7.0)")
    sigma_min: Optional[float] = Field(0.3, ge=0.1, le=1.0, description="Minimum noise level (default: 0.3)")
    sigma_max: Optional[float] = Field(500.0, ge=100.0, le=1000.0, description="Maximum noise level (default: 500.0)")
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "128 BPM tech house drum loop",
                "seconds_total": 30.0,
                "steps": 100,
                "cfg_scale": 7.0,
                "sigma_min": 0.3,
                "sigma_max": 500.0
            }
        }

# Global variables for model
model = None
model_config = None
sample_rate = None
sample_size = None
device = None

app = FastAPI(
    title="Text-to-Audio Generation API v2",
    description="Generate high-quality audio from text prompts using Stable Audio Open 1.0",
    version="2.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Load the model once at startup"""
    global model, model_config, sample_rate, sample_size, device
    
    logger.info("Starting up the application...")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Download and load model (stable-audio-open-1.0)
        logger.info("Loading Stable Audio Open 1.0 model...")
        model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]
        model = model.to(device)
        
        logger.info(f"Model loaded successfully. Sample rate: {sample_rate}, Sample size: {sample_size}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e

@app.post("/generate-audio", response_class=FileResponse)
async def generate_audio(request: AudioGenerationRequest, background_tasks: BackgroundTasks):
    """
    Generate high-quality audio from text prompt and duration
    
    - **prompt**: Text description of the audio to generate
    - **seconds_total**: Duration of the audio in seconds (max 120s)
    - **steps**: Number of diffusion steps (default: 100, higher = better quality but slower)
    - **cfg_scale**: Classifier-free guidance scale (default: 7.0, higher = more prompt adherence)
    - **sigma_min**: Minimum noise level (default: 0.3)
    - **sigma_max**: Maximum noise level (default: 500.0)
    
    Returns a high-quality WAV audio file
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Generating audio: '{request.prompt}' for {request.seconds_total} seconds")
        logger.info(f"Parameters - Steps: {request.steps}, CFG: {request.cfg_scale}, Sigma: {request.sigma_min}-{request.sigma_max}")
        
        # Set up text and timing conditioning with seconds_start
        conditioning = [{
            "prompt": request.prompt,
            "seconds_start": 0,
            "seconds_total": request.seconds_total
        }]
        
        # Generate stereo audio with enhanced parameters
        output = generate_diffusion_cond(
            model,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=request.sigma_min,
            sigma_max=request.sigma_max,
            sampler_type="dpmpp-3m-sde",
            device=device
        )
        
        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")
        
        # Peak normalize, clip, convert to int16
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        
        # Clear CUDA cache as requested
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            torchaudio.save(tmp_file.name, output, sample_rate)
            temp_filepath = tmp_file.name
        
        logger.info(f"High-quality audio generated successfully, saved to: {temp_filepath}")
        
        # Schedule cleanup of temporary file after response is sent
        background_tasks.add_task(cleanup_temp_file, temp_filepath)
        
        # Return the file with descriptive filename
        filename = f"stable_audio_{int(request.seconds_total)}s_steps{request.steps}.wav"
        return FileResponse(
            temp_filepath,
            media_type="audio/wav",
            filename=filename
        )
        
    except Exception as e:
        # Clear CUDA cache even on error
        torch.cuda.empty_cache()
        logger.error(f"Audio generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

@app.post("/generate-audio-basic", response_class=FileResponse)
async def generate_audio_basic(background_tasks: BackgroundTasks, prompt: str, seconds_total: float):
    """
    Simple endpoint with just prompt and duration (uses default parameters)
    
    - **prompt**: Text description of the audio to generate
    - **seconds_total**: Duration of the audio in seconds
    """
    # Create request with default parameters
    request = AudioGenerationRequest(
        prompt=prompt,
        seconds_total=seconds_total
    )
    
    return await generate_audio(request, background_tasks)

def cleanup_temp_file(filepath: str):
    """Background task to clean up temporary file after response is sent"""
    try:
        os.unlink(filepath)
        logger.info(f"Cleaned up temporary file: {filepath}")
    except OSError as e:
        logger.warning(f"Failed to clean up temporary file {filepath}: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Text-to-Audio Generation API v2 - Stable Audio Open 1.0",
        "model": "stabilityai/stable-audio-open-1.0",
        "features": [
            "High-quality audio generation",
            "Advanced diffusion parameters",
            "DPMPP-3M-SDE sampler",
            "Classifier-free guidance"
        ],
        "endpoints": {
            "generate": "/generate-audio",
            "generate_basic": "/generate-audio-basic",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "stabilityai/stable-audio-open-1.0",
        "device": device,
        "model_loaded": model is not None,
        "sample_rate": sample_rate,
        "sample_size": sample_size,
        "cuda_available": torch.cuda.is_available(),
        "cuda_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    }

@app.get("/model-info")
async def model_info():
    """Get detailed information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "stabilityai/stable-audio-open-1.0",
        "model_config": model_config,
        "device": device,
        "sample_rate": sample_rate,
        "sample_size": sample_size,
        "default_parameters": {
            "steps": 100,
            "cfg_scale": 7.0,
            "sigma_min": 0.3,
            "sigma_max": 500.0,
            "sampler_type": "dpmpp-3m-sde"
        }
    }

@app.get("/generation-presets")
async def generation_presets():
    """Get recommended parameter presets for different quality/speed tradeoffs"""
    return {
        "fast": {
            "description": "Fast generation with good quality",
            "steps": 50,
            "cfg_scale": 5.0,
            "sigma_min": 0.3,
            "sigma_max": 500.0
        },
        "balanced": {
            "description": "Balanced quality and speed (default)",
            "steps": 100,
            "cfg_scale": 7.0,
            "sigma_min": 0.3,
            "sigma_max": 500.0
        },
        "high_quality": {
            "description": "High quality generation (slower)",
            "steps": 150,
            "cfg_scale": 9.0,
            "sigma_min": 0.2,
            "sigma_max": 500.0
        },
        "experimental": {
            "description": "Experimental settings for creative results",
            "steps": 100,
            "cfg_scale": 12.0,
            "sigma_min": 0.1,
            "sigma_max": 800.0
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
