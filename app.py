import os
import time
import numpy as np
import onnxruntime as ort
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from engine import ZeroDayInferenceEngine

# Define schema schemas to guarantee payload contract integrity
class TelemetryPayload(BaseModel):
    packet_features: List[float] = Field(..., description="256-dimensional single packet feature array")
    flow_features: List[List[float]] = Field(..., description="100x256 sequential flow feature matrix")
    campaign_features: List[float] = Field(..., description="1024-dimensional contextual campaign vector")

class DetectionResponse(BaseModel):
    status: str
    processing_time_ms: float
    closest_known_class: int
    anomaly_score: float
    is_zero_day_candidate: bool
    raw_distances: dict

# Initialize FastAPI App
def load_artifacts():
    global SESSION, INFERENCE_ENGINE
    onnx_model_path = "malzda_encoder.onnx"
    
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"Critical Error: Optimized graph '{onnx_model_path}' missing. Run export_onnx.py first.")
    if ort is None:
        raise RuntimeError("onnxruntime is not available. Install onnxruntime to run the inference service.")
        
    print("Spawning optimized ONNX Runtime CPU Inference Session...")
    # Configure thread execution settings tailored for single-node CPU architecture
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 2
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    SESSION = ort.InferenceSession(onnx_model_path, opts, providers=['CPUExecutionProvider'])
    INFERENCE_ENGINE = ZeroDayInferenceEngine()
    print("Microservice warm-up sequence complete. Awaiting packet stream ingestion.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load artifacts on startup
    load_artifacts()
    yield
    # Cleanup on shutdown (if needed)
    pass

app = FastAPI(
    title="MAL-ZDA Threat Detection Engine",
    description="Cloud-Native Edge Inference Microservice for Zero-Day Network Anomaly Processing",
    version="1.0.0",
    lifespan=lifespan
)

# Global variables loaded on startup to eliminate runtime latency
SESSION: Optional[Any] = None
INFERENCE_ENGINE: Optional[Any] = None

@app.post("/api/v1/detect", response_model=DetectionResponse, status_code=status.HTTP_200_OK)
async def process_telemetry(payload: TelemetryPayload):
    start_time = time.perf_counter()
    
    try:
        # Convert incoming JSON arrays to structured NumPy matrices and force float32 alignment
        packet_np = np.array([payload.packet_features], dtype=np.float32)
        flow_np = np.array([payload.flow_features], dtype=np.float32)
        campaign_np = np.array([payload.campaign_features], dtype=np.float32)
        
        # Validate exact runtime dimensions match exported model architecture signatures
        if packet_np.shape != (1, 256) or flow_np.shape != (1, 100, 256) or campaign_np.shape != (1, 1024):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Dimension mismatch. Expected shapes: packet (1,256), flow (1,100,256), campaign (1,1024). Got: packet {packet_np.shape}, flow {flow_np.shape}, campaign {campaign_np.shape}"
            )
        
        # Execute forward evaluation via ONNX runtime engine
        inputs = {
            'packet_input': packet_np,
            'flow_input': flow_np,
            'campaign_input': campaign_np
        }
        
        outputs = SESSION.run(['fused_embedding'], inputs)  # type: ignore
        # Isolate first array element in batch output. Handle possible sparse or non-indexable return types
        out0 = outputs[0]
        if hasattr(out0, '__getitem__'):
            # array-like objects that support indexing
            fused_embedding = out0[0]  # type: ignore
        else:
            # Some runtimes may return sparse-like or custom tensor objects without __getitem__.
            # Coerce to a NumPy array then index.
            fused_arr = np.asarray(out0)
            if fused_arr.ndim == 0:
                # Fallback: try converting to dense if it's a 0-d object containing array-like
                fused_arr = np.asarray(fused_arr.tolist())
            fused_embedding = fused_arr[0]
        
        # Pass the extracted abstract feature embedding to the Prototype distance classifier
        evaluation = INFERENCE_ENGINE.compute_anomaly_score(fused_embedding)  # type: ignore
        
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        
        return {
            "status": "success",
            "processing_time_ms": round(duration_ms, 3),
            "closest_known_class": evaluation["closest_known_class"],
            "anomaly_score": evaluation["anomaly_score"],
            "is_zero_day_candidate": evaluation["is_zero_day_candidate"],
            "raw_distances": evaluation["raw_distances"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution fault: {str(e)}"
        )