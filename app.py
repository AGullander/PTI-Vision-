from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Tuple, Literal
from copy import deepcopy

import cv2
import numpy as np
import subprocess
import tempfile
import os
import re
import json
import base64
import requests
import time
from pathlib import Path
from uuid import uuid4
from PIL import Image  # för LLaVA (BGR->PIL)

from ultralytics import YOLO  # AI-modell för objekt

# Load .env file BEFORE reading environment variables
from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------
# Grund-path
# -------------------------------------------------------

ROOT = Path(__file__).resolve().parent

# -------------------------------------------------------
# API Key for authentication
# -------------------------------------------------------

API_KEY = os.getenv("API_KEY", "pti_demo_key_2025")  # Change this for production
# Comma-separated origins. Use "*" only for local dev.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

if ALLOWED_ORIGINS.strip() == "*":
    CORS_ORIGINS = ["*"]
else:
    CORS_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]

# Optional kill switch for on-demand training in production
ENABLE_TRAIN_YOLO = os.getenv("ENABLE_TRAIN_YOLO", "0") == "1"


def verify_api_key(api_key: str = Header(None, alias="X-API-Key")):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

# -------------------------------------------------------
# OpenAI / ChatGPT / LLaVA konfiguration
# -------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

# LLaVA – lokal visionmodell (valfri)
LLAVA_MODEL_ID = os.getenv("LLAVA_MODEL_ID", "llava-hf/llava-1.5-7b-hf")

# OpenVINO GenAI – lokal vision och textmodell
# Model paths (convert first with: optimum-cli export openvino --model <model> --weight-format int4 --trust-remote-code <output_dir>)
OPENVINO_VLM_PATH = os.getenv("OPENVINO_VLM_PATH", "")  # e.g. "./models/MiniCPM_V_2_6_ov"
OPENVINO_LLM_PATH = os.getenv("OPENVINO_LLM_PATH", "")  # e.g. "./models/Qwen2.5-3B-Instruct_ov"
OPENVINO_DEVICE = os.getenv("OPENVINO_DEVICE", "CPU")  # CPU, GPU, or NPU

# Global toggles
USE_OPENAI_VISION = os.getenv("USE_OPENAI_VISION", "1") == "1"
USE_OPENAI_TEXT = os.getenv("USE_OPENAI_TEXT", "1") == "1"
USE_LLAVA = os.getenv("USE_LLAVA", "0") == "1"
USE_OPENVINO = os.getenv("USE_OPENVINO", "0") == "1"

# Startup logging for model configuration
print("=" * 60)
print("[STARTUP] Model Configuration:")
print(f"  USE_OPENVINO     = {USE_OPENVINO}")
print(f"  OPENVINO_VLM_PATH = '{OPENVINO_VLM_PATH}'")
print(f"  OPENVINO_LLM_PATH = '{OPENVINO_LLM_PATH}'")
print(f"  OPENVINO_DEVICE   = '{OPENVINO_DEVICE}'")
print(f"  USE_OPENAI_VISION = {USE_OPENAI_VISION}")
print(f"  USE_OPENAI_TEXT   = {USE_OPENAI_TEXT}")
print(f"  USE_LLAVA         = {USE_LLAVA}")
print("=" * 60)

LLAVA_MODEL = None
LLAVA_PROCESSOR = None
LLAVA_DEVICE = "cpu"
LLAVA_LOADED = False
LLAVA_ERROR: Optional[str] = None

# OpenVINO GenAI state
OPENVINO_VLM_PIPELINE = None
OPENVINO_LLM_PIPELINE = None
OPENVINO_VLM_LOADED = False
OPENVINO_LLM_LOADED = False
OPENVINO_ERROR: Optional[str] = None


def _openai_headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }


def call_openai_chat_completions(
    payload: dict,
    timeout: int = 40,
    max_retries: int = 3,
) -> Optional[dict]:
    if not OPENAI_API_KEY:
        print("call_openai_chat_completions: OPENAI_API_KEY saknas")
        return None

    url = f"{OPENAI_API_BASE}/chat/completions"

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                headers=_openai_headers(),
                data=json.dumps(payload),
                timeout=timeout,
            )

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = 1.0
                else:
                    delay = 1.0 * (2 ** attempt)

                print(
                    f"OpenAI 429 – attempt {attempt+1}/{max_retries}, "
                    f"väntar {delay:.1f} s..."
                )
                time.sleep(delay)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.RequestException as e:
            print(f"OpenAI request error ({attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None
            delay = 1.0 * (2 ** attempt)
            time.sleep(delay)

    print("OpenAI: max_retries uppnått.")
    return None


# ---------------- LLaVA-laddning ----------------

def load_llava_model():
    global LLAVA_MODEL, LLAVA_PROCESSOR, LLAVA_DEVICE, LLAVA_LOADED, LLAVA_ERROR

    if LLAVA_LOADED and LLAVA_MODEL is not None and LLAVA_PROCESSOR is not None:
        return LLAVA_MODEL, LLAVA_PROCESSOR

    if not USE_LLAVA:
        LLAVA_ERROR = "USE_LLAVA=0 – laddar inte LLaVA."
        print(LLAVA_ERROR)
        return None, None

    try:
        print(f"Laddar LLaVA-modell: {LLAVA_MODEL_ID}")
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq

        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)
        model = AutoModelForVision2Seq.from_pretrained(
            LLAVA_MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
        model.to(device)

        LLAVA_MODEL = model
        LLAVA_PROCESSOR = processor
        LLAVA_DEVICE = device
        LLAVA_LOADED = True
        LLAVA_ERROR = None

        print(f"LLaVA laddad på device={device}")
        return LLAVA_MODEL, LLAVA_PROCESSOR

    except Exception as e:
        LLAVA_ERROR = str(e)
        print(f"Kunde inte ladda LLaVA-modell: {e}")
        return None, None


# ---------------- OpenVINO GenAI-laddning ----------------

def load_openvino_vlm():
    """
    Load OpenVINO GenAI VLMPipeline for vision-language model inference.
    
    Prerequisites:
      1. pip install openvino-genai
      2. Convert model: optimum-cli export openvino --model openbmb/MiniCPM-V-2_6 --weight-format int4 --trust-remote-code MiniCPM_V_2_6_ov
      3. Set OPENVINO_VLM_PATH to the converted model directory
    """
    global OPENVINO_VLM_PIPELINE, OPENVINO_VLM_LOADED, OPENVINO_ERROR

    print(f"[VLM_LOAD] load_openvino_vlm called: LOADED={OPENVINO_VLM_LOADED}, USE_OPENVINO={USE_OPENVINO}, PATH='{OPENVINO_VLM_PATH}'")

    if OPENVINO_VLM_LOADED and OPENVINO_VLM_PIPELINE is not None:
        print("[VLM_LOAD] Already loaded, returning cached pipeline")
        return OPENVINO_VLM_PIPELINE

    if not USE_OPENVINO:
        OPENVINO_ERROR = "USE_OPENVINO=0 – laddar inte OpenVINO GenAI."
        print(f"[VLM_LOAD] {OPENVINO_ERROR}")
        return None

    if not OPENVINO_VLM_PATH:
        OPENVINO_ERROR = "OPENVINO_VLM_PATH saknas – kan inte ladda VLM."
        print(f"[VLM_LOAD] {OPENVINO_ERROR}")
        return None

    vlm_path = Path(OPENVINO_VLM_PATH)
    if not vlm_path.exists():
        OPENVINO_ERROR = f"VLM-modellsökväg finns inte: {vlm_path.absolute()}"
        print(f"[VLM_LOAD] {OPENVINO_ERROR}")
        return None

    try:
        print(f"[VLM_LOAD] Loading MiniCPM VLM from: {vlm_path.absolute()} on {OPENVINO_DEVICE}")
        import openvino_genai as ov_genai
        print(f"[VLM_LOAD] openvino_genai imported successfully")

        pipeline = ov_genai.VLMPipeline(str(vlm_path), OPENVINO_DEVICE)
        print(f"[VLM_LOAD] VLMPipeline created successfully!")

        OPENVINO_VLM_PIPELINE = pipeline
        OPENVINO_VLM_LOADED = True
        OPENVINO_ERROR = None

        print(f"[VLM_LOAD] ✓ OpenVINO VLM (MiniCPM) loaded on device={OPENVINO_DEVICE}")
        return OPENVINO_VLM_PIPELINE

    except Exception as e:
        OPENVINO_ERROR = str(e)
        print(f"[VLM_LOAD] ✗ Failed to load OpenVINO VLM: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_openvino_llm():
    """
    Load OpenVINO GenAI LLMPipeline for text generation.
    
    Prerequisites:
      1. pip install openvino-genai
      2. Convert model: optimum-cli export openvino --model Qwen/Qwen2.5-3B-Instruct --weight-format int4 --trust-remote-code Qwen2.5-3B-Instruct_ov
      3. Set OPENVINO_LLM_PATH to the converted model directory
    """
    global OPENVINO_LLM_PIPELINE, OPENVINO_LLM_LOADED, OPENVINO_ERROR

    print(f"[LLM_LOAD] load_openvino_llm called: LOADED={OPENVINO_LLM_LOADED}, USE_OPENVINO={USE_OPENVINO}, PATH='{OPENVINO_LLM_PATH}'")

    if OPENVINO_LLM_LOADED and OPENVINO_LLM_PIPELINE is not None:
        print("[LLM_LOAD] Already loaded, returning cached pipeline")
        return OPENVINO_LLM_PIPELINE

    if not USE_OPENVINO:
        OPENVINO_ERROR = "USE_OPENVINO=0 – laddar inte OpenVINO GenAI."
        print(f"[LLM_LOAD] {OPENVINO_ERROR}")
        return None

    if not OPENVINO_LLM_PATH:
        OPENVINO_ERROR = "OPENVINO_LLM_PATH saknas – kan inte ladda LLM."
        print(f"[LLM_LOAD] {OPENVINO_ERROR}")
        return None

    llm_path = Path(OPENVINO_LLM_PATH)
    if not llm_path.exists():
        OPENVINO_ERROR = f"LLM-modellsökväg finns inte: {llm_path.absolute()}"
        print(f"[LLM_LOAD] {OPENVINO_ERROR}")
        return None

    try:
        print(f"[LLM_LOAD] Loading TinyLlama LLM from: {llm_path.absolute()} on {OPENVINO_DEVICE}")
        import openvino_genai as ov_genai
        print(f"[LLM_LOAD] openvino_genai imported successfully")

        pipeline = ov_genai.LLMPipeline(str(llm_path), OPENVINO_DEVICE)
        print(f"[LLM_LOAD] LLMPipeline created successfully!")

        OPENVINO_LLM_PIPELINE = pipeline
        OPENVINO_LLM_LOADED = True
        OPENVINO_ERROR = None

        print(f"[LLM_LOAD] ✓ OpenVINO LLM (TinyLlama) loaded on device={OPENVINO_DEVICE}")
        return OPENVINO_LLM_PIPELINE

    except Exception as e:
        OPENVINO_ERROR = str(e)
        print(f"[LLM_LOAD] ✗ Failed to load OpenVINO LLM: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------- FastAPI-app & CORS ----------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
async def index():
    return FileResponse(ROOT / "index.html")


@app.get("/favicon.ico")
async def favicon():
    icon_path = ROOT / "static" / "pti-logo.png"
    if icon_path.exists():
        return FileResponse(icon_path, media_type="image/png")
    raise HTTPException(404, "favicon not found")


# ---------------- Datamodeller ----------------

class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class Detection(BaseModel):
    label: str
    category: Optional[str] = None
    code: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[float] = None
    bbox: Optional[BBox] = None
    legend: Optional[str] = None


class DiffEntry(BaseModel):
    label: str
    category: Optional[str] = None
    bbox: Optional[BBox] = None


class StageDiff(BaseModel):
    reference_stage: str
    new_findings: List[DiffEntry] = Field(default_factory=list)
    resolved_findings: List[DiffEntry] = Field(default_factory=list)


class StructuralRemark(BaseModel):
    container_id: str
    timestamp: str
    label: str
    category: Optional[str] = None
    defect_type: Literal["dirt", "damage", "scratch", "other"]
    severity: Optional[str] = None
    bbox: Optional[BBox] = None
    contamination_index: int


class AnalyzeResponse(BaseModel):
    container_id: str
    container_type: Optional[str] = None

    status: str
    detections: List[Detection]
    timestamp: str

    people_nearby: bool = False
    door_status: Optional[str] = None
    lock_boxes: List[BBox] = Field(default_factory=list)
    anomalies_present: bool = False

    inspection_stage: Optional[str] = None
    diff: Optional[StageDiff] = None

    scene_tags: List[str] = Field(default_factory=list)
    risk_score: int = 0
    risk_explanations: List[str] = Field(default_factory=list)

    prewash_remarks: List[DiffEntry] = Field(default_factory=list)
    resolved_remarks: List[DiffEntry] = Field(default_factory=list)

    contamination_index: int = 1
    contamination_label: str = "Low"
    contamination_scale: List[bool] = Field(default_factory=list)

    scene_caption: Optional[str] = None
    semantic_people_count: Optional[int] = None

    anomaly_summary: Optional[str] = None
    recommended_actions: List[str] = Field(default_factory=list)

    # Optional debug payload so we can inspect why detections may be empty
    debug_info: Optional[dict] = None


class TrainMarking(BaseModel):
    code: str
    title: str


class SemanticVisionResult(BaseModel):
    scene_caption: Optional[str] = None
    extra_tags: List[str] = Field(default_factory=list)
    extra_risks: List[str] = Field(default_factory=list)
    people_count_estimate: Optional[int] = None
    contamination_hint: Optional[int] = None  # 1–9 eller None


# ---------------- Märknings-träning ----------------

DEFAULT_MARKING_LABELS = {
    "45R1": "45R1 – 40ft High Cube Reefer",
    "42R1": "42R1 – 40ft Reefer",
}

TRAINING_FILE = ROOT / "marking_labels.json"


def load_marking_labels() -> dict:
    labels = DEFAULT_MARKING_LABELS.copy()
    if TRAINING_FILE.exists():
        try:
            with TRAINING_FILE.open("r", encoding="utf-8") as f:
                stored = json.load(f)
                labels.update(stored)
        except Exception:
            pass
    return labels


MARKING_LABELS = load_marking_labels()

CONTAINER_TYPE_DEFINITIONS = {
    "45R1": "40ft High Cube Reefer",
    "42R1": "40ft Reefer",
    "22R1": "20ft Reefer",
    "22G1": "20ft General Purpose",
    "42G1": "40ft General Purpose",
    "45G1": "40ft High Cube Dry",
    "L5R1": "Reefer Special",
}

# ---------------- Visuell minnesbank ----------------

VISUAL_MEMORY_FILE = ROOT / "visual_memory.json"


def load_visual_memory() -> list:
    if VISUAL_MEMORY_FILE.exists():
        try:
            with VISUAL_MEMORY_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return []


VISUAL_MEMORY = load_visual_memory()


def save_visual_memory() -> None:
    try:
        with VISUAL_MEMORY_FILE.open("w", encoding="utf-8") as f:
            json.dump(VISUAL_MEMORY, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def compute_feature(patch_bgr: np.ndarray) -> List[float]:
    patch = cv2.resize(patch_bgr, (64, 64), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)

    hist_b = cv2.calcHist([patch], [0], None, [16], [0, 256])
    hist_g = cv2.calcHist([patch], [1], None, [16], [0, 256])
    hist_r = cv2.calcHist([patch], [2], None, [16], [0, 256])
    hist = np.concatenate([hist_b, hist_g, hist_r]).flatten()

    vec = np.concatenate([l.astype("float32").flatten(), hist.astype("float32")])
    vec /= (np.linalg.norm(vec) + 1e-6)
    return vec.tolist()


def best_match_label(patch_bgr: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
    if not VISUAL_MEMORY:
        return None, None

    feat = np.array(compute_feature(patch_bgr), dtype=np.float32)

    best_label = None
    best_dist: Optional[float] = None

    for entry in VISUAL_MEMORY:
        label = entry.get("label")
        if not label:
            continue

        feats_raw = entry.get("features")
        if feats_raw:
            for mvec_list in feats_raw:
                if not mvec_list:
                    continue
                mvec = np.array(mvec_list, dtype=np.float32)
                if mvec.shape != feat.shape:
                    continue
                d = float(np.linalg.norm(feat - mvec))
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_label = label
        else:
            mvec_list = entry.get("feature")
            if not mvec_list:
                continue
            mvec = np.array(mvec_list, dtype=np.float32)
            if mvec.shape != feat.shape:
                continue
            d = float(np.linalg.norm(feat - mvec))
            if best_dist is None or d < best_dist:
                best_dist = d
                best_label = label

    if best_label is None or best_dist is None:
        return None, None

    threshold = 50.0
    if best_dist > threshold:
        return None, None

    confidence = max(0.0, min(1.0, 1.0 - best_dist / threshold))
    return best_label, confidence


# ---------------- Kunskapsbas ----------------

KNOWLEDGE_BASE_FILE = ROOT / "knowledge_base.json"


def load_knowledge_base() -> dict:
    kb = {"classes": {}, "rules": []}
    if KNOWLEDGE_BASE_FILE.exists():
        try:
            with KNOWLEDGE_BASE_FILE.open("r", encoding="utf-8") as f:
                kb_file = json.load(f)
                if isinstance(kb_file, dict):
                    kb.update(kb_file)
        except Exception as e:
            print("Kunde inte ladda knowledge_base.json:", e)
    else:
        print("knowledge_base.json saknas – tom KB")
    return kb


KNOWLEDGE_BASE = load_knowledge_base()


def kb_lookup_class(raw_name: str) -> tuple[str, str, int]:
    info = KNOWLEDGE_BASE.get("classes", {}).get(raw_name, {})
    display_name = info.get("display_name", raw_name)
    semantic_category = info.get("category", "object")
    risk_weight = int(info.get("risk_weight", 0))
    return display_name, semantic_category, risk_weight


# ---------------- YOLO ----------------


def dbg(msg: str) -> None:
    """Lightweight debug logger for analysis pipeline."""
    print(f"[ANALYZE_DEBUG] {msg}")

def normalize_label_name(label: str) -> str:
    """
    Returnerar label som den är - ingen översättning.
    Kan användas för framtida normalisering om behövs.
    """
    return label


def load_generic_model() -> Optional[YOLO]:
    try:
        print("Laddar GENERIC YOLO-modell: yolov8n.pt")
        return YOLO("yolov8n.pt")
    except Exception as e:
        print("Kunde inte ladda yolov8n.pt:", e)
        return None


def load_container_model() -> Optional[YOLO]:
    try:
        if (ROOT / "pti_best.pt").exists():
            print("Laddar CONTAINER YOLO-modell: pti_best.pt")
            return YOLO(str(ROOT / "pti_best.pt"))
        else:
            print("pti_best.pt saknas – ingen separat container-modell")
            return None
    except Exception as e:
        print("Kunde inte ladda pti_best.pt:", e)
        return None


GENERIC_MODEL = load_generic_model()
CONTAINER_MODEL = load_container_model()

# Grundkänslighet – sänkt för att fånga fler defekter
MIN_CONF = 0.25
MIN_BOX_AREA_RATIO = 0.00005


def get_thresholds(damage_sensitivity: str) -> Tuple[float, float]:
    """
    Returnerar (min_conf, min_box_area_ratio) beroende på vald känslighet.
    'high' = mest känslig, 'low' = minst känslig.
    NOTE: Thresholds lowered significantly for under-trained custom model.
    """
    s = (damage_sensitivity or "medium").lower()
    if s == "high":
        # mest aggressivt: mycket låg conf för att fånga svaga detektioner
        return 0.05, 0.00001
    if s == "low":
        # mer konservativt
        return 0.35, 0.0001
    # medium - sänkt markant för undertränad modell
    return 0.10, 0.00002


# COCO classes that are irrelevant for container inspection
# These are filtered out from the generic YOLO model to avoid false positives
IRRELEVANT_COCO_CLASSES = {
    "train", "airplane", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "chair", "bus", "motorcycle", "bicycle", "boat"
}


def run_yolo_model(
    model: YOLO,
    frame_bgr: np.ndarray,
    scale_to: Optional[Tuple[int, int]],
    source_name: str,
    min_conf: float,
    min_box_area_ratio: float,
) -> List[Detection]:
    dbg(f"run_yolo_model(source={source_name}) start, frame={frame_bgr.shape}, "
        f"min_conf={min_conf}, min_box_area_ratio={min_box_area_ratio}")

    results = model(frame_bgr, verbose=False)
    if not results:
        dbg(f"run_yolo_model(source={source_name}): no results object")
        return []

    r = results[0]
    if not hasattr(r, "boxes") or r.boxes is None:
        dbg(f"run_yolo_model(source={source_name}): results[0] has no boxes")
        return []

    # Debug: show raw box count before any filtering
    raw_box_count = len(r.boxes) if r.boxes is not None else 0
    if raw_box_count > 0:
        confs = [float(b.conf[0]) for b in r.boxes if b.conf is not None and len(b.conf) > 0]
        if confs:
            dbg(f"run_yolo_model(source={source_name}): raw_boxes={raw_box_count}, "
                f"conf_range=[{min(confs):.3f}, {max(confs):.3f}]")
        else:
            dbg(f"run_yolo_model(source={source_name}): raw_boxes={raw_box_count}, no valid confs")
    else:
        dbg(f"run_yolo_model(source={source_name}): raw_boxes=0")

    h_src, w_src, _ = frame_bgr.shape

    if scale_to is not None:
        w_out, h_out = scale_to
        scale_x = w_out / float(w_src)
        scale_y = h_out / float(h_src)
    else:
        w_out, h_out = w_src, h_src
        scale_x = scale_y = 1.0

    detections: List[Detection] = []

    for box in r.boxes:
        if box.conf is None or len(box.conf) == 0:
            continue
        conf = float(box.conf[0])
        if conf < min_conf:
            continue

        if box.cls is None or len(box.cls) == 0:
            continue
        cls_id = int(box.cls[0])

        if hasattr(model, "names"):
            raw_name = model.names.get(cls_id, str(cls_id))
        elif hasattr(model.model, "names"):
            raw_name = model.model.names.get(cls_id, str(cls_id))
        else:
            raw_name = str(cls_id)

        # Filter out irrelevant COCO classes for generic model
        if source_name == "generic" and raw_name.lower() in IRRELEVANT_COCO_CLASSES:
            dbg(f"run_yolo_model: SKIPPING irrelevant class '{raw_name}' (conf={conf:.2f})")
            continue

        display_name, semantic_category, _ = kb_lookup_class(raw_name)
        display_name = normalize_label_name(display_name)

        xyxy = box.xyxy[0].cpu().numpy()
        x1_s, y1_s, x2_s, y2_s = xyxy

        x1 = int(x1_s * scale_x)
        y1 = int(y1_s * scale_y)
        x2 = int(x2_s * scale_x)
        y2 = int(y2_s * scale_y)

        x1 = max(0, min(x1, w_out - 1))
        y1 = max(0, min(y1, h_out - 1))
        x2 = max(0, min(x2, w_out))
        y2 = max(0, min(y2, h_out))

        if x2 <= x1 or y2 <= y1:
            continue

        box_area = float((x2 - x1) * (y2 - y1))
        if box_area / float(w_out * h_out) < min_box_area_ratio:
            continue

        bbox = BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
        legend = f"{display_name} ({conf*100:.1f}%)"

        detections.append(
            Detection(
                label=display_name,
                category=semantic_category,
                confidence=conf,
                bbox=bbox,
                legend=legend,
            )
        )

    dbg(
        f"run_yolo_model(source={source_name}) finished with "
        f"{len(detections)} detections (after thresholds)"
    )

    return detections


def detect_objects_yolo(
    frame_bgr: np.ndarray,
    scale_to: Optional[Tuple[int, int]] = None,
    damage_sensitivity: str = "medium",
) -> List[Detection]:
    dbg(
        f"detect_objects_yolo: frame={frame_bgr.shape}, "
        f"damage_sensitivity={damage_sensitivity}"
    )

    detections: List[Detection] = []
    min_conf, min_area = get_thresholds(damage_sensitivity)

    if GENERIC_MODEL is not None:
        dbg("detect_objects_yolo: using GENERIC_MODEL")
        det_gen = run_yolo_model(
            GENERIC_MODEL,
            frame_bgr,
            scale_to=scale_to,
            source_name="generic",
            min_conf=min_conf,
            min_box_area_ratio=min_area,
        )
        dbg(f"detect_objects_yolo: GENERIC_MODEL produced {len(det_gen)} detections")
        detections.extend(det_gen)
    else:
        dbg("detect_objects_yolo: GENERIC_MODEL is None – YOLOv8n not loaded")

    if CONTAINER_MODEL is not None:
        dbg("detect_objects_yolo: using CONTAINER_MODEL")
        det_cont = run_yolo_model(
            CONTAINER_MODEL,
            frame_bgr,
            scale_to=scale_to,
            source_name="container",
            min_conf=min_conf,
            min_box_area_ratio=min_area,
        )
        dbg(
            f"detect_objects_yolo: CONTAINER_MODEL produced {len(det_cont)} detections"
        )
        detections.extend(det_cont)
    else:
        dbg("detect_objects_yolo: CONTAINER_MODEL is None – pti_best.pt not loaded")

    dbg(f"detect_objects_yolo: total detections before further filtering: {len(detections)}")
    return detections


# ---------------- LAB-baserad dark-spot-detektering (strikt) ----------------

def detect_dark_spots(
    frame_bgr: np.ndarray,
    spot_mode: str = "auto",
    roi_margin_ratio: float = 0.01,
) -> List[Detection]:
    """
    Detects dark spots on light backgrounds - typically mold, dirt, or contamination.

    FOCUS:
      - Clearly dark/black spots
      - On white/silver/light container backgrounds
      - Limits false positives while catching real contamination

    spot_mode:
      - "auto": standard sensitivity (recommended)
      - "mold_only": stricter, only very dark spots
      - "sensitive": more aggressive, catches more subtle spots
      - "off": disabled
    """
    spot_mode = (spot_mode or "auto").lower()
    if spot_mode == "off":
        return []

    h, w, _ = frame_bgr.shape
    if h == 0 or w == 0:
        return []

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    L = L.astype(np.float32)
    A = A.astype(np.float32)
    B = B.astype(np.float32)

    # Global std for L channel
    _, std_L_mat = cv2.meanStdDev(L)
    std_L = float(std_L_mat[0][0]) + 1e-6

    # Background level - 75th percentile for light background estimation
    flat_L = L.flatten()
    bg_L = float(np.percentile(flat_L, 75))

    # Adaptive background threshold based on mode
    if spot_mode == "sensitive":
        min_bg_L = 140.0  # Accept slightly darker backgrounds
    elif spot_mode == "mold_only":
        min_bg_L = 175.0  # Require very light background
    else:  # auto
        min_bg_L = 145.0  # Slightly more permissive for catching mold

    if bg_L < min_bg_L:
        dbg(f"detect_dark_spots: skipping, bg_L={bg_L:.1f} < {min_bg_L}")
        return []

    # Color deviation (z-score) from global mean
    mean_A, std_A_mat = cv2.meanStdDev(A)
    mean_B, std_B_mat = cv2.meanStdDev(B)
    mean_A = float(mean_A[0][0])
    mean_B = float(mean_B[0][0])
    std_A = float(std_A_mat[0][0]) + 1e-6
    std_B = float(std_B_mat[0][0]) + 1e-6

    Az = (A - mean_A) / std_A
    Bz = (B - mean_B) / std_B
    chroma_z = np.sqrt(Az**2 + Bz**2)

    # Debug: show L channel stats
    dbg(f"detect_dark_spots: bg_L={bg_L:.1f}, std_L={std_L:.1f}, min_L={flat_L.min():.1f}, threshold_L={bg_L - 0.4*std_L:.1f}")

    # Thresholds based on mode
    # Real-world container dirt/stains typically have dark_delta 0.3-0.7
    if spot_mode == "mold_only":
        darker = L < (bg_L - 1.0 * std_L)
        low_chroma = chroma_z < 1.5
        dark_delta_threshold = 1.5  # Only very dark spots (likely mold)
    elif spot_mode == "sensitive":
        darker = L < (bg_L - 0.2 * std_L)  # Very sensitive
        low_chroma = chroma_z < 4.0  # Allow any color
        dark_delta_threshold = 0.3  # Catch even subtle stains
    else:  # auto - based on real container inspection data
        darker = L < (bg_L - 0.3 * std_L)  # Catch light stains
        low_chroma = chroma_z < 3.5  # Allow colored stains (rust, water marks)
        dark_delta_threshold = 0.4  # Real dirt has dark_delta ~0.45-0.65

    mask = np.logical_and(darker, low_chroma)
    
    # Debug: show how many pixels pass the mask
    mask_count = np.sum(mask)
    dbg(f"detect_dark_spots: mask pixels={mask_count} ({100.0*mask_count/(h*w):.2f}%)")

    mask_u8 = np.zeros_like(L, dtype=np.uint8)
    mask_u8[mask] = 255

    # Morphological operations to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)

    margin_x = int(w * roi_margin_ratio)
    margin_y = int(h * roi_margin_ratio)
    roi = mask_u8[margin_y:h - margin_y, margin_x:w - margin_x]

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: List[Detection] = []
    img_area = float(w * h)

    # Size thresholds - slightly more permissive
    if spot_mode == "sensitive":
        min_area_ratio = 0.00006   # ~120 px on 1920x1080
        small_max_ratio = 0.008
        big_max_ratio = 0.15
    elif spot_mode == "mold_only":
        min_area_ratio = 0.00015
        small_max_ratio = 0.005
        big_max_ratio = 0.10
    else:  # auto - optimized for mold detection
        min_area_ratio = 0.00005   # ~100 px on 1920x1080 (catch smaller spots)
        small_max_ratio = 0.008    # Allow slightly larger "small" spots
        big_max_ratio = 0.15       # Allow larger contamination areas

    min_area = img_area * min_area_ratio
    small_max_area = img_area * small_max_ratio
    big_max_area = img_area * big_max_ratio

    # Debug: show contour filtering stats
    dbg(f"detect_dark_spots: num_contours={len(contours)}, min_area={min_area:.0f}, max_area={big_max_area:.0f}")
    rejected_area = 0
    rejected_circ = 0
    rejected_delta = 0
    delta_values: list = []  # Collect (dark_delta, area, pL, x, y, w, h) for analysis

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > big_max_area:
            rejected_area += 1
            continue

        peri = cv2.arcLength(cnt, True)
        circularity = 0.0
        if peri > 0:
            circularity = 4.0 * np.pi * area / (peri * peri)

        is_small = area <= small_max_area

        # Small spots: require some shape coherence but allow elongated shapes (streaks)
        # Lowered from 0.4 to 0.10 to catch streak-like contamination
        if is_small and circularity < 0.10:
            rejected_circ += 1
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        x_full = x + margin_x
        y_full = y + margin_y

        x_full = max(0, min(x_full, w - 1))
        y_full = max(0, min(y_full, h - 1))
        bw = max(1, min(bw, w - x_full))
        bh = max(1, min(bh, h - y_full))

        patch_L = L[y_full:y_full + bh, x_full:x_full + bw]
        if patch_L.size == 0:
            continue

        pL = float(patch_L.mean())

        # How dark is the spot relative to background?
        dark_delta = (bg_L - pL) / (std_L + 1e-6)

        # Collect dark_delta values for analysis
        delta_values.append((dark_delta, area, pL, x_full, y_full, bw, bh))

        if dark_delta < dark_delta_threshold:
            rejected_delta += 1
            continue

        # Confidence scales with darkness
        raw_score = max(0.0, min(10.0, dark_delta))
        conf = 0.45 + 0.05 * raw_score
        conf = max(0.45, min(0.99, conf))

        bbox = BBox(x=x_full, y=y_full, w=bw, h=bh)

        # Categorize based on size and darkness
        if dark_delta > 4.0 or area > img_area * 0.01:
            label = "Mold/Dark Spot"
        else:
            label = "Dark spot"

        detections.append(
            Detection(
                label=label,
                category="dirt",
                confidence=conf,
                bbox=bbox,
                legend=f"{label} ({conf*100:.1f}%)",
            )
        )

    # Detailed analysis of rejected contours
    if delta_values:
        delta_values.sort(key=lambda x: x[0], reverse=True)  # Sort by dark_delta descending
        passed = [d for d in delta_values if d[0] >= dark_delta_threshold]
        rejected = [d for d in delta_values if d[0] < dark_delta_threshold]
        
        if rejected:
            # Show distribution of rejected dark_delta values
            rejected_deltas = [d[0] for d in rejected]
            dbg(f"detect_dark_spots: REJECTED contours dark_delta range: "
                f"[{min(rejected_deltas):.2f}, {max(rejected_deltas):.2f}], "
                f"mean={sum(rejected_deltas)/len(rejected_deltas):.2f}")
            # Show top 5 rejected (closest to threshold)
            top_rejected = sorted(rejected, key=lambda x: x[0], reverse=True)[:5]
            for i, (delta, area, pL, x, y, bw, bh) in enumerate(top_rejected):
                dbg(f"  Rejected #{i+1}: dark_delta={delta:.2f}, pL={pL:.1f}, "
                    f"area={area:.0f}, pos=({x},{y}), size={bw}x{bh}")
        
        if passed:
            passed_deltas = [d[0] for d in passed]
            dbg(f"detect_dark_spots: PASSED contours dark_delta range: "
                f"[{min(passed_deltas):.2f}, {max(passed_deltas):.2f}]")

    dbg(f"detect_dark_spots: rejected={rejected_area} by area, {rejected_circ} by circularity, "
        f"{rejected_delta} by dark_delta<{dark_delta_threshold}")

    dbg(
        f"detect_dark_spots: found {len(detections)} detections "
        f"(mode={spot_mode}, bg_L={bg_L:.1f}, img={w}x{h})"
    )

    return detections


# ---------------- Defektklassning & risk ----------------

def classify_defect(det: Detection) -> str:
    """
    Classifies detection into: dirt, damage, scratch, obstruction, or other.
    Recognizes both English and Swedish labels.
    """
    cat = (det.category or "").lower()
    lbl = (det.label or "").lower()

    # 1. Check category first (from knowledge base)
    if cat in ("dirt", "smuts", "contamination"):
        return "dirt"

    if cat in ("damage", "structural_damage", "dent", "hole", "crack"):
        return "damage"

    if cat in ("obstruction",):
        return "obstruction"

    # 2. Check label for dirt-related terms (Swedish + English)
    dirt_terms = [
        "dark spot", "mold", "mould", "mögel",
        "smuts", "dirty", "dirt", "stain",
        "missfärgning", "discoloration", "fläck",
        "contamination", "smutsig", "oren"
    ]
    if any(word in lbl for word in dirt_terms):
        return "dirt"

    # 3. Check label for damage-related terms (Swedish + English)
    damage_terms = [
        "dent", "buckla", "buckled", "buckling",
        "hole", "hål", "crack", "spricka",
        "bent", "deformation", "damage", "skada",
        "damagedent", "dent/damage", "rost", "rust",
        "korrosion", "corrosion"
    ]
    if any(word in lbl for word in damage_terms):
        return "damage"

    # 4. Check for scratch specifically
    scratch_terms = ["scratch", "repa", "repor", "scrape"]
    if any(word in lbl for word in scratch_terms):
        return "scratch"

    # 5. Check for obstruction
    obstruction_terms = [
        "blocked", "blockerad", "obstruction",
        "löst föremål", "loose object", "hinder"
    ]
    if any(word in lbl for word in obstruction_terms):
        return "obstruction"

    return "other"


def is_real_defect(
    det: Detection,
    image_w: int,
    image_h: int,
    min_conf: float = 0.5,
    min_area_ratio: float = 0.0003,
) -> bool:
    """
    Returns True if the detection is a real defect worth reporting:
    - sufficient confidence
    - sufficient area
    - relevant defect type (damage, dirt, scratch, obstruction)
    """
    if det.category in ("marking", "ignore"):
        return False

    if (det.confidence or 0.0) < min_conf:
        return False

    if det.bbox is None:
        return False

    area = det.bbox.w * det.bbox.h
    img_area = float(image_w * image_h)
    if area / img_area < min_area_ratio:
        return False

    defect_type = classify_defect(det)
    return defect_type in ("damage", "dirt", "scratch", "obstruction")


def keep_for_output(det: Detection, image_w: int, image_h: int) -> bool:
    """
    Determines if a detection should be kept for anomaly output.
    Keeps dirt, damage, scratch, and obstruction detections.
    Uses lower thresholds to avoid missing real defects.
    """
    defect_type = classify_defect(det)
    
    # Keep dirt, damage, scratch, and obstruction
    if defect_type not in ("dirt", "damage", "scratch", "obstruction"):
        return False

    # Lower confidence thresholds to catch more real defects
    if defect_type == "damage":
        min_conf = 0.20  # Very low - damage is critical
        min_area_ratio = 0.00003
    elif defect_type == "dirt":
        min_conf = 0.25
        min_area_ratio = 0.00004
    elif defect_type == "scratch":
        min_conf = 0.25
        min_area_ratio = 0.00003
    else:  # obstruction
        min_conf = 0.30
        min_area_ratio = 0.0001

    return is_real_defect(
        det,
        image_w=image_w,
        image_h=image_h,
        min_conf=min_conf,
        min_area_ratio=min_area_ratio,
    )


def summarize_people_nearby(dets: List[Detection]) -> bool:
    return any(d.category == "human" for d in dets)


def summarize_door_status(dets: List[Detection]) -> Optional[str]:
    has_open = any(d.category == "door_open" for d in dets)
    has_closed = any(d.category == "door_closed" for d in dets)

    if has_open:
        return "open"
    if has_closed:
        return "closed"
    return None


def extract_lock_boxes(dets: List[Detection]) -> List[BBox]:
    locks: List[BBox] = []
    for d in dets:
        if d.category == "lock" and d.bbox is not None:
            locks.append(d.bbox)
    return locks


def interpret_scene_with_kb(dets: List[Detection]) -> tuple[List[str], int, List[str]]:
    tags: set[str] = set()
    risk_score = 0
    explanations: List[str] = []

    for d in dets:
        cat = (d.category or "object").lower()
        tags.add(cat)
        kb_classes = KNOWLEDGE_BASE.get("classes", {})
        risk_weight = 0
        for raw_name, info in kb_classes.items():
            if info.get("display_name") == d.label:
                risk_weight = int(info.get("risk_weight", 0))
                break
        if risk_weight > 0:
            risk_score += risk_weight
            explanations.append(f"{d.label}: +{risk_weight} risk")

    for rule in KNOWLEDGE_BASE.get("rules", []):
        req = set(rule.get("required_categories", []))
        if req.issubset(tags):
            delta = int(rule.get("risk_delta", 0))
            risk_score += delta
            explanations.append(f"Regel '{rule.get('description')}': +{delta} risk")
            tags.add(rule.get("id", ""))

    return sorted(list(tags)), risk_score, explanations


def compute_contamination_index(
    dets: List[Detection],
    risk_score: int
) -> tuple[int, str]:
    """
    Beräknar ett index 1–9 där både smuts/mögel ("dirt") och skador väger in.
    Mörka/mögelliknande fläckar får något högre vikt än vanlig smuts.
    """
    raw = 0.0
    damage_count = 0
    dark_spot_like = 0

    for d in dets:
        defect = classify_defect(d)
        label_low = (d.label or "").lower()

        if defect == "dirt":
            # Basvikt för "dirt"
            base = 1.0
            # Mörka/mögelliknande fläckar väger lite tyngre
            if any(w in label_low for w in ["dark spot", "mold", "mould", "mögel"]):
                base += 0.5
                dark_spot_like += 1
            raw += base
        elif defect == "damage":
            raw += 2.0
            damage_count += 1

    if damage_count > 1:
        raw += (damage_count - 1) * 1.0

    # Extra litet påslag om flera mörka fläckar hittats
    if dark_spot_like > 1:
        raw += 0.5

    # Riskscore ger en mjuk extra höjning
    raw += risk_score / 4.0

    idx = int(round(raw)) + 1
    idx = max(1, min(9, idx))

    if idx <= 3:
        label = "Low"
    elif idx <= 6:
        label = "Medium"
    else:
        label = "High"

    return idx, label


# ---------------- OCR / Container-ID ----------------

def run_tesseract_text(frame_bgr: np.ndarray) -> str:
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        scale = 2.0
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        variants = []

        _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(th_otsu)

        th_adapt = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10
        )
        variants.append(th_adapt)

        texts = []

        for th in variants:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, th)

            result = subprocess.run(
                [
                    "tesseract",
                    tmp_path,
                    "stdout",
                    "--psm",
                    "6",
                    "--oem",
                    "3",
                    "-c",
                    "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            texts.append(result.stdout)

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return "\n".join(texts)
    except FileNotFoundError:
        return ""


def parse_container_info(text: str) -> Tuple[str, List[str]]:
    cleaned = text.upper()
    normalized = re.sub(r"[^A-Z0-9\s]", " ", cleaned)

    container_id = "UNKNOWN"

    for match in re.finditer(r"([A-Z]{4})\s*([0-9\s]{6,12})", normalized, flags=re.MULTILINE):
        owner = match.group(1)
        digits_raw = match.group(2)
        digits = re.sub(r"\s+", "", digits_raw)

        if len(digits) >= 7:
            digits = digits[:7]
            candidate = owner + digits
            if len(candidate) == 11:
                container_id = candidate
                break

    codes = set()
    for match in re.finditer(r"\b[0-9]{2}[A-Z0-9]{2}\b", normalized):
        codes.add(match.group(0))

    return container_id, sorted(codes)


def enhanced_id_ocr(frame_bgr: np.ndarray) -> Tuple[str, List[str]]:
    h, w, _ = frame_bgr.shape

    crops = [
        frame_bgr[0:int(0.5 * h), int(0.5 * w):w],
        frame_bgr[0:int(0.4 * h), int(0.25 * w):int(0.75 * w)],
        frame_bgr[int(0.2 * h):int(0.7 * h), int(0.5 * w):w],
        frame_bgr,
    ]

    id_counts: dict[str, int] = {}
    code_counts: dict[str, int] = {}
    all_codes: set[str] = set()

    for crop in crops:
        if crop is None or crop.size == 0:
            continue
        t = run_tesseract_text(crop)
        cid, codes = parse_container_info(t)

        if cid != "UNKNOWN":
            id_counts[cid] = id_counts.get(cid, 0) + 1

        for c in codes:
            all_codes.add(c)
            code_counts[c] = code_counts.get(c, 0) + 1

    best_id = "UNKNOWN"
    if id_counts:
        best_id = max(id_counts.items(), key=lambda kv: kv[1])[0]

    codes_sorted = sorted(
        list(all_codes),
        key=lambda c: (-code_counts.get(c, 0), c)
    )

    return best_id, codes_sorted


def ocr_on_markings(frame_bgr: np.ndarray, dets: List[Detection]) -> Tuple[str, List[str]]:
    texts = []

    for d in dets:
        if d.category != "marking" or d.bbox is None:
            continue

        x, y, w, h = d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h

        pad = int(0.1 * max(w, h))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(frame_bgr.shape[1], x + w + pad)
        y1 = min(frame_bgr.shape[0], y + h + pad)

        crop = frame_bgr[y0:y1, x0:x1]
        if crop is None or crop.size == 0:
            continue

        t = run_tesseract_text(crop)
        if t:
            texts.append(t)

    if not texts:
        return "UNKNOWN", []

    combined = "\n".join(texts)
    return parse_container_info(combined)


# ---------------- Strukturella anmärkningar ----------------

STRUCTURAL_LOG_FILE = ROOT / "structural_remarks.jsonl"


def extract_structural_remarks(
    container_id: str,
    dets: List[Detection],
    contamination_index: int,
) -> List[StructuralRemark]:
    ts = datetime.utcnow().isoformat() + "Z"
    remarks: List[StructuralRemark] = []

    for d in dets:
        defect = classify_defect(d)
        if defect != "damage":
            continue

        remarks.append(
            StructuralRemark(
                container_id=container_id,
                timestamp=ts,
                label=d.label,
                category=d.category,
                defect_type="damage",
                severity=d.severity,
                bbox=d.bbox,
                contamination_index=contamination_index,
            )
        )

    return remarks


def log_structural_remarks(remarks: List[StructuralRemark]) -> None:
    if not remarks:
        return

    try:
        with STRUCTURAL_LOG_FILE.open("a", encoding="utf-8") as f:
            for r in remarks:
                f.write(r.json(ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Failed to log structural remarks: {e}")


# ---------------- Semantisk vision (OpenAI & LLaVA) ----------------

def _semantic_vision_openai(frame_bgr: np.ndarray) -> Optional[SemanticVisionResult]:
    if not OPENAI_API_KEY or not USE_OPENAI_VISION:
        print("Semantic vision (OpenAI) INAKTIV")
        return None

    try:
        ok, buf = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return None

        b64_img = base64.b64encode(buf.tobytes()).decode("utf-8")

        system_prompt = (
            "Du är en expert på inspektion av sjöcontainrar och säkerhet.\n"
            "Du får en bild och ska fokusera på anomalier och risker.\n"
            "Gör följande:\n"
            "- Beskriv scenen kort (1–2 meningar),\n"
            "- Identifiera smuts, skador, repor, läckage, hinder eller andra avvikelser,\n"
            "- Uppskatta en kontaminationsnivå 1–9 (1 = ren, 9 = mycket smutsig),\n"
            "- Uppmärksamma risker för personskada, fallrisk, öppna dörrar, personer på farlig plats,\n"
            "- Uppskatta hur många personer som syns.\n"
            "Svara ENDAST som JSON:\n"
            "{\n"
            '  \"scene_caption\": str eller null,\n'
            '  \"extra_tags\": [str],\n'
            '  \"extra_risks\": [str],\n'
            '  \"people_count_estimate\": int eller null,\n'
            '  \"contamination_hint\": int (1–9) eller null\n'
            "}"
        )

        user_text = (
            "Analysera bilden av containern. Fokusera på smuts, skador, repor, "
            "människor i närheten, hinder och säkerhetsrisker. Svara bara med JSON."
        )

        payload = {
            "model": OPENAI_VISION_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_img}"
                            },
                        },
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
        }

        data = call_openai_chat_completions(payload, timeout=40, max_retries=3)
        if data is None:
            return None

        content = data["choices"][0]["message"]["content"]
        print("Semantic vision (OpenAI) raw:", content)
        parsed = json.loads(content)

        hint = parsed.get("contamination_hint")
        if isinstance(hint, (int, float)):
            hint_int = int(round(hint))
            if 1 <= hint_int <= 9:
                contamination_hint = hint_int
            else:
                contamination_hint = None
        else:
            contamination_hint = None

        return SemanticVisionResult(
            scene_caption=parsed.get("scene_caption"),
            extra_tags=parsed.get("extra_tags") or [],
            extra_risks=parsed.get("extra_risks") or [],
            people_count_estimate=parsed.get("people_count_estimate"),
            contamination_hint=contamination_hint,
        )
    except Exception as e:
        print(f"Semantic vision (OpenAI) error: {e}")
        return None


def _semantic_vision_llava(frame_bgr: np.ndarray) -> Optional[SemanticVisionResult]:
    if not USE_LLAVA:
        print("Semantic vision (LLaVA) INAKTIV")
        return None

    model, processor = load_llava_model()
    if model is None or processor is None:
        return None

    try:
        import torch

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        instructions = (
            "Du är en expert på inspektion av sjöcontainrar och säkerhet.\n"
            "Identifiera smuts, skador, repor, läckage, hinder, människor och risker.\n"
            "Uppskatta kontamineringsnivå 1–9 (1=ren, 9=mycket smutsig).\n"
            "Svara ENDAST som JSON med nycklar:\n"
            "{\n"
            '  \"scene_caption\", \"extra_tags\", \"extra_risks\", '
            '  \"people_count_estimate\", \"contamination_hint\"\n'
            "}"
        )

        prompt = f"USER: {instructions}\nASSISTANT:"

        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(LLAVA_DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.1,
            )

        output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        print("Semantic vision (LLaVA) raw:", output_text)

        m = re.search(r"\{.*\}", output_text, re.S)
        if not m:
            return None

        json_str = m.group(0)
        parsed = json.loads(json_str)

        hint = parsed.get("contamination_hint")
        if isinstance(hint, (int, float)):
            hint_int = int(round(hint))
            if 1 <= hint_int <= 9:
                contamination_hint = hint_int
            else:
                contamination_hint = None
        else:
            contamination_hint = None

        return SemanticVisionResult(
            scene_caption=parsed.get("scene_caption"),
            extra_tags=parsed.get("extra_tags") or [],
            extra_risks=parsed.get("extra_risks") or [],
            people_count_estimate=parsed.get("people_count_estimate"),
            contamination_hint=contamination_hint,
        )

    except Exception as e:
        print(f"Semantic vision (LLaVA) error: {e}")
        return None


def _semantic_vision_openvino(frame_bgr: np.ndarray) -> Optional[SemanticVisionResult]:
    """
    Use OpenVINO GenAI VLMPipeline for vision-language analysis.
    
    Supports models like MiniCPM-V-2_6, LLaVA, Qwen2-VL, etc.
    """
    if not USE_OPENVINO:
        print("Semantic vision (OpenVINO) INAKTIV")
        return None

    pipeline = load_openvino_vlm()
    if pipeline is None:
        print("Semantic vision (OpenVINO): pipeline is None")
        return None

    try:
        import openvino as ov
        import openvino_genai as ov_genai

        # Convert BGR to RGB and ensure uint8
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Ensure the image is in HWC format with uint8 dtype
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        
        # Make sure the array is contiguous
        rgb = np.ascontiguousarray(rgb)
        
        print(f"Semantic vision (OpenVINO): image shape={rgb.shape}, dtype={rgb.dtype}")
        
        # Convert numpy array to OpenVINO Tensor
        # VLMPipeline expects ov.Tensor with shape [H, W, C] in RGB format
        image_tensor = ov.Tensor(rgb)
        print(f"Semantic vision (OpenVINO): tensor created, shape={image_tensor.shape}")

        # MiniCPM-V-2_6 uses the image tag <image>./</image>\n
        prompt = (
            "<image>./</image>\n"
            "You are an expert at inspecting shipping containers and safety. "
            "Analyze this image and identify: "
            "1) Dirt, damage, scratches, leaks or other anomalies. "
            "2) People nearby. "
            "3) Safety risks. "
            "Respond ONLY as JSON with format: "
            '{"scene_caption": "description", "extra_tags": ["tag1", "tag2"], '
            '"extra_risks": ["risk1"], "people_count_estimate": 0, "contamination_hint": 1}'
        )

        print(f"Semantic vision (OpenVINO): calling generate with prompt length={len(prompt)}")

        # VLMPipeline.generate() takes prompt and images parameter
        result = pipeline.generate(prompt, images=[image_tensor], max_new_tokens=512)
        output_text = str(result)
        
        print("Semantic vision (OpenVINO) raw:", output_text)

        # Extract JSON from response
        m = re.search(r"\{.*\}", output_text, re.S)
        if not m:
            # Try to construct a basic result from the text
            return SemanticVisionResult(
                scene_caption=output_text[:200] if output_text else None,
                extra_tags=[],
                extra_risks=[],
                people_count_estimate=None,
                contamination_hint=None,
            )

        json_str = m.group(0)
        parsed = json.loads(json_str)

        hint = parsed.get("contamination_hint")
        if isinstance(hint, (int, float)):
            hint_int = int(round(hint))
            if 1 <= hint_int <= 9:
                contamination_hint = hint_int
            else:
                contamination_hint = None
        else:
            contamination_hint = None

        return SemanticVisionResult(
            scene_caption=parsed.get("scene_caption"),
            extra_tags=parsed.get("extra_tags") or [],
            extra_risks=parsed.get("extra_risks") or [],
            people_count_estimate=parsed.get("people_count_estimate"),
            contamination_hint=contamination_hint,
        )

    except Exception as e:
        print(f"Semantic vision (OpenVINO) error: {e}")
        return None


def semantic_vision_analyze(
    frame_bgr: np.ndarray,
    backend: str = "auto",
) -> Optional[SemanticVisionResult]:
    backend = (backend or "auto").lower()
    dbg(f"semantic_vision_analyze: backend={backend}, USE_OPENVINO={USE_OPENVINO}, VLM_PATH={OPENVINO_VLM_PATH}")

    if backend == "none":
        return None

    if backend == "openai":
        return _semantic_vision_openai(frame_bgr)

    if backend == "llava":
        return _semantic_vision_llava(frame_bgr)

    if backend == "openvino":
        return _semantic_vision_openvino(frame_bgr)

    # Auto mode: try backends in order of preference
    if USE_OPENVINO and OPENVINO_VLM_PATH:
        dbg("semantic_vision_analyze: trying OpenVINO VLM...")
        res = _semantic_vision_openvino(frame_bgr)
        if res is not None:
            dbg("semantic_vision_analyze: OpenVINO VLM success!")
            return res
        dbg("semantic_vision_analyze: OpenVINO VLM returned None")

    if USE_OPENAI_VISION and OPENAI_API_KEY:
        res = _semantic_vision_openai(frame_bgr)
        if res is not None:
            return res

    if USE_LLAVA:
        return _semantic_vision_llava(frame_bgr)

    dbg("semantic_vision_analyze: no backend available")
    return None


# ---------------- Textbaserad analys (OpenAI, OpenVINO) ----------------

def _semantic_reasoning_openai(resp: AnalyzeResponse) -> Tuple[Optional[str], List[str]]:
    """OpenAI-based text reasoning from detections."""
    if not OPENAI_API_KEY or not USE_OPENAI_TEXT:
        print("Semantic reasoning (OpenAI) INAKTIV")
        return None, []

    try:
        payload_obj = {
            "container_id": resp.container_id,
            "container_type": resp.container_type,
            "status": resp.status,
            "detections": [
                {
                    "label": d.label,
                    "category": d.category,
                    "severity": d.severity,
                    "bbox": d.bbox.dict() if d.bbox else None,
                }
                for d in resp.detections
            ],
            "risk_score": resp.risk_score,
            "risk_explanations": resp.risk_explanations,
            "scene_tags": resp.scene_tags,
            "contamination_index": resp.contamination_index,
            "contamination_label": resp.contamination_label,
            "scene_caption": resp.scene_caption,
            "semantic_people_count": resp.semantic_people_count,
            "inspection_stage": resp.inspection_stage,
        }

        system_prompt = (
            "Du är en teknisk inspektionsassistent för sjöcontainrar.\n"
            "Du får strukturerad data från ett detekteringssystem.\n"
            "Du ska INTE hitta på egna objekt, bara resonera utifrån den data du får.\n"
            "Svara ENDAST som JSON:\n"
            "{ \"anomaly_summary\": str, \"recommended_actions\": [str] }"
        )

        user_prompt = (
            "Här är data från en container-inspektion. Analysera avvikelser och risker.\n\n"
            + json.dumps(payload_obj, ensure_ascii=False, indent=2)
        )

        body = {
            "model": OPENAI_TEXT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        data = call_openai_chat_completions(body, timeout=40, max_retries=3)
        if data is None:
            return None, []

        content = data["choices"][0]["message"]["content"]
        print("Semantic reasoning (OpenAI) raw:", content)

        parsed = json.loads(content)
        anomaly_summary = parsed.get("anomaly_summary")
        actions = parsed.get("recommended_actions") or []

        if not isinstance(actions, list):
            actions = []

        return anomaly_summary, actions

    except Exception as e:
        print(f"Semantic reasoning (OpenAI) error: {e}")
        return None, []


def _semantic_reasoning_openvino(resp: AnalyzeResponse) -> Tuple[Optional[str], List[str]]:
    """OpenVINO GenAI LLM-based text reasoning from detections."""
    if not USE_OPENVINO:
        print("Semantic reasoning (OpenVINO) INAKTIV")
        return None, []

    pipeline = load_openvino_llm()
    if pipeline is None:
        return None, []

    try:
        payload_obj = {
            "container_id": resp.container_id,
            "container_type": resp.container_type,
            "status": resp.status,
            "detections": [
                {
                    "label": d.label,
                    "category": d.category,
                    "severity": d.severity,
                }
                for d in resp.detections
            ],
            "risk_score": resp.risk_score,
            "contamination_index": resp.contamination_index,
            "contamination_label": resp.contamination_label,
            "scene_caption": resp.scene_caption,
        }

        prompt = (
            "Du är en teknisk inspektionsassistent för sjöcontainrar. "
            "Analysera följande inspektionsdata och ge en sammanfattning av avvikelser "
            "och rekommenderade åtgärder. Svara ENDAST som JSON med formatet: "
            '{"anomaly_summary": "beskrivning", "recommended_actions": ["åtgärd1", "åtgärd2"]}\n\n'
            f"Inspektionsdata:\n{json.dumps(payload_obj, ensure_ascii=False, indent=2)}"
        )

        # Configure generation
        config = pipeline.get_generation_config()
        config.max_new_tokens = 512
        config.temperature = 0.1

        result = pipeline.generate(prompt, config)
        output_text = str(result)
        
        print("Semantic reasoning (OpenVINO) raw:", output_text)

        # Extract JSON from response
        m = re.search(r"\{.*\}", output_text, re.S)
        if not m:
            return output_text[:500] if output_text else None, []

        json_str = m.group(0)
        parsed = json.loads(json_str)
        anomaly_summary = parsed.get("anomaly_summary")
        actions = parsed.get("recommended_actions") or []

        if not isinstance(actions, list):
            actions = []

        return anomaly_summary, actions

    except Exception as e:
        print(f"Semantic reasoning (OpenVINO) error: {e}")
        return None, []


def semantic_reasoning_from_detections(
    resp: AnalyzeResponse,
    backend: str = "auto",
) -> Tuple[Optional[str], List[str]]:
    """
    Analyze detections and generate reasoning/recommendations.
    
    backend options: "auto", "openai", "openvino"
    """
    backend = (backend or "auto").lower()

    if backend == "openai":
        return _semantic_reasoning_openai(resp)

    if backend == "openvino":
        return _semantic_reasoning_openvino(resp)

    # Auto mode: try backends in order of preference
    if USE_OPENVINO and OPENVINO_LLM_PATH:
        result = _semantic_reasoning_openvino(resp)
        if result[0] is not None:
            return result

    if USE_OPENAI_TEXT and OPENAI_API_KEY:
        return _semantic_reasoning_openai(resp)

    print("Semantic reasoning: no backend available")
    return None, []


# ---------------- Video-sammanfattning ----------------

def video_summary_with_gpt(results: List[AnalyzeResponse], backend: str = "auto") -> str:
    """
    Generate a summary of video analysis results.
    
    backend options: "auto", "openai", "openvino"
    """
    backend = (backend or "auto").lower()

    condensed = [
        {
            "frame_index": i,
            "container_id": r.container_id,
            "status": r.status,
            "scene_caption": r.scene_caption,
            "risk_score": r.risk_score,
            "contamination_index": r.contamination_index,
            "contamination_label": r.contamination_label,
            "people_nearby": r.people_nearby,
            "detections": [d.label for d in r.detections],
        }
        for i, r in enumerate(results)
    ]

    prompt_text = (
        "Du är en expert som sammanfattar anomalier och risker från en videobaserad "
        "inspektion av sjöcontainrar. Här är analysdata från videon, frame för frame:\n\n"
        + json.dumps(condensed, ensure_ascii=False, indent=2)
    )

    # Try OpenVINO first if configured
    if backend == "openvino" or (backend == "auto" and USE_OPENVINO and OPENVINO_LLM_PATH):
        pipeline = load_openvino_llm()
        if pipeline is not None:
            try:
                config = pipeline.get_generation_config()
                config.max_new_tokens = 1024
                config.temperature = 0.3

                result = pipeline.generate(prompt_text, config)
                output_text = str(result)
                print("Video summary (OpenVINO) raw:", output_text)
                return output_text
            except Exception as e:
                print(f"Video summary (OpenVINO) error: {e}")
                if backend == "openvino":
                    return f"OpenVINO-fel: {e}"

    # Fall back to OpenAI
    if backend == "openai" or backend == "auto":
        if not OPENAI_API_KEY or not USE_OPENAI_TEXT:
            return "Sammanfattning inaktiv (ingen backend tillgänglig)."

        system_prompt = (
            "Du är en expert som sammanfattar anomalier och risker från en videobaserad "
            "inspektion av sjöcontainrar."
        )

        user_prompt = (
            "Här är analysdata från videon, frame för frame:\n\n"
            + json.dumps(condensed, ensure_ascii=False, indent=2)
        )

        body = {
            "model": OPENAI_TEXT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        data = call_openai_chat_completions(body, timeout=60, max_retries=3)
        if data is None:
            err = "GPT-fel i video_summary_with_gpt."
            print(err)
            return err

        content = data["choices"][0]["message"]["content"]
        print("Video summary (OpenAI) raw:", content)
        return content

    return "Sammanfattning inaktiv (ingen backend tillgänglig)."


# ---------------- Pre/Post-logik ----------------

LAST_CONTAINER_ID: Optional[str] = None
LAST_CONTAINER_TYPE: Optional[str] = None
LAST_CODES: List[str] = []
OCR_CALL_COUNTER: int = 0

INSPECTION_SNAPSHOTS: dict[str, dict[str, AnalyzeResponse]] = {}


def detection_key(d: Detection) -> str:
    if d.bbox is None:
        return d.label
    rx = (d.bbox.x // 10) * 10
    ry = (d.bbox.y // 10) * 10
    rw = (d.bbox.w // 10) * 10
    rh = (d.bbox.h // 10) * 10
    return f"{d.label}|{rx}|{ry}|{rw}|{rh}"


def compute_stage_diff(current: AnalyzeResponse, other: AnalyzeResponse, reference_stage: str) -> StageDiff:
    def filtered(dets: List[Detection]) -> List[Detection]:
        return [d for d in dets if d.category not in ("marking", "ignore")]

    cur_dets = filtered(current.detections)
    oth_dets = filtered(other.detections)

    cur_map = {detection_key(d): d for d in cur_dets}
    oth_map = {detection_key(d): d for d in oth_dets}

    new_keys = set(cur_map.keys()) - set(oth_map.keys())
    resolved_keys = set(oth_map.keys()) - set(cur_map.keys())

    diff = StageDiff(reference_stage=reference_stage)

    for k in new_keys:
        d = cur_map[k]
        diff.new_findings.append(DiffEntry(label=d.label, category=d.category, bbox=d.bbox))
    for k in resolved_keys:
        d = oth_map[k]
        diff.resolved_findings.append(DiffEntry(label=d.label, category=d.category, bbox=d.bbox))

    return diff


def extract_prewash_remarks(pre_resp: AnalyzeResponse) -> List[DiffEntry]:
    remarks: List[DiffEntry] = []
    for d in pre_resp.detections:
        if d.category in ("marking", "ignore"):
            continue
        remarks.append(DiffEntry(label=d.label, category=d.category, bbox=d.bbox))
    return remarks


def refine_with_visual_memory(frame_bgr: np.ndarray, detections: List[Detection]) -> None:
    if not VISUAL_MEMORY:
        return

    h, w, _ = frame_bgr.shape

    for det in detections:
        if not det.bbox:
            continue

        x, y, bw, bh = det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        if x >= x2 or y >= y2:
            continue

        patch = frame_bgr[y:y2, x:x2]
        if patch is None or patch.size == 0:
            continue

        vlabel, vconf = best_match_label(patch)
        if not vlabel:
            continue

        if vlabel == "__IGNORE__":
            if (vconf or 0) > 0.6:
                det.label = "__IGNORE__"
                det.category = "ignore"
                det.legend = "Ignorerad"
                det.confidence = max(det.confidence or 0, vconf or 0)
            continue

        vlabel = normalize_label_name(vlabel)

        yolo_conf = det.confidence or 0.0
        mem_conf = vconf or 0.0

        if yolo_conf < 0.6 and mem_conf > 0.7:
            det.label = vlabel
            det.confidence = max(yolo_conf, mem_conf)

            low = vlabel.lower()
            if low.startswith("damage"):
                det.category = "damage"
            elif vlabel in ("Smuts", "Löst föremål", "Missfärgning"):
                det.category = (
                    "dirt" if vlabel == "Smuts"
                    else "loose_object" if vlabel == "Löst föremål"
                    else "discoloration"
                )
            elif "människa" in low or "person" in low:
                det.category = "human"

            if det.legend is None:
                det.legend = det.label


def analyze_frame_bytes(
    image_bytes: bytes,
    damage_sensitivity: str = "medium",
    inspection_stage: Optional[Literal["pre", "post"]] = None,
    enable_vision_gpt: bool = True,
    enable_text_gpt: bool = True,
    vision_backend: str = "auto",
    spot_mode: str = "auto",
) -> AnalyzeResponse:
    global LAST_CONTAINER_ID, LAST_CONTAINER_TYPE, LAST_CODES, OCR_CALL_COUNTER

    arr = np.frombuffer(image_bytes, np.uint8)
    frame_full = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame_full is None:
        raise HTTPException(400, "Kunde inte läsa bilddata")

    h_full, w_full, _ = frame_full.shape
    dbg(
        f"analyze_frame_bytes: received image_bytes={len(image_bytes)}, "
        f"decoded frame size={w_full}x{h_full}, "
        f"damage_sensitivity={damage_sensitivity}, "
        f"inspection_stage={inspection_stage}, "
        f"vision_backend={vision_backend}, spot_mode={spot_mode}, "
        f"enable_vision_gpt={enable_vision_gpt}, enable_text_gpt={enable_text_gpt}"
    )

    MAX_YOLO_WIDTH = 960
    if w_full > MAX_YOLO_WIDTH:
        scale = MAX_YOLO_WIDTH / float(w_full)
        new_w = int(w_full * scale)
        new_h = int(h_full * scale)
        frame_small = cv2.resize(frame_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        frame_small = frame_full

    # YOLO-detektioner
    det_objects = detect_objects_yolo(
        frame_small,
        scale_to=(w_full, h_full),
        damage_sensitivity=damage_sensitivity,
    )

    detections: List[Detection] = list(det_objects)
    yolo_raw_count = len(det_objects)
    dbg(f"analyze_frame_bytes: YOLO detections (raw)={yolo_raw_count}")

    # Dark-spot-detektering (strikt)
    dark_spot_count = 0
    try:
        dark_spot_dets = detect_dark_spots(frame_full, spot_mode=spot_mode)
        if dark_spot_dets:
            detections.extend(dark_spot_dets)
            dark_spot_count = len(dark_spot_dets)
        dbg(
            f"analyze_frame_bytes: dark-spot detections={dark_spot_count}, "
            f"total_detections_after_dark_spots={len(detections)}"
        )
    except Exception as e:
        print(f"detect_dark_spots error: {e}")

    # Visual memory refinement - runs AFTER dark spots are added
    # so visual memory can filter out both YOLO detections AND dark spots
    refine_with_visual_memory(frame_full, detections)

    container_id = "UNKNOWN"
    codes: List[str] = []

    OCR_CALL_COUNTER += 1
    run_ocr_now = (LAST_CONTAINER_ID is None) or (OCR_CALL_COUNTER % 5 == 0)

    if run_ocr_now:
        cid, ocr_codes = ocr_on_markings(frame_full, detections)

        if cid == "UNKNOWN" and not ocr_codes:
            cid, ocr_codes = enhanced_id_ocr(frame_full)

        container_id = cid
        codes = ocr_codes
        LAST_CODES = codes

        if container_id != "UNKNOWN":
            LAST_CONTAINER_ID = container_id

        container_type_now = None
        for c in codes:
            if c in CONTAINER_TYPE_DEFINITIONS:
                container_type_now = CONTAINER_TYPE_DEFINITIONS[c]
                break
        if container_type_now is not None:
            LAST_CONTAINER_TYPE = container_type_now
    else:
        container_id = LAST_CONTAINER_ID or "UNKNOWN"
        codes = LAST_CODES

    container_type = LAST_CONTAINER_TYPE

    dbg(
        f"analyze_frame_bytes: OCR run_now={run_ocr_now}, "
        f"container_id={container_id}, codes={codes}, "
        f"container_type={container_type}"
    )

    # Lägg till märkningar som detections
    for code in codes:
        label = MARKING_LABELS.get(code, f"Märkning {code}")
        label = normalize_label_name(label)

        if label == "__IGNORE__":
            continue

        legend = f"{label} ({code})"
        detections.append(
            Detection(
                label=label,
                category="marking",
                code=code,
                confidence=0.9,
                legend=legend,
            )
        )

    # 1) Ta bort __IGNORE__
    detections = [d for d in detections if d.label != "__IGNORE__" and d.category != "ignore"]
    dbg(f"analyze_frame_bytes: detections after visual_memory/__IGNORE__ filter={len(detections)}")

    # 2) Spara ALLA detektioner för UI/learning
    all_detections: List[Detection] = list(detections)

    # 3) Filtrera fram "riktiga" anomalier (smuts & bucklor)
    anomaly_detections: List[Detection] = [
        d for d in detections if keep_for_output(d, w_full, h_full)
    ]
    dbg(f"analyze_frame_bytes: Raw detections (for UI)={len(all_detections)}, "
        f"kept anomalies (for risk)={len(anomaly_detections)}")

    # All vidare risk/contamination-analys baseras på anomaly_detections
    scene_tags, risk_score, risk_explanations = interpret_scene_with_kb(anomaly_detections)

    base_contamination_index, base_contamination_label = compute_contamination_index(
        anomaly_detections, risk_score
    )

    sem_result = None
    if enable_vision_gpt:
        dbg(f"analyze_frame_bytes: calling semantic_vision_analyze with backend={vision_backend}")
        sem_result = semantic_vision_analyze(frame_full, backend=vision_backend)
        dbg(f"analyze_frame_bytes: semantic_vision_analyze returned: {sem_result is not None}")

    scene_caption: Optional[str] = None
    semantic_people_count: Optional[int] = None
    contamination_index = base_contamination_index
    contamination_label = base_contamination_label

    if sem_result:
        if sem_result.scene_caption:
            scene_caption = sem_result.scene_caption
            scene_tags.append(f"caption:{sem_result.scene_caption}")

        for t in sem_result.extra_tags:
            if t not in scene_tags:
                scene_tags.append(t)

        for r in sem_result.extra_risks:
            if r not in risk_explanations:
                risk_explanations.append(r)

        semantic_people_count = sem_result.people_count_estimate

        # Create synthetic detections from VLM extra_risks if no YOLO/dark spot detections found
        # This ensures the UI shows something when VLM sees issues but CV methods miss them
        if len(all_detections) == 0 and sem_result.extra_risks:
            dbg(f"analyze_frame_bytes: creating synthetic detections from VLM risks: {sem_result.extra_risks}")
            for risk in sem_result.extra_risks:
                # Map common risks to detection labels
                risk_lower = risk.lower()
                if "leak" in risk_lower:
                    label = "Potential leak (AI)"
                    category = "damage"
                elif "damage" in risk_lower or "wear" in risk_lower:
                    label = "Wear/Damage (AI)"
                    category = "damage"
                elif "contamination" in risk_lower or "dirt" in risk_lower:
                    label = "Contamination (AI)"
                    category = "dirt"
                elif "rust" in risk_lower or "corrosion" in risk_lower:
                    label = "Rust/Corrosion (AI)"
                    category = "damage"
                else:
                    label = f"{risk} (AI)"
                    category = "other"
                
                # Create a detection without specific bbox (center of image as indicator)
                all_detections.append(
                    Detection(
                        label=label,
                        category=category,
                        confidence=0.6,  # Moderate confidence for AI-only detection
                        bbox=BBox(x=w_full//4, y=h_full//4, w=w_full//2, h=h_full//2),  # Center region
                        legend=f"{label} (60%)",
                    )
                )

        if sem_result.contamination_hint is not None:
            combined = (base_contamination_index + sem_result.contamination_hint) / 2.0
            contamination_index = int(round(combined))
            contamination_index = max(1, min(9, contamination_index))
            if contamination_index <= 3:
                contamination_label = "Low"
            elif contamination_index <= 6:
                contamination_label = "Medium"
            else:
                contamination_label = "High"

    anomalies_present = any(
        classify_defect(d) in ("damage", "dirt")
        for d in anomaly_detections
    )

    has_damage = any(
        classify_defect(d) == "damage"
        for d in anomaly_detections
    )

    # Mindre nervös statuslogik, men låter mögel/mörka fläckar och högre index väga tyngre
    status = "ok"
    if has_damage:
        status = "alert"
    elif anomalies_present:
        # Gränsen för alert sänks något och påverkas både av index och risk_score
        if contamination_index >= 4 or risk_score >= 5:
            status = "alert"

    # Build a compact debug payload that explains why detections may be empty
    debug_payload = {
        "image_size": {"width": w_full, "height": h_full},
        "models": {
            "generic_loaded": GENERIC_MODEL is not None,
            "container_loaded": CONTAINER_MODEL is not None,
        },
        "counts": {
            "yolo_raw": yolo_raw_count,
            "dark_spots": dark_spot_count,
            "all_detections_after_filters": len(all_detections),
            "anomaly_detections": len(anomaly_detections),
        },
        "container": {
            "id": container_id,
            "type": container_type,
            "codes": codes,
        },
        "status_logic": {
            "anomalies_present": anomalies_present,
            "has_damage": has_damage,
            "risk_score": risk_score,
            "contamination_index": contamination_index,
            "contamination_label": contamination_label,
        },
        "pipeline": {
            "inspection_stage": inspection_stage,
            "vision_backend": vision_backend,
            "enable_vision_gpt": enable_vision_gpt,
            "enable_text_gpt": enable_text_gpt,
            "spot_mode": spot_mode,
        },
    }

    # Närvaro av personer, dörrstatus, lås m.m. – använd ALLA detektioner
    people_nearby = summarize_people_nearby(all_detections)
    door_status = summarize_door_status(all_detections)
    lock_boxes = extract_lock_boxes(all_detections)

    contamination_scale = [i < contamination_index for i in range(9)]

    resp = AnalyzeResponse(
        container_id=container_id,
        container_type=container_type,
        status=status,
        detections=all_detections,  # UI ser ALLA boxar
        timestamp=datetime.utcnow().isoformat() + "Z",
        people_nearby=people_nearby,
        door_status=door_status,
        lock_boxes=lock_boxes,
        anomalies_present=anomalies_present,
        inspection_stage=inspection_stage,
        diff=None,
        scene_tags=scene_tags,
        risk_score=risk_score,
        risk_explanations=risk_explanations,
        prewash_remarks=[],
        resolved_remarks=[],
        contamination_index=contamination_index,
        contamination_label=contamination_label,
        contamination_scale=contamination_scale,
        scene_caption=scene_caption,
        semantic_people_count=semantic_people_count,
        anomaly_summary=None,
        recommended_actions=[],
        debug_info=debug_payload,
    )

    # Pre/Post-logik bygger på ALLA detektioner, men StageDiff filtrerar bort "marking"/"ignore"
    if inspection_stage in ("pre", "post") and container_id != "UNKNOWN":
        stages = INSPECTION_SNAPSHOTS.setdefault(container_id, {})
        stages[inspection_stage] = deepcopy(resp)

        other_stage = "post" if inspection_stage == "pre" else "pre"

        if other_stage in stages:
            other = stages[other_stage]
            resp.diff = compute_stage_diff(resp, other, reference_stage=other_stage)

        pre_snapshot = stages.get("pre")
        if pre_snapshot is not None:
            resp.prewash_remarks = extract_prewash_remarks(pre_snapshot)

        if inspection_stage == "post" and resp.diff is not None:
            if resp.diff.reference_stage == "pre":
                resp.resolved_remarks = resp.diff.resolved_findings

    # Logga bara riktiga skador (inte alla boxar)
    structural_remarks = extract_structural_remarks(
        container_id=container_id,
        dets=anomaly_detections,
        contamination_index=contamination_index,
    )
    log_structural_remarks(structural_remarks)

    # GPT/LLM-resonemang (får ALLA detektioner i payloaden)
    # Check if any text backend is available
    has_openai_text = OPENAI_API_KEY and USE_OPENAI_TEXT
    has_openvino_text = USE_OPENVINO and OPENVINO_LLM_PATH
    
    if enable_text_gpt and (has_openai_text or has_openvino_text):
        dbg(f"analyze_frame_bytes: calling semantic_reasoning (OpenAI={has_openai_text}, OpenVINO={has_openvino_text})")
        anomaly_summary, actions = semantic_reasoning_from_detections(resp)
        if anomaly_summary is None and not actions:
            resp.anomaly_summary = "LLM-analysen är tillfälligt otillgänglig."
            resp.recommended_actions = []
        else:
            resp.anomaly_summary = anomaly_summary
            resp.recommended_actions = actions
    else:
        dbg(f"analyze_frame_bytes: skipping text reasoning (enable_text_gpt={enable_text_gpt})")
        resp.anomaly_summary = None
        resp.recommended_actions = []

    return resp


# ---------------- YOLO-träning / logg ----------------

YOLO_TRAIN_ROOT = ROOT / "yolo_train_data"
YOLO_IMAGES_DIR = YOLO_TRAIN_ROOT / "images"
YOLO_META_FILE = YOLO_TRAIN_ROOT / "annotations.jsonl"

YOLO_TRAIN_ROOT.mkdir(exist_ok=True)
YOLO_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- API-endpoints ----------------

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    api_key: str = Depends(verify_api_key),
    damage_sensitivity: str = "medium",  # default
    inspection_stage: Optional[Literal["pre", "post"]] = None,
    vision_backend: Literal["auto", "openai", "llava", "openvino", "none"] = "auto",
    use_vision_gpt: bool = True,
    use_text_gpt: bool = True,
    # dark spots with sensitive option added
    spot_mode: Literal["auto", "mold_only", "sensitive", "off"] = "auto",
    image: UploadFile = File(...),
):
    data = await image.read()
    print("Uploaded image:", image.filename, image.content_type, "bytes:", len(data))

    # Check if ANY vision backend is available
    has_openvino_vision = USE_OPENVINO and OPENVINO_VLM_PATH
    has_openai_vision = USE_OPENAI_VISION and OPENAI_API_KEY
    has_llava = USE_LLAVA
    
    print(f"[API] Vision backends: OpenVINO={has_openvino_vision}, OpenAI={has_openai_vision}, LLaVA={has_llava}")
    
    # Only disable vision if NO backend is available
    if not has_openvino_vision and not has_openai_vision and not has_llava:
        print("[API] No vision backend available - disabling vision GPT")
        use_vision_gpt = False

    # Check text backend availability (OpenVINO LLM or OpenAI)
    has_openvino_text = USE_OPENVINO and OPENVINO_LLM_PATH
    has_openai_text = USE_OPENAI_TEXT and OPENAI_API_KEY
    
    if not has_openvino_text and not has_openai_text:
        print("[API] No text backend available - disabling text GPT")
        use_text_gpt = False

    # Validate specific backend requests
    if vision_backend == "openai" and not has_openai_vision:
        print("vision_backend=openai men OpenAI ej tillgänglig – sätter 'auto'")
        vision_backend = "auto"
    if vision_backend == "llava" and not has_llava:
        print("vision_backend=llava men USE_LLAVA=0 – sätter 'auto'")
        vision_backend = "auto"
    if vision_backend == "openvino" and not has_openvino_vision:
        print("vision_backend=openvino men OpenVINO VLM ej tillgänglig – sätter 'auto'")
        vision_backend = "auto"

    print(f"[API] Final settings: use_vision_gpt={use_vision_gpt}, use_text_gpt={use_text_gpt}, vision_backend={vision_backend}")

    return analyze_frame_bytes(
        data,
        damage_sensitivity=damage_sensitivity,
        inspection_stage=inspection_stage,
        enable_vision_gpt=use_vision_gpt,
        enable_text_gpt=use_text_gpt,
        vision_backend=vision_backend,
        spot_mode=spot_mode,
    )


@app.post("/api/analyze_video")
async def analyze_video(
    api_key: str = Depends(verify_api_key),
    damage_sensitivity: str = "medium",
    inspection_stage: Optional[Literal["pre", "post"]] = None,
    video: UploadFile = File(...),
):
    raw = await video.read()
    if not raw:
        raise HTTPException(400, "Tom videofil")

    suffix = Path(video.filename).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(raw)
    tmp.close()
    path = tmp.name
    print("Uploaded video:", video.filename, "->", path)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(400, "Kunde inte läsa videofil")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    results: List[AnalyzeResponse] = []
    frame_index = 0

    seconds_between_samples = 2.0
    step = max(1, int(fps * seconds_between_samples))

    MAX_ANALYZED_FRAMES = 40

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % step == 0:
            if len(results) >= MAX_ANALYZED_FRAMES:
                break
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                try:
                    resp = analyze_frame_bytes(
                        buf.tobytes(),
                        damage_sensitivity=damage_sensitivity,
                        inspection_stage=inspection_stage,
                        enable_vision_gpt=False,
                        enable_text_gpt=False,
                        vision_backend="none",
                        spot_mode="auto",
                    )
                    results.append(resp)
                except HTTPException as e:
                    print(f"Frame {frame_index}: analyze_frame_bytes error: {e.detail}")

        frame_index += 1

    cap.release()

    try:
        os.remove(path)
    except Exception:
        pass

    video_summary = video_summary_with_gpt(results) if results else "Inga frames analyserades."

    return {
        "video_filename": video.filename,
        "total_frames": frame_count,
        "analyzed_frames": len(results),
        "fps_estimate": fps,
        "frame_step": step,
        "results": results,
        "video_summary": video_summary,
    }


@app.post("/api/train_marking")
async def train_marking(
    item: TrainMarking,
    api_key: str = Depends(verify_api_key),
):
    global MARKING_LABELS
    title = normalize_label_name(item.title)
    MARKING_LABELS[item.code] = title
    with TRAINING_FILE.open("w", encoding="utf-8") as f:
        json.dump(MARKING_LABELS, f, ensure_ascii=False, indent=2)
    return {"ok": True}


@app.post("/api/train_visual")
async def train_visual(
    api_key: str = Depends(verify_api_key),
    label: str = Form(...),
    image: UploadFile = File(...)
):
    global VISUAL_MEMORY

    label = label.strip()
    if not label:
        raise HTTPException(400, "Empty label")

    label = normalize_label_name(label)

    data = await image.read()
    arr = np.frombuffer(data, np.uint8)
    patch = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if patch is None:
        raise HTTPException(400, "Kunde inte läsa patch-bild")

    feat = compute_feature(patch)

    found = False
    for entry in VISUAL_MEMORY:
        if entry.get("label") == label:
            feats = entry.get("features")
            if feats is None:
                old = entry.get("feature")
                feats = []
                if old:
                    feats.append(old)
                entry["features"] = feats
                entry.pop("feature", None)
            entry["features"].append(feat)
            entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
            found = True
            break

    if not found:
        VISUAL_MEMORY.append({
            "label": label,
            "features": [feat],
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    save_visual_memory()

    return {"ok": True, "label": label, "count": len(VISUAL_MEMORY)}


@app.post("/api/log_yolo_sample")
async def log_yolo_sample(
    api_key: str = Depends(verify_api_key),
    label: str = Form(...),
    x: int = Form(...),
    y: int = Form(...),
    w: int = Form(...),
    h: int = Form(...),
    image: UploadFile = File(...),
):
    label = label.strip()
    if not label:
        raise HTTPException(400, "Empty label")

    if label == "__IGNORE__":
        return {"ok": True, "ignored": True}

    data = await image.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Kunde inte läsa bild")

    h_img, w_img, _ = frame.shape

    x2 = max(0, min(x + w, w_img))
    y2 = max(0, min(y + h, h_img))
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, x2 - x)
    h = max(1, y2 - y)

    img_name = f"frame_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}.jpg"
    img_path = YOLO_IMAGES_DIR / img_name
    cv2.imwrite(str(img_path), frame)

    record = {
        "image": img_name,
        "width": w_img,
        "height": h_img,
        "bbox": {"x": x, "y": y, "w": w, "h": h},
        "label": label,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    with YOLO_META_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"ok": True, "image": img_name}


@app.post("/api/train_yolo_now")
async def train_yolo_now(
    api_key: str = Depends(verify_api_key)
):
    if not ENABLE_TRAIN_YOLO:
        raise HTTPException(403, "Train endpoint disabled. Set ENABLE_TRAIN_YOLO=1 to allow.")
    try:
        result = subprocess.run(
            ["python3", "train_yolo.py"],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        ok = result.returncode == 0
        return {
            "ok": ok,
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-2000:],
        }
    except Exception as e:
        raise HTTPException(500, f"Train script error: {e}")


@app.options("/api/analyze")
async def options_analyze():
    return {}
