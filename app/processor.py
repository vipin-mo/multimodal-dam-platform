import os
import cv2
import uuid
import boto3
import logging
import requests
from PIL import Image
from celery import Celery
from pypdf import PdfReader
from typing import Dict, Any
from pydub import AudioSegment
from qdrant_client import QdrantClient
from faster_whisper import WhisperModel
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

# System Logger Initialization
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DAM-Worker")

# Environment Infrastructure Configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq:5672//")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis-cache:6379/0")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin_vipin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "admin_vipin")
BUCKET_NAME = "dam-media"
QDRANT_HOST = os.getenv("QDRANT_HOST", "vector-db")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://api-gateway:8000")
SHARED_MEDIA_DIR = os.getenv("SHARED_MEDIA_DIR", "/app/shared-media")

os.makedirs(SHARED_MEDIA_DIR, exist_ok=True)

# Celery Application Setup
celery_app = Celery("dam_tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
celery_app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    result_persistent=True
)

# Object Storage Client Initialization
s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=boto3.session.Config(signature_version="s3v4"),
    region_name="us-east-1"
)

# Vector Database Client Initialization
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Lazy-loaded Model Singletons
_clip_model = None
_whisper_model = None

def get_clip_model():
    """Initializes and caches the local CLIP model instance."""
    global _clip_model
    if _clip_model is None:
        logger.info("Initializing worker-level CLIP model (clip-ViT-B-32)...")
        _clip_model = SentenceTransformer("clip-ViT-B-32")
    return _clip_model

def get_whisper_model():
    """Initializes and caches the Whisper audio transcription engine."""
    global _whisper_model
    if _whisper_model is None:
        logger.info("Initializing worker-level Whisper engine (tiny, int8)...")
        _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _whisper_model


# --------------------------------------------------------------------
# Celery Operational Core Processing Orchestrator
# --------------------------------------------------------------------

@celery_app.task(bind=True, name="app.processor.process_media_asset")
def process_media_asset(self, filename: str) -> Dict[str, Any]:
    """Ingests media assets from MinIO and routes them to type-specific processing pipelines."""
    logger.info(f"Starting ingestion pipeline for file: {filename}")
    self.update_state(state="PROGRESS", meta={"progress": 10})

    local_path = os.path.join(SHARED_MEDIA_DIR, filename)
    try:
        s3_client.download_file(BUCKET_NAME, filename, local_path)
        self.update_state(state="PROGRESS", meta={"progress": 25})

        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext == ".pdf":
            result = process_pdf(self, filename, local_path)
        elif ext in [".png", ".jpg", ".jpeg"]:
            result = process_static_image(self, filename, local_path)
        elif ext in [".mp4", ".avi", ".mov", ".mkv", ".mp3", ".wav"]:
            result = process_multimedia(self, filename, local_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        self.update_state(state="SUCCESS", meta={"progress": 100})
        return result

    except Exception as e:
        logger.error(f"Pipeline processing failed for asset '{filename}': {str(e)}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise e
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.info(f"Temporary file cleared: {local_path}")


# --------------------------------------------------------------------
# Structural File Domain Parsers
# --------------------------------------------------------------------

def process_pdf(task, filename: str, path: str) -> Dict[str, Any]:
    """Extracts text content from PDF pages and vectorizes chunks for search indexing."""
    logger.info(f"Parsing document text layout maps for target: {filename}")
    reader = PdfReader(path)
    total_pages = len(reader.pages)
    points = []

    for idx, page in enumerate(reader.pages):
        raw_text = page.extract_text()
        if not raw_text or not raw_text.strip():
            continue

        paragraphs = [p.strip() for p in raw_text.split("\n") if len(p.strip()) > 20]

        for paragraph in paragraphs:
            try:
                response = requests.post(
                    f"{API_GATEWAY_URL}/api/internal/embed",
                    json={"text": paragraph},
                    timeout=15
                )
                if response.status_code == 200:
                    text_vector = response.json()["vector"]
                else:
                    logger.error(f"Embedding API returned status code: {response.status_code}")
                    continue
            except Exception as embed_err:
                logger.error(f"Internal embedding network connection failed: {str(embed_err)}")
                continue

            points.append(
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=text_vector,
                    payload={
                        "filename": filename,
                        "text": paragraph,
                        "timestamp": float(idx + 1),
                        "type": "document"
                    }
                )
            )

        progress_weight = 25 + int((idx / total_pages) * 65)
        task.update_state(state="PROGRESS", meta={"progress": progress_weight})

    if points:
        qdrant_client.upsert(collection_name="text_chunks", points=points)

    return {"processed_type": "PDF_DOCUMENT", "chunks_indexed": len(points)}


def process_static_image(task, filename: str, path: str) -> Dict[str, Any]:
    """Calculates CLIP visual feature vectors for flat image matrices and indexes them,
    while also generating text embeddings for keyword search cross-compatibility."""
    logger.info(f"Computing visual vectors for image target: {filename}")

    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image matrix data is unreadable or corrupted.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # 1. Index visual features into CLIP collection
    clip_engine = get_clip_model()
    image_vector = clip_engine.encode(pil_img).tolist()

    task.update_state(state="PROGRESS", meta={"progress": 60})

    qdrant_client.upsert(
        collection_name="image_features",
        points=[
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=image_vector,
                payload={
                    "filename": filename,
                    "timestamp": 0.0,
                    "type": "image",
                },
            )
        ],
    )

    # 2. Index filename/metadata text into text_chunks collection via gateway embedding endpoint
    task.update_state(state="PROGRESS", meta={"progress": 85})
    text_content = f"Uploaded image asset: {filename}. Visual digital graphic or poster resource."

    try:
        embed_res = requests.post(
            f"{API_GATEWAY_URL}/api/internal/embed",
            json={"text": text_content},
            timeout=10
        )
        if embed_res.status_code == 200:
            text_vector = embed_res.json()["vector"]
            qdrant_client.upsert(
                collection_name="text_chunks",
                points=[
                    qmodels.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=text_vector,
                        payload={
                            "filename": filename,
                            "text": text_content,
                            "timestamp": 0.0,
                            "type": "image_metadata"
                        }
                    )
                ]
            )
    except Exception as e:
        logger.warning(f"Failed to push image metadata text vector: {str(e)}")

    return {"processed_type": "STATIC_IMAGE", "chunks_indexed": 2}


def process_multimedia(task, filename: str, path: str) -> Dict[str, Any]:
    """Unified pipeline for processing sequential video frames and speech-to-text streams."""
    logger.info(f"Analyzing multimedia streams for target asset: {filename}")

    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    audio_path = os.path.join(SHARED_MEDIA_DIR, f"{uuid.uuid4()}.mp3")
    has_video = ext in [".mp4", ".avi", ".mov", ".mkv"]
    has_audio = True

    video_points = []
    audio_points = []

    # --- 1. Downsampled Frame Matrix Processing Loop ---
    if has_video:
        logger.info("Extracting sample frames from video file matrix...")
        video = cv2.VideoCapture(path)
        fps = video.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = int(fps * 2)

        clip_engine = get_clip_model()
        frame_count = 0

        try:
            while True:
                ret, frame = video.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    current_ts = float(frame_count / fps)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    feat_vector = clip_engine.encode(pil_img).tolist()

                    video_points.append(
                        qmodels.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=feat_vector,
                            payload={
                                "filename": filename,
                                "timestamp": current_ts,
                                "type": "video_frame",
                            },
                        )
                    )
                frame_count += 1
        finally:
            video.release()

        task.update_state(state="PROGRESS", meta={"progress": 60})

    # --- 2. Isolation and Processing of Audio Streams ---
    logger.info("Extracting audio channels to MP3 format...")
    try:
        track = AudioSegment.from_file(path)
        track.export(audio_path, format="mp3")
    except Exception as audio_err:
        logger.warning(f"Audio channel extraction skipped or unavailable: {str(audio_err)}")
        has_audio = False

    if has_audio and os.path.exists(audio_path):
        try:
            whisper_engine = get_whisper_model()
            segments_gen, _ = whisper_engine.transcribe(audio_path, beam_size=1)
            segments = list(segments_gen)

            for segment in segments:
                text_chunk = segment.text.strip()
                if not text_chunk:
                    continue

                try:
                    embed_res = requests.post(
                        f"{API_GATEWAY_URL}/api/internal/embed",
                        json={"text": text_chunk},
                        timeout=10
                    )
                    if embed_res.status_code == 200:
                        text_vector = embed_res.json()["vector"]
                        audio_points.append(
                            qmodels.PointStruct(
                                id=str(uuid.uuid4()),
                                vector=text_vector,
                                payload={
                                    "filename": filename,
                                    "text": text_chunk,
                                    "timestamp": float(segment.start),
                                    "type": "audio_transcript"
                                }
                            )
                        )
                    else:
                        logger.error(f"Gateway embedding failure status: {embed_res.status_code}")
                except Exception as embed_err:
                    logger.error(f"Failed internal text transcription embedding offload: {str(embed_err)}")
                    continue
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    task.update_state(state="PROGRESS", meta={"progress": 90})

    # --- 3. Synchronous Mass Index Upserting ---
    if video_points:
        qdrant_client.upsert(collection_name="image_features", points=video_points)
    if audio_points:
        qdrant_client.upsert(collection_name="text_chunks", points=audio_points)

    return {
        "processed_type": "MULTIMEDIA_STREAM",
        "visual_frames_indexed": len(video_points),
        "transcript_segments_indexed": len(audio_points),
    }
