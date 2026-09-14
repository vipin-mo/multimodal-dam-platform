import os
import jwt
import boto3
import base64
import bcrypt
import logging
import httpx
import asyncio
import requests
from typing import List, Dict, Any
from celery import Celery
from celery.result import AsyncResult
from pydantic import BaseModel
from sqlalchemy.orm import Session
from botocore.config import Config
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from fastapi.security import OAuth2PasswordBearer
from fastapi.concurrency import run_in_threadpool
from datetime import datetime, timedelta, timezone
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Query
from fastapi.responses import JSONResponse

# Core Infrastructure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DAM-Gateway")

# System Configurations
SECRET_KEY = os.getenv("JWT_SECRET", "super-secret-enterprise-dam-signing-key-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin_vipin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "admin_vipin")
BUCKET_NAME = "dam-media"

QDRANT_HOST = os.getenv("QDRANT_HOST", "vector-db")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama-engine:11434/v1/chat/completions")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq:5672//")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis-cache:6379/0")

SHARED_MEDIA_DIR = os.getenv("SHARED_MEDIA_DIR", "/app/shared-media")

# Lazy-Loaded Embedding Model Singletons
text_model = None
clip_model = None

# Shared tracking app instance for worker status evaluations
celery_tracker = Celery("gateway_tracker_runtime", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

oauth2_password_bearer = OAuth2PasswordBearer(tokenUrl="/api/token")

async def lifespan(_):
    """Asynchronous lifespan manager handling blocking infrastructure hot-starts."""
    logger.info("Initializing multi-modal enterprise storage clusters...")
    global text_model, clip_model

    await run_in_threadpool(init_db)
    await run_in_threadpool(init_object_store)
    await run_in_threadpool(init_vector_store)

    logger.info("Caching local Transformer weights into active system memory...")
    text_model = await run_in_threadpool(SentenceTransformer, "all-MiniLM-L6-v2")
    clip_model = await run_in_threadpool(SentenceTransformer, "clip-ViT-B-32")

    logger.info("All engine pipelines online.")
    yield
    logger.info("Shutting down FastAPI Gateway cleanly...")


app = FastAPI(
    title="Multimodal Digital Asset Management Gateway",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

# Storage & Database Client Core Connections
s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1"
)

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

from app.database import get_db, User, init_db

def init_object_store():
    """Validates presence of primary object storage namespace in MinIO."""
    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        logger.info(f"Connected to MinIO bucket: '{BUCKET_NAME}'")
    except Exception:
        logger.info(f"Provisioning missing MinIO bucket: '{BUCKET_NAME}'")
        s3_client.create_bucket(Bucket=BUCKET_NAME)

def init_vector_store():
    """Validates schema integrity for multi-modal vector partitions."""
    existing_collections = [c.name for c in qdrant_client.get_collections().collections]

    if "text_chunks" not in existing_collections:
        logger.info("Provisioning Qdrant index space: 'text_chunks' (384 Dim)")
        qdrant_client.create_collection(
            collection_name="text_chunks",
            vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE)
        )

    if "image_features" not in existing_collections:
        logger.info("Provisioning Qdrant index space: 'image_features' (512 Dim)")
        qdrant_client.create_collection(
            collection_name="image_features",
            vectors_config=qmodels.VectorParams(size=512, distance=qmodels.Distance.COSINE)
        )


# --------------------------------------------------------------------
# Role-Based Access Control & Cryptography Infrastructure
# --------------------------------------------------------------------

class TokenRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    role: str

class EmbeddingPayload(BaseModel):
    text: str


def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_password_bearer), db: Session = Depends(get_db)) -> User:
    """Decodes token parameters and checks credentials against database records."""
    auth_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token validation signature failed.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise auth_exception
    except jwt.PyJWTError:
        raise auth_exception

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise auth_exception
    return user

class RoleChecker:
    """RBAC boundary evaluator ensuring structural execution clearance."""
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: User = Depends(get_current_user)):
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operational clearance level insufficient for this context."
            )
        return current_user


# --------------------------------------------------------------------
# 1. Identity & Database Management Controllers
# --------------------------------------------------------------------

@app.post("/api/token", response_model=TokenResponse)
def login(payload: TokenRequest, db: Session = Depends(get_db)):
    """Authenticates stateful user credentials and issues a secure JWT token."""
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid username or password configuration.")

    password_bytes = payload.password.encode('utf-8')
    hashed_bytes = user.hashed_password.encode('utf-8')

    if not bcrypt.checkpw(password_bytes, hashed_bytes):
        raise HTTPException(status_code=400, detail="Invalid username or password configuration.")

    token_expiry = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(data={"sub": user.username, "role": user.role}, expires_delta=token_expiry)
    return {"access_token": token, "token_type": "bearer", "role": user.role}


@app.post("/api/init-db")
def reset_storage_environments(current_user: User = Depends(RoleChecker(["admin"]))):
    """Administrative utility to drop and re-initialize vector spaces and object storage pools."""
    try:
        objects = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
        if "Contents" in objects:
            delete_keys = [{"Key": obj["Key"]} for obj in objects["Contents"]]
            s3_client.delete_objects(Bucket=BUCKET_NAME, Delete={"Objects": delete_keys})

        try:
            qdrant_client.delete_collection("text_chunks")
            qdrant_client.delete_collection("image_features")
        except Exception:
            pass

        init_vector_store()
        return {"status": "SUCCESS", "detail": "All media assets and vector profiles dropped and reinitialized."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System scrubbing routine failed: {str(e)}")


# --------------------------------------------------------------------
# 2. Ingestion Pipeline & Streaming Controllers
# --------------------------------------------------------------------

@app.post("/api/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_asset(
    file: UploadFile = File(...),
    current_user: User = Depends(RoleChecker(["admin", "viewer"]))
):
    """Handles direct file streaming to MinIO and dispatches a background Celery ingestion task."""
    filename = file.filename
    try:
        await run_in_threadpool(
            s3_client.upload_fileobj,
            file.file,
            BUCKET_NAME,
            filename
        )
        task = celery_tracker.send_task("app.processor.process_media_asset", args=[filename])
        return {"status": "QUEUED", "task_id": task.id, "target_file": filename}
    except Exception as e:
        logger.error(f"Upload pipeline failed for {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Asset ingestion entry rejected: {str(e)}")


@app.get("/api/media/{filename}")
def get_presigned_streaming_link(
    filename: str,
    current_user: User = Depends(RoleChecker(["admin", "viewer"]))
):
    """Generates a temporary presigned URL tailored for direct browser media streaming headers."""
    try:
        ext = os.path.splitext(filename)[1].lower()
        content_type = "video/mp4"

        if ext in [".mp3", ".wav"]:
            content_type = "audio/mpeg"
        elif ext in [".png", ".jpg", ".jpeg"]:
            content_type = "image/png"

        browser_s3_client = boto3.client(
            "s3",
            endpoint_url="http://localhost:9000",
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        url = browser_s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": BUCKET_NAME,
                "Key": filename,
                "ResponseContentType": content_type,
            },
            ExpiresIn=3600,
        )
        return {"filename": filename, "url": url}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not secure asset locator link: {str(e)}",
        )


@app.get("/api/assets")
def list_all_assets(
    current_user: User = Depends(RoleChecker(["admin", "viewer"]))
):
    """Fetches flat metadata summaries of all resource objects residing inside MinIO."""
    try:
        objects = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
        file_list = []
        if "Contents" in objects:
            for obj in objects["Contents"]:
                file_list.append(
                    {
                        "filename": obj["Key"],
                        "size_bytes": obj["Size"],
                        "last_modified": obj["LastModified"].isoformat(),
                    }
                )
        return {"assets": file_list}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to query repository indices: {str(e)}"
        )


# ====================================================================
# 3. Parallel Search, Grounded RAG & Automated Summarization
# ====================================================================


@app.get("/api/search")
async def execute_multimodal_search(
    query: str = Query(...),
    limit: int = Query(5),
    current_user: User = Depends(RoleChecker(["admin", "viewer"])),
):
    """Executes a dual-vector parallel search across textual and visual semantic indices."""
    try:
        query_vector_raw = await run_in_threadpool(text_model.encode, query)
        text_results = qdrant_client.search(
            collection_name="text_chunks",
            query_vector=query_vector_raw.tolist(),
            limit=limit,
        )
        image_results = []
        try:
            clip_vector_raw = await run_in_threadpool(clip_model.encode, query)
            image_results = qdrant_client.search(
                collection_name="image_features",
                query_vector=clip_vector_raw.tolist(),
                limit=limit,
            )
        except Exception as qdrant_err:
            logger.warning(
                f"Visual cross-modal matching sequence skipped: {qdrant_err}"
            )

        formatted_text = []
        for r in text_results:
            payload = r.payload or {}
            formatted_text.append(
                {
                    "filename": payload.get("filename", "Unknown"),
                    "text": payload.get("text", "No context text found."),
                    "timestamp": payload.get("timestamp", 0.0),
                    "score": getattr(r, "score", 0.0),
                }
            )

        formatted_images = []
        for r in image_results:
            payload = r.payload or {}
            formatted_images.append(
                {
                    "filename": payload.get("filename", "Unknown"),
                    "timestamp": payload.get("timestamp", 0.0),
                    "score": getattr(r, "score", 0.0),
                }
            )

        return {
            "text_matches": formatted_text,
            "visual_matches": formatted_images,
        }
    except Exception as e:
        logger.error(f"Search pipeline global runtime failure: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search pipeline encountered an error state: {str(e)}",
        )


@app.get("/api/rag")
async def execute_rag_pipeline(
    query: str = Query(...),
    current_user: User = Depends(RoleChecker(["admin", "viewer"])),
):
    """Grounds workspace queries against indexed vector context using an LLM layer."""
    try:
        clean_query = query.strip()
        if len(clean_query.lower()) < 2:
            raise HTTPException(
                status_code=400, detail="Query density threshold insufficient."
            )

        query_vector = await run_in_threadpool(text_model.encode, clean_query)
        text_results = qdrant_client.search(
            collection_name="text_chunks",
            query_vector=query_vector.tolist(),
            limit=5,
            score_threshold=0.20,
        )

        if not text_results:
            return {
                "query": clean_query,
                "answer": f"No explicit grounding metrics found across storage nodes for: '{clean_query}'.",
                "grounding_context": [],
            }

        context_blocks = []
        for r in text_results:
            payload = r.payload or {}
            filename = payload.get("filename", "Unknown Asset")
            ts = payload.get("timestamp", 0.0)
            txt = payload.get("text", "")
            asset_type = payload.get("type", "unknown")

            if asset_type == "document":
                meta = f"[Document Asset: {filename} @ Page: {int(ts)}]"
            elif asset_type == "audio_transcript":
                meta = f"[Media Audio/Video Asset: {filename} @ Timeline: {ts:.2f}s]"
            else:
                meta = f"[Asset: {filename}]"
            context_blocks.append(f"{meta}: {txt}")

        context_str = "\n".join(context_blocks)
        system_prompt = (
            "You are an expert enterprise Digital Asset Management analytics assistant.\n"
            "Using ONLY the provided context blocks, answer the user query accurately and directly.\n"
            "If the text matches a spoken phrase or document page, point out exactly which asset file contains it."
        )
        user_prompt = f"Grounded Context Blocks:\n{context_str}\n\nUser Question: {clean_query}\nAnswer:"

        response_text = await call_ollama_with_backoff(system_prompt, user_prompt)
        return {
            "query": clean_query,
            "answer": response_text,
            "grounding_context": context_blocks,
        }
    except Exception as e:
        logger.error(f"RAG Runtime Failure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/summarize")
async def generate_asset_summary(
    filename: str = Query(...),
    current_user: User = Depends(RoleChecker(["admin", "viewer"])),
):
    """Generates precise executive summaries from asset text indices or visual feature maps using a VLM."""
    try:
        ext = os.path.splitext(filename)[1].lower()
        if ext in [".png", ".jpg", ".jpeg"]:
            temp_img_path = os.path.join(SHARED_MEDIA_DIR, f"temp_{filename}")
            try:
                s3_client.download_file(BUCKET_NAME, filename, temp_img_path)
                with open(temp_img_path, "rb") as img_file:
                    encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
            finally:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)

            system_prompt = (
                "You are an elite Digital Asset Management visual analysis module. "
                "Generate a professional executive report describing the contents, elements, text, and context of this image."
            )
            summary_output = await call_ollama_vision(system_prompt, encoded_image)
            return {"filename": filename, "summary": f"### Visual Asset Analysis Report: {filename}\n\n{summary_output}"}

        text_scroll = qdrant_client.scroll(
            collection_name="text_chunks",
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="filename", match=qmodels.MatchValue(value=filename)
                    )
                ]
            ),
            limit=100,
        )
        points = text_scroll[0] if isinstance(text_scroll, tuple) else text_scroll
        if not points:
            raise HTTPException(
                status_code=404,
                detail="No matching indexed footprints found for this file signature.",
            )

        sorted_points = sorted(
            points, key=lambda x: x.payload.get("timestamp", 0.0)
        )
        context_lines = []
        total_text_len = 0

        for p in sorted_points:
            txt = p.payload.get("text", "").strip()
            total_text_len += len(txt)
            if p.payload.get("type") == "document":
                context_lines.append(f"[Page {int(p.payload.get('timestamp', 1))}]: {txt}")
            else:
                context_lines.append(f"[Timestamp {p.payload.get('timestamp', 0.0)}s]: {txt}")

        full_context = "\n".join(context_lines)
        if total_text_len < 100:
            return {
                "filename": filename,
                "summary": (
                    f"### Executive Intelligence Summary Report: {filename}\n"
                    f'Direct Transcript Footprint: "{full_context}"\n\n'
                    "⚠️ Notice: This resource footprint contains insufficient textual metadata to warrant an automated "
                    "multi-tier summary report. The absolute content is displayed directly above."
                ),
            }

        system_prompt = (
            "You are an elite, literal Digital Asset Management summarization module.\n"
            "Generate a highly professional executive report based strictly on the user context provided. "
            "CRITICAL: Do not invent any metrics, external sequences, or details. "
            "If the source text is short, keep your output brief and concise."
        )
        user_prompt = f"Target Resource Context Lines for '{filename}':\n{full_context}\n\nGenerate accurate summary document:"

        summary_output = await call_ollama_with_backoff(system_prompt, user_prompt)
        return {"filename": filename, "summary": summary_output}
    except Exception as e:
        logger.error(f"Summarizer execution crash profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tasks/{task_id}")
def check_celery_task_status(
    task_id: str, current_user: User = Depends(RoleChecker(["admin", "viewer"]))
):
    """Queries the Celery backend storage cache for real-time task status tracking maps."""
    try:
        res = AsyncResult(task_id, app=celery_tracker)
        response = {"task_id": task_id, "state": res.state, "progress": 0, "result": None}

        if res.state == "PROGRESS":
            if isinstance(res.info, dict):
                response["progress"] = res.info.get("progress", 0)
        elif res.state == "SUCCESS":
            response["progress"] = 100
            response["result"] = res.result
        elif res.state == "FAILURE":
            response["progress"] = 0
            response["result"] = str(res.info)

        return response
    except Exception as e:
        logger.error(f"Failed to extract task metadata: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Task engine communication loss: {str(e)}"
        )


# ====================================================================
# 4. Internal Endpoint & Helper Utilities
# ====================================================================


@app.post("/api/internal/embed")
async def generate_internal_text_embedding(payload: EmbeddingPayload):
    """Offloads worker text embedding compute directly over fast network loop boundaries."""
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Missing text key boundary.")

    try:
        vector = await run_in_threadpool(text_model.encode, payload.text)
        return {"vector": vector.tolist()}
    except Exception as e:
        logger.error(f"Inference gateway matrix failure: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal inference execution exception.")


async def call_ollama_with_backoff(system_msg: str, user_msg: str, max_retries: int = 3) -> str:
    """Invokes local Ollama inference using native async drivers and exponential backoff hooks."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.1,
        "stream": False
    }

    delay = 2
    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(OLLAMA_URL, json=payload, timeout=90.0)
                if response.status_code == 200:
                    res_json = response.json()

                    choices = res_json.get("choices", [])
                    if choices and isinstance(choices, list):
                        content = choices[0].get("message", {}).get("content")
                        if content is not None:
                            return content

                    if "message" in res_json:
                        return res_json["message"].get("content", "")

                logger.warning(f"Ollama returned bad status {response.status_code}, retrying...")
            except Exception as e:
                logger.warning(f"Ollama connection attempt {attempt + 1} failed: {str(e)}")

            await asyncio.sleep(delay)
            delay *= 2

    raise RuntimeError("Ollama compute cluster failed to return valid response after backoff timeout.")



async def call_ollama_vision(
    system_prompt: str, base64_image: str, max_retries: int = 3
) -> str:
    """Queries an Ollama vision model with base64-encoded image data and backoff logic."""
    payload = {
        "model": "llava",
        "prompt": "Analyze this image and provide a thorough executive summary report.",
        "system": system_prompt,
        "images": [base64_image],
        "stream": False,
    }
    delay = 2
    async with httpx.AsyncClient(timeout=90.0) as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    f"{OLLAMA_HOST}/api/generate", json=payload
                )
                if response.status_code == 200:
                    return response.json().get("response", "No visual description generated.")
                logger.warning(f"Ollama vision returned status {response.status_code}, retrying...")
            except Exception as e:
                logger.warning(f"Ollama vision connection attempt {attempt + 1} failed: {str(e)}")
            await asyncio.sleep(delay)
            delay *= 2

    raise RuntimeError("Ollama vision cluster failed to return valid response after backoff timeout.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)