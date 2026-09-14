import pytest
import numpy as np
from app.main import app
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from qdrant_client.http import models as qmodels

client = TestClient(app)

# --------------------------------------------------------------------
# 1. Identity & Security Validation Tests
# --------------------------------------------------------------------

def get_auth_token(username, password):
    """Helper utility to generate standard access tokens for individual test cases."""
    res = client.post("/api/token", json={"username": username, "password": password})
    return res.json()["access_token"]


def test_token_issuance_valid_admin_credentials():
    """Verifies token generation logic and claim parsing for administrative users."""
    response = client.post("/api/token", json={"username": "test_admin", "password": "admin123"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["role"] == "admin"


def test_token_issuance_valid_viewer_credentials():
    """Verifies token generation logic and claim parsing for standard viewer users."""
    response = client.post("/api/token", json={"username": "test_viewer", "password": "viewer123"})
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "viewer"


def test_token_issuance_invalid_password():
    """Ensures incorrect passwords trigger a validation rejection code."""
    response = client.post("/api/token", json={"username": "test_admin", "password": "wrong_password"})
    assert response.status_code == 400
    assert "Invalid" in response.json()["detail"]


def test_secured_route_unauthenticated_request():
    """Enforces that requesting secured resources without tokens triggers an unauthorized status."""
    response = client.get("/api/assets")
    assert response.status_code in [401, 422]


def test_secured_route_malformed_jwt_signature():
    """Enforces that requests using invalid tokens are explicitly blocked."""
    response = client.get("/api/assets", params={"token": "invalid-jwt-token-string"})
    assert response.status_code == 401
    assert "validation" in response.json()["detail"].lower()


# --------------------------------------------------------------------
# 2. Role-Based Access Control (RBAC) Tests
# --------------------------------------------------------------------

def test_rbac_admin_clearance_on_sensitive_route():
    """Validates that users with administrative clearance can access infrastructure routes."""
    token = get_auth_token("test_admin", "admin123")
    response = client.post("/api/init-db", params={"token": token})
    assert response.status_code == 200
    assert response.json()["status"] == "SUCCESS"


def test_rbac_viewer_rejection_on_sensitive_route():
    """Validates that standard viewers are completely blocked from admin operations."""
    token = get_auth_token("test_viewer", "viewer123")
    response = client.post("/api/init-db", params={"token": token})
    assert response.status_code == 403
    assert "clearance" in response.json()["detail"]

# --------------------------------------------------------------------
# 3. Multimodal Search, RAG, and Summarization Simulation Tests
# --------------------------------------------------------------------

@patch("app.main.text_model")
@patch("app.main.clip_model")
def test_parallel_multimodal_search_execution(mock_clip, mock_text):
    """Verifies dual-vector searches against index collections."""
    mock_text.encode.return_value = np.zeros((1, 384))
    mock_clip.encode.return_value = np.zeros((1, 512))
    token = get_auth_token("test_viewer", "viewer123")

    response = client.get("/api/search", params={"query": "context", "limit": 3, "token": token})
    assert response.status_code == 200
    data = response.json()
    assert "text_matches" in data
    assert "visual_matches" in data
    assert data["text_matches"][0]["filename"] == "mock_file_1.mp4"
    assert "text" in data["text_matches"][0]
    assert data["visual_matches"][0]["score"] == 0.78


@patch("app.main.text_model")
@patch("app.main.call_ollama_with_backoff")
def test_agentic_rag_pipeline_synthesis(mock_ollama, mock_text):
    """Mocks local LLM runtimes to verify grounded context formatting workflows."""
    mock_text.encode.return_value = np.zeros((1, 384))
    mock_ollama.return_value = "Grounded response matching transcription markers."
    token = get_auth_token("test_viewer", "viewer123")

    response = client.get("/api/rag", params={"query": "Analyze tracking data", "token": token})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "grounding_context" in data
    assert len(data["grounding_context"]) > 0
    assert "mock_file_1.mp4" in data["grounding_context"][0]


@patch("app.main.qdrant_client.scroll")
@patch("app.main.call_ollama_with_backoff")
def test_automated_summary_report_generation(mock_ollama, mock_scroll):
    """Verifies narrative generation processing layouts based on scroll chunks."""
    mock_ollama.return_value = "### Corporate Executive Summary\n- Key timelines verified."
    mock_scroll.return_value = ([
        qmodels.Record(id=1, payload={
            "filename": "mock_file_1.mp4",
            "text": "A" * 150,
            "timestamp": 10.0,
            "type": "transcript"
        })
    ], None)
    token = get_auth_token("test_admin", "admin123")

    response = client.get("/api/summarize", params={"filename": "mock_file_1.mp4", "token": token})
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "mock_file_1.mp4"
    assert "summary" in data
    assert "Corporate Executive Summary" in data["summary"]

# --------------------------------------------------------------------
# 4. Edge-Case Resilience & Boundary Tests
# --------------------------------------------------------------------

@patch("app.main.text_model")
@patch("app.main.qdrant_client.search")
def test_search_pipeline_with_zero_vector_hits(mock_qdrant_search, mock_text):
    """Ensures search endpoints respond gracefully with clean empty arrays on zero matches."""
    mock_text.encode.return_value = np.zeros((1, 384))
    mock_qdrant_search.return_value = []
    token = get_auth_token("test_viewer", "viewer123")

    response = client.get("/api/search", params={"query": "non-existent concept", "token": token})
    assert response.status_code == 200
    data = response.json()
    assert data["text_matches"] == []
    assert data["visual_matches"] == []


@patch("celery.result.AsyncResult")
def test_celery_task_status_propagation_progress(mock_async_result):
    """Verifies tracking progression pipelines for background tasks in PROGRESS state."""
    mock_instance = MagicMock()
    mock_instance.state = "PROGRESS"
    mock_instance.info = {"progress": 65}
    mock_async_result.return_value = mock_instance
    token = get_auth_token("test_viewer", "viewer123")

    response = client.get("/api/tasks/mock-task-id-123", params={"token": token})
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "PROGRESS"
    assert data["progress"] == 65


@patch("celery.result.AsyncResult")
def test_celery_task_status_propagation_failure(mock_async_result):
    """Verifies tracking failure pipelines capture processing exception messages."""
    mock_instance = MagicMock()
    mock_instance.state = "FAILURE"
    mock_instance.info = ValueError("OpenCV image parsing matrix corrupted.")
    mock_async_result.return_value = mock_instance
    token = get_auth_token("test_viewer", "viewer123")

    response = client.get("/api/tasks/mock-task-id-fail", params={"token": token})
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "FAILURE"
    assert "corrupted" in data.get("result", "")
