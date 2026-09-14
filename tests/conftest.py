import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import MagicMock, patch

# Enforce an isolated in-memory SQLite engine for test isolation
TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Standardize mock environments before core application loads
os.environ["JWT_SECRET"] = "test-suite-secret-key-2026-enterprise-dam"
os.environ["MINIO_ENDPOINT"] = "http://mock-minio:9000"
os.environ["QDRANT_HOST"] = "mock-vector-db"
os.environ["SHARED_MEDIA_DIR"] = "/tmp/shared-media"

from app.database import Base, User, hash_password
from app.main import app, get_db


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Initializes in-memory relational schemas and seeds standard authorization roles."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        admin_user = User(
            username="test_admin",
            hashed_password=hash_password("admin123"),
            role="admin"
        )
        viewer_user = User(
            username="test_viewer",
            hashed_password=hash_password("viewer123"),
            role="viewer"
        )
        db.add(admin_user)
        db.add(viewer_user)
        db.commit()
    finally:
        db.close()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session():
    """Provides an isolated transactional database session wrapper for individual tests."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(autouse=True)
def override_dependencies(db_session):
    """Overrides application database dependencies with the test session context."""
    app.dependency_overrides[get_db] = lambda: db_session
    yield
    app.dependency_overrides.clear()


@pytest.fixture(scope="session", autouse=True)
def mock_infrastructure_drivers():
    """Mocks external network infrastructure dependencies (S3/MinIO, Qdrant, and Celery)."""
    with patch("boto3.client") as mock_boto, \
         patch("app.main.qdrant_client") as mock_qdrant, \
         patch("app.main.s3_client") as mock_main_s3, \
         patch("celery.Celery.send_task") as mock_celery_task:

        # Configure Boto3/S3 Client Mocks
        mock_s3_instance = MagicMock()
        mock_s3_instance.list_objects_v2.return_value = {
            "Contents": [
                {
                    "Key": "mock_file_1.mp4",
                    "Size": 1048576,
                    "LastModified": MagicMock(isoformat=lambda: "2026-09-12T00:00:00Z")
                }
            ]
        }
        mock_s3_instance.generate_presigned_url.return_value = "http://localhost:9000/dam-media/mock_file_1.mp4?signature=valid"
        mock_boto.return_value = mock_s3_instance
        mock_main_s3.upload_fileobj.return_value = True

        # Configure Celery Async Dispatch Mock
        mock_celery_task.return_value = MagicMock(id="mock-task-uuid-12345")

        # Configure Qdrant Vector Search Client Mocks
        mock_text_search_result = MagicMock()
        mock_text_search_result.payload = {
            "filename": "mock_file_1.mp4",
            "text": "Extracted search context snippet.",
            "timestamp": 20.0
        }
        mock_text_search_result.score = 0.89

        mock_visual_search_result = MagicMock()
        mock_visual_search_result.payload = {
            "filename": "mock_file_1.mp4",
            "timestamp": 20.0
        }
        mock_visual_search_result.score = 0.78

        mock_qdrant.search.side_effect = lambda collection_name, **kwargs: (
            [mock_text_search_result] if collection_name == "text_chunks" else [mock_visual_search_result]
        )

        # Mock Scroll endpoint for RAG context extraction
        mock_point = MagicMock()
        mock_point.payload = {
            "filename": "mock_file_1.mp4",
            "text": "Grounding transcript slice content.",
            "timestamp": 10.0
        }
        mock_qdrant.scroll.return_value = ([mock_point], None)

        yield {
            "s3": mock_s3_instance,
            "qdrant": mock_qdrant,
            "celery": mock_celery_task
        }
