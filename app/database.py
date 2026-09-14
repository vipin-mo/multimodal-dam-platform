import os
import time
import bcrypt
import logging
from sqlalchemy import create_engine, Column, String
from sqlalchemy.exc import OperationalError, IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# System Logger Initialization
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DAM-Database")

# Infrastructure Environment Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres-db:5432/dam_db")

# Setup SQLAlchemy engine with standard pooling configuration
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    """Database model mapping user credentials and authorization roles."""
    __tablename__ = "users"

    username = Column(String, primary_key=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False)  # admin or viewer

    def verify_password(self, plain_password: str) -> bool:
        """Verifies plain text password strings against recorded bcrypt hashes."""
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            self.hashed_password.encode('utf-8')
        )


def get_db():
    """FastAPI context dependency yielding isolated database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    """Generates standard bcrypt hashes from plaintext password strings."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def init_db():
    """Handles cold starts with retry loops, builds tables, and seeds initial users."""
    logger.info("Initializing database schema...")

    # 1. Cold start polling loop for database readiness
    max_retries = 10
    for attempt in range(max_retries):
        try:
            with engine.connect():
                logger.info("Connection to PostgreSQL successfully validated.")
                break
        except OperationalError as e:
            if attempt == max_retries - 1:
                logger.critical("Database connection attempts exhausted. Ingress rejected.")
                raise e
            logger.warning(f"Database not ready (Attempt {attempt + 1}/{max_retries}). Retrying in 3s...")
            time.sleep(3)

    # 2. Safe schema application protecting concurrent application replicas
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as migration_err:
        logger.warning(f"Schema generation encountered race lock: {str(migration_err)}. Proceeding safely.")

    # 3. Race condition proof seeding implementation
    db: Session = SessionLocal()
    try:
        if not db.query(User).first():
            logger.info("Seeding default credential sets...")

            default_users = [
                User(
                    username="admin_vipin",
                    hashed_password=hash_password("admin123"),
                    role="admin"
                ),
                User(
                    username="viewer_guest",
                    hashed_password=hash_password("guest123"),
                    role="viewer"
                )
            ]

            try:
                db.add_all(default_users)
                db.commit()
                logger.info("Database seeding successfully executed.")
            except IntegrityError:
                db.rollback()
                logger.info("Concurrent worker instance completed seed sequence first. Bypassed.")
        else:
            logger.info("Credentials verified. Skipping seed execution.")
    except Exception as e:
        db.rollback()
        logger.error(f"Initialization sequence failed: {str(e)}")
        raise e
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
