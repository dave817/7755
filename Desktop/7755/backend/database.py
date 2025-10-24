"""
Database setup and models for the dating chatbot
Phase 2: Conversation persistence and history management
"""
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from backend.config import settings

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Database Models

class User(Base):
    """User model"""
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    characters = relationship("Character", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")


class Character(Base):
    """AI Character model"""
    __tablename__ = "characters"

    character_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    name = Column(String(50), nullable=False)
    gender = Column(String(50), nullable=False)
    identity = Column(String(200))
    nickname = Column(String(50))
    detail_setting = Column(Text)  # Up to 500 chars
    other_setting = Column(JSON)  # JSON stored as text
    knowledge_base_id = Column(String(100))  # SenseChat knowledge base ID

    # Image generation fields
    appearance_description = Column(Text)  # Physical appearance for image generation
    base_image_url = Column(String(500))  # URL of base character face image
    image_seed = Column(Integer)  # Seed for consistent character generation
    image_model_id = Column(String(100))  # Miaohua model ID used

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="characters")
    messages = relationship("Message", back_populates="character", cascade="all, delete-orphan")
    favorability = relationship(
        "FavorabilityTracking",
        back_populates="character",
        uselist=False,
        cascade="all, delete-orphan"
    )
    generated_images = relationship("GeneratedImage", back_populates="character", cascade="all, delete-orphan")


class Message(Base):
    """Conversation message model"""
    __tablename__ = "messages"

    message_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    character_id = Column(Integer, ForeignKey("characters.character_id"), nullable=False)
    speaker_name = Column(String(50), nullable=False)  # Who said this message
    message_content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    favorability_level = Column(Integer, default=1)  # 1, 2, or 3

    # Relationships
    user = relationship("User", back_populates="messages")
    character = relationship("Character", back_populates="messages")


class UserPreference(Base):
    """User custom memory/preferences model"""
    __tablename__ = "user_preferences"

    preference_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    category = Column(String(50), nullable=False)  # likes, dislikes, habits, background
    content = Column(JSON, nullable=False)  # JSON content
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="preferences")


class FavorabilityTracking(Base):
    """Favorability level tracking model"""
    __tablename__ = "favorability_tracking"

    tracking_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    character_id = Column(Integer, ForeignKey("characters.character_id"), nullable=False, unique=True)
    current_level = Column(Integer, default=1)  # 1, 2, or 3
    message_count = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    character = relationship("Character", back_populates="favorability")


class GeneratedImage(Base):
    """Generated images for characters at different milestones"""
    __tablename__ = "generated_images"

    image_id = Column(Integer, primary_key=True, index=True)
    character_id = Column(Integer, ForeignKey("characters.character_id"), nullable=False)
    image_url = Column(String(500), nullable=False)
    image_type = Column(String(50), nullable=False)  # 'first_message', 'milestone', 'message_50'
    trigger_count = Column(Integer)  # Message count or milestone when generated
    activity_description = Column(Text)  # Activity/scene description
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    character = relationship("Character", back_populates="generated_images")


# Create all tables
def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")


if __name__ == "__main__":
    init_db()
