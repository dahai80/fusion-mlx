"""
SQLAlchemy models for fusion_gui database schema.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
import json

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Float,
    ForeignKey, UniqueConstraint, Index, event
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func

Base = declarative_base()


class ModelStatus(Enum):
    UNLOADED = "unloaded"
    LOADED = "loaded"
    LOADING = "loading"
    FAILED = "failed"


class ModelType(Enum):
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


class InferenceStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QueueStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    path = Column(String, nullable=False)
    version = Column(String)
    model_type = Column(String, nullable=False)
    huggingface_id = Column(String)
    memory_required_gb = Column(Float, nullable=False)
    status = Column(String, nullable=False, default=ModelStatus.UNLOADED.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_used_at = Column(DateTime)
    use_count = Column(Integer, default=0)
    error_message = Column(Text)
    model_metadata = Column(Text)   # JSON string

    # Relationships
    capabilities = relationship("ModelCapability", back_populates="model", cascade="all, delete-orphan")
    inference_sessions = relationship("InferenceSession", back_populates="model", cascade="all, delete-orphan")
    inference_requests = relationship("InferenceRequest", back_populates="model", cascade="all, delete-orphan")
    queue_items = relationship("RequestQueue", back_populates="model", cascade="all, delete-orphan")

    def get_metadata(self) -> Dict[str, Any]:
        if self.model_metadata:
            return json.loads(self.model_metadata)
        return {}

    def set_metadata(self, metadata: Dict[str, Any]):
        self.model_metadata = json.dumps(metadata)

    def increment_use_count(self):
        self.use_count += 1
        self.last_used_at = datetime.utcnow()

    def __repr__(self):
        return f"<Model(name='{self.name}', status='{self.status}', type='{self.model_type}')>"


class ModelCapability(Base):
    __tablename__ = "model_capabilities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    capability = Column(String, nullable=False)

    # Relationships
    model = relationship("Model", back_populates="capabilities")

    __table_args__ = (
        UniqueConstraint("model_id", "capability"),
    )

    def __repr__(self):
        return f"<ModelCapability(model_id={self.model_id}, capability='{self.capability}')>"


class InferenceSession(Base):
    __tablename__ = "inference_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, unique=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, nullable=False, default="active")

    # Relationships
    model = relationship("Model", back_populates="inference_sessions")

    def update_activity(self):
        self.last_activity_at = datetime.utcnow()

    def __repr__(self):
        return f"<InferenceSession(session_id='{self.session_id}', model_id={self.model_id})>"


class InferenceRequest(Base):
    __tablename__ = "inference_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    request_type = Column(String, nullable=False)
    input_data = Column(Text, nullable=False)   # JSON string
    output_data = Column(Text)   # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    duration_ms = Column(Integer)
    status = Column(String, nullable=False, default=InferenceStatus.PENDING.value)
    error_message = Column(Text)

    # Relationships
    model = relationship("Model", back_populates="inference_requests")
    session = relationship("InferenceSession", foreign_keys=[session_id], primaryjoin="InferenceRequest.session_id == InferenceSession.session_id")

    def get_input_data(self) -> Dict[str, Any]:
        return json.loads(self.input_data)

    def set_input_data(self, data: Dict[str, Any]):
        self.input_data = json.dumps(data)

    def get_output_data(self) -> Optional[Dict[str, Any]]:
        if self.output_data:
            return json.loads(self.output_data)
        return None

    def set_output_data(self, data: Dict[str, Any]):
        self.output_data = json.dumps(data)

    def mark_completed(self, output_data: Dict[str, Any]):
        self.status = InferenceStatus.COMPLETED.value
        self.completed_at = datetime.utcnow()
        self.set_output_data(output_data)
        if self.created_at and self.completed_at:
            self.duration_ms = int((self.completed_at - self.created_at).total_seconds() * 1000)

    def mark_failed(self, error_message: str):
        self.status = InferenceStatus.FAILED.value
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        if self.created_at and self.completed_at:
            self.duration_ms = int((self.completed_at - self.created_at).total_seconds() * 1000)

    def __repr__(self):
        return f"<InferenceRequest(id={self.id}, session_id='{self.session_id}', status='{self.status}')>"


class SystemMetrics(Base):
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    memory_used_gb = Column(Float, nullable=False)
    memory_total_gb = Column(Float, nullable=False)
    cpu_usage_percent = Column(Float)
    gpu_memory_used_gb = Column(Float)
    gpu_memory_total_gb = Column(Float)
    active_models_count = Column(Integer, default=0)

    @property
    def memory_usage_percent(self) -> float:
        if self.memory_total_gb > 0:
            return (self.memory_used_gb / self.memory_total_gb) * 100
        return 0.0

    def __repr__(self):
        return f"<SystemMetrics(timestamp='{self.timestamp}', memory_used={self.memory_used_gb}GB)>"


class AppSettings(Base):
    __tablename__ = "app_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String, nullable=False, unique=True)
    value = Column(String, nullable=False)
    value_type = Column(String, nullable=False, default="string")
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def get_typed_value(self) -> Any:
        if self.value_type == "integer":
            return int(self.value)
        elif self.value_type == "boolean":
            return self.value.lower() in ("true", "1", "yes", "on")
        elif self.value_type == "json":
            return json.loads(self.value)
        else:
            return self.value

    def set_typed_value(self, value: Any):
        if isinstance(value, bool):
            self.value = "true" if value else "false"
            self.value_type = "boolean"
        elif isinstance(value, int):
            self.value = str(value)
            self.value_type = "integer"
        elif isinstance(value, (dict, list)):
            self.value = json.dumps(value)
            self.value_type = "json"
        else:
            self.value = str(value)
            self.value_type = "string"

    def __repr__(self):
        return f"<AppSettings(key='{self.key}', value='{self.value}', type='{self.value_type}')>"


class RequestQueue(Base):
    __tablename__ = "request_queue"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    request_data = Column(Text, nullable=False)   # JSON string
    priority = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    status = Column(String, nullable=False, default=QueueStatus.QUEUED.value)

    # Relationships
    model = relationship("Model", back_populates="queue_items")

    def get_request_data(self) -> Dict[str, Any]:
        return json.loads(self.request_data)

    def set_request_data(self, data: Dict[str, Any]):
        self.request_data = json.dumps(data)

    def start_processing(self):
        self.status = QueueStatus.PROCESSING.value
        self.started_at = datetime.utcnow()

    def __repr__(self):
        return f"<RequestQueue(id={self.id}, session_id='{self.session_id}', status='{self.status}')>"


# Create indexes
Index("idx_models_name", Model.name)
Index("idx_models_status", Model.status)
Index("idx_models_last_used", Model.last_used_at)
Index("idx_inference_sessions_model_id", InferenceSession.model_id)
Index("idx_inference_requests_session_id", InferenceRequest.session_id)
Index("idx_inference_requests_model_id", InferenceRequest.model_id)
Index("idx_inference_requests_created_at", InferenceRequest.created_at)
Index("idx_system_metrics_timestamp", SystemMetrics.timestamp)
Index("idx_request_queue_status", RequestQueue.status)
Index("idx_request_queue_priority", RequestQueue.priority)