"""
Inference Request Queue Manager

Handles queuing and processing of inference requests to prevent
concurrency issues and provide better user experience for 2-5 concurrent users.
"""

import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from .database import get_database_manager
from .models import Model, RequestQueue, InferenceRequest, QueueStatus, InferenceStatus
from .mlx_integration import GenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class QueuedRequest:
    """Represents a queued inference request."""
    session_id: str
    model_name: str
    prompt: str
    config: GenerationConfig
    priority: int = 0
    created_at: datetime = None
    callback: Optional[Callable] = None
    streaming: bool = False
    stream_callback: Optional[Callable] = None  # For streaming responses
    request_type: str = "text"  # "text", "transcription", "tts", "embeddings", "vision_generation"
    audio_data: Optional[Dict] = None  # For audio requests

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class InferenceRequestManager:
    """
    Manages queuing and processing of inference requests.

    Features:
    - Per-model concurrency limits
    - FIFO queuing with priority support
    - Request serialization
    - Background processing
    """

    def __init__(self, max_concurrent_per_model: int = 1):
        self.max_concurrent_per_model = max_concurrent_per_model

        # Track active requests per model
        self._active_requests: Dict[str, int] = {}
        self._lock = threading.RLock()

        # Background worker
        self._worker_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_requested = False

        # Request tracking for callbacks
        self._pending_callbacks: Dict[str, Callable] = {}

    def start(self):
        """Start the background queue processor."""
        if not self._worker_running:
            self._worker_running = True
            self._shutdown_requested = False
            self._worker_thread = threading.Thread(
                target=self._queue_worker,
                name="inference_queue_worker",
                daemon=True
            )
            self._worker_thread.start()
            logger.info("Inference queue worker started")

    def stop(self):
        """Stop the background queue processor."""
        self._shutdown_requested = True
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        self._worker_running = False
        logger.info("Inference queue worker stopped")

    async def queue_request(self, request: QueuedRequest) -> str:
        """
        Queue an inference request.

        Returns:
            str: Request ID for tracking
        """
        request_id = str(uuid.uuid4())

        with self._lock:
            # Check if we can process immediately
            active_count = self._active_requests.get(request.model_name, 0)
            can_process_now = active_count < self.max_concurrent_per_model

            # Always queue the request for persistence and tracking
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                # Get model record
                model_record = session.query(Model).filter(Model.name == request.model_name).first()
                if not model_record:
                    raise ValueError(f"Model {request.model_name} not found in database")

                # Create queue entry
                queue_item = RequestQueue(
                    session_id=request.session_id,
                    model_id=model_record.id,
                    priority=request.priority,
                    status=QueueStatus.QUEUED.value if not can_process_now else QueueStatus.PROCESSING.value
                )

                # Store request data
                request_data = {
                    "request_id": request_id,
                    "prompt": request.prompt,
                    "streaming": request.streaming,
                    "request_type": request.request_type,
                    "audio_data": request.audio_data,
                    "config": {
                        "max_tokens": request.config.max_tokens,
                        "temperature": request.config.temperature,
                        "top_p": request.config.top_p,
                        "top_k": request.config.top_k,
                        "repetition_penalty": request.config.repetition_penalty,
                        "seed": request.config.seed
                    }
                }
                queue_item.set_request_data(request_data)

                session.add(queue_item)
                session.commit()

                if can_process_now:
                    # Mark as processing immediately
                    queue_item.start_processing()
                    self._active_requests[request.model_name] = active_count + 1
                    session.commit()
                    logger.info(f"Processing request {request_id} immediately for model {request.model_name}")
                else:
                    logger.info(f"Queued request {request_id} for model {request.model_name} (active: {active_count})")

                # Store callback if provided
                if request.callback:
                    self._pending_callbacks[request_id] = request.callback
                elif request.stream_callback:
                    self._pending_callbacks[request_id] = request.stream_callback

        return request_id

    def get_queue_status(self, model_name: str) -> Dict[str, Any]:
        """Get queue status for a specific model."""
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            model_record = session.query(Model).filter(Model.name == model_name).first()
            if not model_record:
                return {"error": "Model not found"}

            # Count queued requests
            queued_count = session.query(RequestQueue).filter(
                and_(
                    RequestQueue.model_id == model_record.id,
                    RequestQueue.status == QueueStatus.QUEUED.value
                )
            ).count()

            # Count processing requests
            processing_count = session.query(RequestQueue).filter(
                and_(
                    RequestQueue.model_id == model_record.id,
                    RequestQueue.status == QueueStatus.PROCESSING.value
                )
            ).count()

            with self._lock:
                active_count = self._active_requests.get(model_name, 0)

            return {
                "model_name": model_name,
                "queued_requests": queued_count,
                "processing_requests": processing_count,
                "active_requests": active_count,
                "max_concurrent": self.max_concurrent_per_model,
                "can_accept_immediate": active_count < self.max_concurrent_per_model
            }

    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific request."""
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            # Check queue first
            queue_item = session.query(RequestQueue).filter(
                RequestQueue.request_data.like(f'%"request_id": "{request_id}"%')
            ).first()

            if queue_item:
                request_data = queue_item.get_request_data()
                return {
                    "request_id": request_id,
                    "status": queue_item.status,
                    "created_at": queue_item.created_at.isoformat(),
                    "started_at": queue_item.started_at.isoformat() if queue_item.started_at else None,
                    "model_name": queue_item.model.name,
                    "position_in_queue": self._get_queue_position(queue_item)
                }

            # Check completed requests
            inference_request = session.query(InferenceRequest).filter(
                InferenceRequest.request_id == request_id
            ).first()

            if inference_request:
                return {
                    "request_id": request_id,
                    "status": inference_request.status,
                    "created_at": inference_request.created_at.isoformat(),
                    "completed_at": inference_request.completed_at.isoformat() if inference_request.completed_at else None,
                    "duration_ms": inference_request.duration_ms,
                    "error_message": inference_request.error_message
                }

            return None

    def _get_queue_position(self, queue_item: RequestQueue) -> int:
        """Get position of a queue item in the queue."""
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            position = session.query(RequestQueue).filter(
                and_(
                    RequestQueue.model_id == queue_item.model_id,
                    RequestQueue.status == QueueStatus.QUEUED.value,
                    or_(
                        RequestQueue.priority > queue_item.priority,
                        and_(
                            RequestQueue.priority == queue_item.priority,
                            RequestQueue.created_at < queue_item.created_at
                        )
                    )
                )
            ).count()

            return position + 1

    def _queue_worker(self):
        """Background worker that processes queued requests."""
        logger.info("Inference queue worker started")

        while not self._shutdown_requested:
            try:
                self._process_next_requests()
                # Check every 2 seconds
                threading.Event().wait(2.0)

            except Exception as e:
                logger.error(f"Error in queue worker: {e}")
                threading.Event().wait(5.0)  # Wait longer on errors

        logger.info("Inference queue worker stopped")

    def _process_next_requests(self):
        """Process next available requests from all model queues."""
        db_manager = get_database_manager()

        with db_manager.get_session() as session:
            # Get all models with queued requests
            models_with_queue = session.query(Model).join(RequestQueue).filter(
                RequestQueue.status == QueueStatus.QUEUED.value
            ).distinct().all()

            for model in models_with_queue:
                with self._lock:
                    active_count = self._active_requests.get(model.name, 0)
                    can_process = active_count < self.max_concurrent_per_model

                if can_process:
                    # Get next request for this model
                    next_request = session.query(RequestQueue).filter(
                        and_(
                            RequestQueue.model_id == model.id,
                            RequestQueue.status == QueueStatus.QUEUED.value
                        )
                    ).order_by(
                        desc(RequestQueue.priority),
                        RequestQueue.created_at
                    ).first()

                    if next_request:
                        # Mark as processing
                        next_request.start_processing()
                        with self._lock:
                            self._active_requests[model.name] = active_count + 1
                        session.commit()

                        # Process in background thread to avoid blocking
                        processing_thread = threading.Thread(
                            target=self._process_request,
                            args=(next_request.id,),
                            name=f"process_request_{next_request.id}",
                            daemon=True
                        )
                        processing_thread.start()

    def _process_request(self, queue_item_id: int):
        """Process a single request."""
        try:
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                queue_item = session.query(RequestQueue).filter(RequestQueue.id == queue_item_id).first()
                if not queue_item:
                    logger.error(f"Queue item {queue_item_id} not found")
                    return

                request_data = queue_item.get_request_data()
                request_id = request_data["request_id"]
                model_name = queue_item.model.name

                logger.info(f"Processing inference request {request_id} for model {model_name}")

                # Create inference request record
                inference_request = InferenceRequest(
                    request_id=request_id,
                    session_id=queue_item.session_id,
                    model_id=queue_item.model_id,
                    status=InferenceStatus.PROCESSING.value
                )
                inference_request.set_input_data(request_data)
                session.add(inference_request)
                session.commit()

                try:
                    # Import here to avoid circular imports
                    from .model_manager import get_model_manager

                    # Ensure model is loaded
                    model_manager = get_model_manager()
                    loaded_model = model_manager.get_model_for_inference(model_name)
                    if not loaded_model:
                        raise RuntimeError(f"Model {model_name} is not loaded")

                    # Create generation config
                    config_data = request_data["config"]
                    config = GenerationConfig(
                        max_tokens=config_data["max_tokens"],
                        temperature=config_data["temperature"],
                        top_p=config_data["top_p"],
                        top_k=config_data["top_k"],
                        repetition_penalty=config_data["repetition_penalty"],
                        seed=config_data["seed"]
                    )

                    # Check request type and streaming
                    request_type = request_data.get("request_type", "text")
                    is_streaming = request_data.get("streaming", False)

                    if request_type == "transcription":
                        # Handle audio transcription
                        audio_data = request_data.get("audio_data", {})
                        result = loaded_model.mlx_wrapper.transcribe_audio(
                            audio_data["file_path"],
                            language=audio_data.get("language"),
                            initial_prompt=audio_data.get("initial_prompt"),
                            temperature=audio_data.get("temperature", 0.0)
                        )

                        # Format result
                        if isinstance(result, dict) and 'text' in result:
                            text_content = result['text']
                        elif isinstance(result, str):
                            text_content = result
                        else:
                            text_content = str(result)

                        output_data = {"text": text_content, "type": "transcription"}
                        inference_request.mark_completed(output_data)

                        # Update model usage count for transcription
                        try:
                            model_record = session.query(Model).filter(Model.name == model_name).first()
                            if model_record:
                                model_record.increment_use_count()
                        except Exception as e:
                            logger.error(f"Error updating transcription model usage: {e}")

                        session.commit()

                        # Call callback
                        callback = self._pending_callbacks.pop(request_id, None)
                        if callback:
                            callback(request_id, True, {"text": text_content})

                        logger.info(f"Completed transcription request {request_id} for model {model_name}")

                    elif request_type == "tts":
                        # Handle text-to-speech
                        audio_data = request_data.get("audio_data", {})

                        # Import MLX Audio
                        try:
                            import mlx_audio
                            import tempfile
                            import os
                        except ImportError:
                            raise RuntimeError("MLX Audio not installed")

                        # Generate speech
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                            mlx_audio.tts.generate(
                                text=request_data["prompt"],
                                model=audio_data.get("voice", "kokoro"),
                                output_file=temp_file.name,
                                speed=audio_data.get("speed", 1.0)
                            )

                            # Read generated audio
                            with open(temp_file.name, "rb") as audio_file:
                                audio_content = audio_file.read()

                            # Clean up
                            os.unlink(temp_file.name)

                        output_data = {"audio_content": len(audio_content), "type": "tts"}
                        inference_request.mark_completed(output_data)

                        # Update model usage count for TTS (if model exists in database)
                        try:
                            model_record = session.query(Model).filter(Model.name == model_name).first()
                            if model_record:
                                model_record.increment_use_count()
                            else:
                                # TTS models might not be in database, log for info
                                logger.debug(f"TTS model {model_name} not found in database, skipping usage count")
                        except Exception as e:
                            logger.error(f"Error updating TTS model usage: {e}")

                        session.commit()

                        # Call callback with audio content
                        callback = self._pending_callbacks.pop(request_id, None)
                        if callback:
                            callback(request_id, True, audio_content)

                        logger.info(f"Completed TTS request {request_id} for model {model_name}")

                    elif request_type == "embeddings":
                        # Handle embeddings generation
                        audio_data = request_data.get("audio_data", {})  # Reused for embedding data
                        texts = audio_data.get("texts", [])

                        if not texts:
                            raise ValueError("No texts provided for embeddings")

                        # If the helper is missing, attempt generic fallback and let errors bubble up
                        if not hasattr(loaded_model.mlx_wrapper, 'generate_embeddings'):
                            logger.warning(
                                f"Wrapper for {model_name} lacks generate_embeddings; attempting generic fallback"
                            )

                        # Generate embeddings
                        result = loaded_model.mlx_wrapper.generate_embeddings(texts)

                        # Format result
                        if isinstance(result, list):
                            embeddings = result
                            prompt_tokens = sum(len(text.split()) for text in texts)
                            total_tokens = prompt_tokens
                        else:
                            # Assume result is a dict with embeddings and token counts
                            embeddings = result.get("embeddings", [])
                            prompt_tokens = result.get("prompt_tokens", sum(len(text.split()) for text in texts))
                            total_tokens = result.get("total_tokens", prompt_tokens)

                        output_data = {
                            "embeddings": embeddings,
                            "prompt_tokens": prompt_tokens,
                            "total_tokens": total_tokens,
                            "type": "embeddings"
                        }
                        inference_request.mark_completed(output_data)

                        # Update model usage count for embeddings
                        try:
                            model_record = session.query(Model).filter(Model.name == model_name).first()
                            if model_record:
                                model_record.increment_use_count()
                        except Exception as e:
                            logger.error(f"Error updating embeddings model usage: {e}")

                        session.commit()

                        # Call callback
                        callback = self._pending_callbacks.pop(request_id, None)
                        if callback:
                            callback(request_id, True, output_data)

                        logger.info(f"Completed embeddings request {request_id} for model {model_name}")

                    elif request_type == "vision_generation":
                        # For vision, the 'prompt' field contains the structured messages
                        messages = request_data.get("prompt", [])
                        images = request_data.get("audio_data", {}).get("image_file_paths", [])

                        if not messages:
                            raise ValueError("No messages provided for vision generation")

                        # Generate with vision model using structured messages
                        result = loaded_model.mlx_wrapper.generate_with_images(messages, images, config)

                        # Mark as completed
                        output_data = {
                            "text": result.text,
                            "tokens_generated": result.tokens_generated,
                            "generation_time": result.generation_time_ms,
                            "type": "vision"
                        }
                        inference_request.mark_completed(output_data)

                        # Update model usage count for vision
                        try:
                            model_record = session.query(Model).filter(Model.name == model_name).first()
                            if model_record:
                                model_record.increment_use_count()
                        except Exception as e:
                            logger.error(f"Error updating vision model usage: {e}")

                        session.commit()

                        # Call callback
                        callback = self._pending_callbacks.pop(request_id, None)
                        if callback:
                            callback(request_id, True, result)

                        logger.info(f"Completed vision request {request_id} for model {model_name}")

                    elif is_streaming:
                        # For streaming, just notify that streaming can start
                        # The actual streaming happens in the API layer
                        callback = self._pending_callbacks.pop(request_id, None)
                        if callback:
                            # Return a generator that will be consumed by the API
                            generator = model_manager.generate_text_stream(
                                model_name,
                                request_data["prompt"],
                                config
                            )
                            callback(request_id, True, generator)

                        # Mark as completed immediately for streaming
                        output_data = {"streaming": True}
                        inference_request.mark_completed(output_data)
                        session.commit()

                        logger.info(f"Started streaming request {request_id} for model {model_name}")
                    else:
                        # Handle standard text generation
                        prompt = request_data.get("prompt", "")

                        # Generate response (sync call in background thread)
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(
                                model_manager.generate_text(
                                    model_name,
                                    prompt,
                                    config
                                )
                            )
                        finally:
                            loop.close()

                        # Mark as completed
                        output_data = {
                            "text": result.text,
                            "tokens_generated": result.tokens_generated,
                            "generation_time": result.generation_time_ms
                        }
                        inference_request.mark_completed(output_data)
                        session.commit()

                        # Call callback if exists
                        callback = self._pending_callbacks.pop(request_id, None)
                        if callback:
                            callback(request_id, True, result)

                        logger.info(f"Completed inference request {request_id} for model {model_name}")

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Failed to process request {request_id}: {error_msg}")

                    # Mark as failed
                    inference_request.mark_failed(error_msg)
                    session.commit()

                    # Call callback if exists
                    callback = self._pending_callbacks.pop(request_id, None)
                    if callback:
                        callback(request_id, False, error_msg)

                finally:
                    # Remove from queue and decrement active count
                    session.delete(queue_item)
                    session.commit()

                    with self._lock:
                        current_count = self._active_requests.get(model_name, 0)
                        self._active_requests[model_name] = max(0, current_count - 1)

        except Exception as e:
            logger.error(f"Error processing request {queue_item_id}: {e}")


# Global instance
_inference_manager: Optional[InferenceRequestManager] = None


def get_inference_manager() -> InferenceRequestManager:
    """Get the global inference request manager."""
    global _inference_manager
    if _inference_manager is None:
        # Read max concurrent per model from settings
        try:
            from .database import get_database_manager
            db_manager = get_database_manager()
            max_concurrent = db_manager.get_setting("max_concurrent_requests_per_model", 1)
        except Exception:
            max_concurrent = 1

        _inference_manager = InferenceRequestManager(max_concurrent_per_model=max_concurrent)
        _inference_manager.start()

    return _inference_manager


def shutdown_inference_manager():
    """Shutdown the global inference manager."""
    global _inference_manager
    if _inference_manager:
        _inference_manager.stop()
        _inference_manager = None