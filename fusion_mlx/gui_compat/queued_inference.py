"""
Transparent queued inference wrapper that maintains OpenAI compatibility.

This module provides drop-in replacements for direct model inference calls
that automatically handle queuing when models are busy, making the queuing
completely transparent to API consumers.
"""

import asyncio
import logging
import uuid
from typing import AsyncIterator, Optional, List, Dict, Any

from .inference_queue_manager import get_inference_manager, QueuedRequest
from .model_manager import get_model_manager
from .mlx_integration import GenerationConfig

logger = logging.getLogger(__name__)


async def generate_text_queued(
    model_name: str,
    prompt: str,
    config: GenerationConfig,
    priority: int = 5
):
    """
    Generate text with transparent queuing.

    This function provides the same interface as model_manager.generate_text()
    but automatically handles queuing when the model is busy.
    """
    model_manager = get_model_manager()
    inference_manager = get_inference_manager()

    # Check if we can process immediately
    queue_status = inference_manager.get_queue_status(model_name)

    if queue_status.get("can_accept_immediate", False):
        # Model is available, process directly
        logger.debug(f"Processing {model_name} request immediately")
        return await model_manager.generate_text(model_name, prompt, config)

    # Model is busy, use queue
    logger.debug(f"Queuing {model_name} request (active: {queue_status.get('active_requests', 0)})")

    session_id = str(uuid.uuid4())

    # Create queued request
    queued_request = QueuedRequest(
        session_id=session_id,
        model_name=model_name,
        prompt=prompt,
        config=config,
        priority=priority,
        streaming=False
    )

    # Use a completion future to wait for result
    result_future = asyncio.Future()

    def completion_callback(request_id: str, success: bool, result):
        if success:
            result_future.set_result(result)
        else:
            result_future.set_exception(Exception(result))

    queued_request.callback = completion_callback

    # Queue the request
    request_id = await inference_manager.queue_request(queued_request)

    # Wait for completion with timeout
    try:
        result = await asyncio.wait_for(result_future, timeout=300.0)  # 5 minute timeout
        return result
    except asyncio.TimeoutError:
        raise RuntimeError("Request timed out after 5 minutes")


async def generate_text_stream_queued(
    model_name: str,
    prompt: str,
    config: GenerationConfig,
    priority: int = 5
) -> AsyncIterator[str]:
    """
    Generate streaming text with transparent queuing.

    This function provides the same interface as model_manager.generate_text_stream()
    but automatically handles queuing when the model is busy.
    """
    model_manager = get_model_manager()
    inference_manager = get_inference_manager()

    # Check if we can process immediately
    queue_status = inference_manager.get_queue_status(model_name)

    if queue_status.get("can_accept_immediate", False):
        # Model is available, stream directly
        logger.debug(f"Streaming {model_name} request immediately")
        async for chunk in model_manager.generate_text_stream(model_name, prompt, config):
            yield chunk
        return

    # Model is busy, wait for turn then stream
    logger.debug(f"Queuing {model_name} streaming request (active: {queue_status.get('active_requests', 0)})")

    session_id = str(uuid.uuid4())

    # Use an event to signal when streaming can start
    stream_ready_event = asyncio.Event()
    stream_generator = None
    error_exception = None

    def stream_ready_callback(request_id: str, success: bool, generator_or_error):
        nonlocal stream_generator, error_exception
        if success:
            stream_generator = generator_or_error
        else:
            error_exception = Exception(generator_or_error)
        stream_ready_event.set()

    # Create a special queued request for streaming
    queued_request = QueuedRequest(
        session_id=session_id,
        model_name=model_name,
        prompt=prompt,
        config=config,
        priority=priority,
        streaming=True,
        stream_callback=stream_ready_callback
    )

    # Queue the request
    request_id = await inference_manager.queue_request(queued_request)

    # Wait for streaming to be ready
    try:
        await asyncio.wait_for(stream_ready_event.wait(), timeout=300.0)
    except asyncio.TimeoutError:
        raise RuntimeError("Request timed out after 5 minutes")

    # Check for errors
    if error_exception:
        raise error_exception

    # Stream the results
    if stream_generator:
        async for chunk in stream_generator:
            yield chunk


async def queued_transcribe_audio(
    model_name: str,
    file_path: str,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    temperature: float = 0.0,
    priority: int = 5
):
    """
    Transcribe audio with transparent queuing.

    Drop-in replacement for loaded_model.mlx_wrapper.transcribe_audio() with queuing.
    """
    model_manager = get_model_manager()
    inference_manager = get_inference_manager()

    # Check if we can process immediately
    queue_status = inference_manager.get_queue_status(model_name)

    if queue_status.get("can_accept_immediate", False):
        # Model is available, process directly
        logger.debug(f"Processing {model_name} transcription immediately")
        loaded_model = model_manager.get_model_for_inference(model_name)
        if not loaded_model:
            raise RuntimeError(f"Model {model_name} is not loaded")

        result = loaded_model.mlx_wrapper.transcribe_audio(
            file_path,
            language=language,
            initial_prompt=initial_prompt,
            temperature=temperature
        )

        # Update model usage count for direct transcription
        try:
            from .database import get_database_manager
            from .models import Model
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                model_record = session.query(Model).filter(Model.name == model_name).first()
                if model_record:
                    model_record.increment_use_count()
                    session.commit()
        except Exception as e:
            logger.error(f"Error updating transcription model usage: {e}")

        # Format result consistently
        if isinstance(result, dict) and 'text' in result:
            return {"text": result['text']}
        elif isinstance(result, str):
            return {"text": result}
        else:
            return {"text": str(result)}

    # Model is busy, use queue
    logger.debug(f"Queuing {model_name} transcription request (active: {queue_status.get('active_requests', 0)})")

    session_id = str(uuid.uuid4())

    # Create dummy config for audio requests
    dummy_config = GenerationConfig(max_tokens=1)

    # Create queued request
    queued_request = QueuedRequest(
        session_id=session_id,
        model_name=model_name,
        prompt="",  # Not used for transcription
        config=dummy_config,
        priority=priority,
        request_type="transcription",
        audio_data={
            "file_path": file_path,
            "language": language,
            "initial_prompt": initial_prompt,
            "temperature": temperature
        }
    )

    # Use a completion future to wait for result
    result_future = asyncio.Future()

    def completion_callback(request_id: str, success: bool, result):
        if success:
            result_future.set_result(result)
        else:
            result_future.set_exception(Exception(result))

    queued_request.callback = completion_callback

    # Queue the request
    request_id = await inference_manager.queue_request(queued_request)

    # Wait for completion with timeout
    try:
        result = await asyncio.wait_for(result_future, timeout=300.0)  # 5 minute timeout
        return result
    except asyncio.TimeoutError:
        raise RuntimeError("Transcription request timed out after 5 minutes")


async def queued_generate_speech(
    text: str,
    voice: str = "kokoro",
    speed: float = 1.0,
    priority: int = 5
):
    """
    Generate speech with transparent queuing.

    Drop-in replacement for mlx_audio.tts.generate() with queuing.
    """
    inference_manager = get_inference_manager()

    # For TTS, we don't need a specific model to be loaded
    # We'll use a dummy model name for queue management
    model_name = f"tts-{voice}"

    # Check if we can process immediately
    queue_status = inference_manager.get_queue_status(model_name)

    if queue_status.get("can_accept_immediate", False):
        # Process directly
        logger.debug(f"Processing TTS request immediately")
        try:
            import mlx_audio
            import tempfile
            import os
        except ImportError:
            raise RuntimeError("MLX Audio not installed")

        # Generate speech
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            mlx_audio.tts.generate(
                text=text,
                model=voice,
                output_file=temp_file.name,
                speed=speed
            )

            # Read generated audio
            with open(temp_file.name, "rb") as audio_file:
                audio_content = audio_file.read()

            # Clean up
            os.unlink(temp_file.name)

            return audio_content

    # Use queue for TTS
    logger.debug(f"Queuing TTS request (active: {queue_status.get('active_requests', 0)})")

    session_id = str(uuid.uuid4())

    # Create dummy config for TTS requests
    dummy_config = GenerationConfig(max_tokens=1)

    # Create queued request
    queued_request = QueuedRequest(
        session_id=session_id,
        model_name=model_name,
        prompt=text,
        config=dummy_config,
        priority=priority,
        request_type="tts",
        audio_data={
            "voice": voice,
            "speed": speed
        }
    )

    # Use a completion future to wait for result
    result_future = asyncio.Future()

    def completion_callback(request_id: str, success: bool, result):
        if success:
            result_future.set_result(result)
        else:
            result_future.set_exception(Exception(result))

    queued_request.callback = completion_callback

    # Queue the request
    request_id = await inference_manager.queue_request(queued_request)

    # Wait for completion with timeout
    try:
        result = await asyncio.wait_for(result_future, timeout=300.0)  # 5 minute timeout
        return result
    except asyncio.TimeoutError:
        raise RuntimeError("TTS request timed out after 5 minutes")


# Convenience functions that match the model_manager interface
async def queued_generate_text(model_name: str, prompt: str, config: GenerationConfig):
    """Drop-in replacement for model_manager.generate_text() with queuing."""
    return await generate_text_queued(model_name, prompt, config)


async def queued_generate_text_stream(model_name: str, prompt: str, config: GenerationConfig):
    """Drop-in replacement for model_manager.generate_text_stream() with queuing."""
    async for chunk in generate_text_stream_queued(model_name, prompt, config):
        yield chunk


async def queued_generate_embeddings(
    model_name: str,
    texts: list[str],
    priority: int = 5
):
    """
    Generate embeddings with transparent queuing.

    Drop-in replacement for loaded_model.mlx_wrapper.generate_embeddings() with queuing.
    """
    model_manager = get_model_manager()
    inference_manager = get_inference_manager()

    # Check if we can process immediately
    queue_status = inference_manager.get_queue_status(model_name)

    if queue_status.get("can_accept_immediate", False):
        # Model is available, process directly
        logger.debug(f"Processing {model_name} embeddings immediately")
        loaded_model = model_manager.get_model_for_inference(model_name)
        if not loaded_model:
            raise RuntimeError(f"Model {model_name} is not loaded")

        # Proceed if the wrapper exposes an embedding helper; otherwise try and catch errors later.
        if not hasattr(loaded_model.mlx_wrapper, 'generate_embeddings'):
            logger.warning(
                f"Wrapper for {model_name} does not explicitly expose generate_embeddings; attempting generic fallback"
            )

        result = loaded_model.mlx_wrapper.generate_embeddings(texts)

        # Update model usage count for direct embeddings
        try:
            from .database import get_database_manager
            from .models import Model
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                model_record = session.query(Model).filter(Model.name == model_name).first()
                if model_record:
                    model_record.increment_use_count()
                    session.commit()
        except Exception as e:
            logger.error(f"Error updating embeddings model usage: {e}")

        # Ensure result is in expected format
        logger.debug(f"Embedding result type: {type(result)}")
        if isinstance(result, list):
            return {
                "embeddings": result,
                "prompt_tokens": sum(len(text.split()) for text in texts),
                "total_tokens": sum(len(text.split()) for text in texts)
            }
        elif hasattr(result, 'tolist'):
            # Handle numpy arrays or MLX arrays
            result_list = result.tolist()
            return {
                "embeddings": result_list,
                "prompt_tokens": sum(len(text.split()) for text in texts),
                "total_tokens": sum(len(text.split()) for text in texts)
            }
        else:
            # Assume it's already a dict-like result
            return result

    # Model is busy, use queue
    logger.debug(f"Queuing {model_name} embeddings request (active: {queue_status.get('active_requests', 0)})")

    session_id = str(uuid.uuid4())

    # Create dummy config for embedding requests
    dummy_config = GenerationConfig(max_tokens=1)

    # Create queued request
    queued_request = QueuedRequest(
        session_id=session_id,
        model_name=model_name,
        prompt="",  # Not used for embeddings
        config=dummy_config,
        priority=priority,
        request_type="embeddings",
        audio_data={  # Reuse audio_data field for embedding data
            "texts": texts
        }
    )

    # Use a completion future to wait for result
    result_future = asyncio.Future()

    def completion_callback(request_id: str, success: bool, result):
        if success:
            result_future.set_result(result)
        else:
            result_future.set_exception(Exception(result))

    queued_request.callback = completion_callback

    # Queue the request
    request_id = await inference_manager.queue_request(queued_request)

    # Wait for completion with timeout
    try:
        result = await asyncio.wait_for(result_future, timeout=300.0)  # 5 minute timeout
        return result
    except asyncio.TimeoutError:
        raise RuntimeError("Embeddings request timed out after 5 minutes")


async def queued_generate_vision(
    model_name: str,
    messages: List[Dict[str, Any]],
    image_file_paths: list[str],
    config: GenerationConfig,
    priority: int = 5
):
    """Queued generation for vision models with structured messages."""
    model_manager = get_model_manager()
    inference_manager = get_inference_manager()

    # Check if we can process immediately
    queue_status = inference_manager.get_queue_status(model_name)

    if queue_status.get("can_accept_immediate", False):
        # Model is available, process directly
        logger.debug(f"Processing {model_name} vision request immediately")
        loaded_model = model_manager.get_model_for_inference(model_name)
        if not loaded_model:
            raise RuntimeError(f"Model {model_name} is not loaded")

        # Check if this is a vision model
        if not hasattr(loaded_model.mlx_wrapper, 'generate_with_images'):
            raise RuntimeError(f"Model {model_name} is not a vision model")

        result = loaded_model.mlx_wrapper.generate_with_images(messages, image_file_paths, config)

        # Update model usage count for direct vision generation
        try:
            from .database import get_database_manager
            from .models import Model
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                model_record = session.query(Model).filter(Model.name == model_name).first()
                if model_record:
                    model_record.increment_use_count()
                    session.commit()
        except Exception as e:
            logger.error(f"Error updating vision model usage: {e}")

        return result

    # Model is busy, use queue
    logger.debug(f"Queuing {model_name} vision request (active: {queue_status.get('active_requests', 0)})")

    session_id = str(uuid.uuid4())

    # Create queued request
    queued_request = QueuedRequest(
        session_id=session_id,
        model_name=model_name,
        prompt=messages,  # Pass structured messages in the prompt field
        config=config,
        priority=priority,
        request_type="vision_generation",
        audio_data={  # Reuse audio_data field for vision data
            "image_file_paths": image_file_paths
        }
    )

    # Use a completion future to wait for result
    result_future = asyncio.Future()

    def completion_callback(request_id: str, success: bool, result):
        if success:
            result_future.set_result(result)
        else:
            result_future.set_exception(Exception(result))

    queued_request.callback = completion_callback

    # Queue the request
    request_id = await inference_manager.queue_request(queued_request)

    # Wait for completion with timeout
    try:
        result = await asyncio.wait_for(result_future, timeout=300.0)  # 5 minute timeout
        return result
    except asyncio.TimeoutError:
        raise RuntimeError("Vision request timed out after 5 minutes")