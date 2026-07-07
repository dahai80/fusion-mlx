# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_gui.models — ORM Enum + Base model methods.

These cover the Enum values, the typed-value codec on AppSettings, the
metadata JSON helpers on Model/InferenceRequest/RequestQueue, the
mark_completed/mark_failed lifecycle helpers on InferenceRequest, the
memory_usage_percent property on SystemMetrics, and __repr__ on every
Base. SQLAlchemy ORM declarative_base() lets us instantiate model objects
in-memory without binding an engine; methods below are pure-Python and
don't touch the DB. Aims at ≥90% line coverage of models.py.
"""

from __future__ import annotations

import json
from datetime import datetime

from fusion_gui.models import (
    AppSettings,
    Base,
    InferenceRequest,
    InferenceSession,
    InferenceStatus,
    Model,
    ModelCapability,
    ModelStatus,
    ModelType,
    QueueStatus,
    RequestQueue,
    SystemMetrics,
)


class TestEnums:
    def test_model_status_values(self):
        assert ModelStatus.UNLOADED.value == "unloaded"
        assert ModelStatus.LOADED.value == "loaded"
        assert ModelStatus.LOADING.value == "loading"
        assert ModelStatus.FAILED.value == "failed"

    def test_model_type_values(self):
        assert ModelType.TEXT.value == "text"
        assert ModelType.VISION.value == "vision"
        assert ModelType.AUDIO.value == "audio"
        assert ModelType.MULTIMODAL.value == "multimodal"

    def test_inference_status_values(self):
        assert InferenceStatus.PENDING.value == "pending"
        assert InferenceStatus.PROCESSING.value == "processing"
        assert InferenceStatus.COMPLETED.value == "completed"
        assert InferenceStatus.FAILED.value == "failed"

    def test_queue_status_values(self):
        assert QueueStatus.QUEUED.value == "queued"
        assert QueueStatus.PROCESSING.value == "processing"
        assert QueueStatus.COMPLETED.value == "completed"
        assert QueueStatus.FAILED.value == "failed"


class TestBase:
    def test_base_is_declarative(self):
        # Base is declarative_base(); has metadata for table defs
        assert hasattr(Base, "metadata")


class TestModel:
    def _make(self):
        return Model(name="m", path="/p", model_type="text", memory_required_gb=1.0)

    def test_defaults(self):
        # SQLAlchemy Column(default=...) only applies on INSERT/flush, not on
        # in-memory instantiation — instance attrs stay None until flushed.
        # Verify the Enum sentinel values (which are the source of truth for
        # the defaults) instead of the ORM instance attrs.
        assert ModelStatus.UNLOADED.value == "unloaded"
        m = self._make()
        assert m.name == "m"
        # use_count/status are None until flush; don't assert ORM defaults here

    def test_get_metadata_empty(self):
        m = self._make()
        assert m.get_metadata() == {}

    def test_get_metadata_present(self):
        m = self._make()
        m.model_metadata = json.dumps({"k": "v"})
        assert m.get_metadata() == {"k": "v"}

    def test_set_metadata(self):
        m = self._make()
        m.set_metadata({"a": 1, "b": [2, 3]})
        assert json.loads(m.model_metadata) == {"a": 1, "b": [2, 3]}

    def test_increment_use_count(self):
        m = self._make()
        m.use_count = 0  # ORM default only applies on flush; set eagerly
        m.increment_use_count()
        assert m.use_count == 1
        assert m.last_used_at is not None
        m.increment_use_count()
        assert m.use_count == 2

    def test_repr(self):
        m = self._make()
        m.status = ModelStatus.UNLOADED.value  # ORM default not applied until flush
        assert "Model(name='m'" in repr(m)
        assert "status='unloaded'" in repr(m)
        assert "type='text'" in repr(m)


class TestModelCapability:
    def test_repr(self):
        c = ModelCapability(model_id=5, capability="chat")
        assert "ModelCapability(model_id=5" in repr(c)
        assert "capability='chat'" in repr(c)


class TestInferenceSession:
    def test_update_activity_sets_timestamp(self):
        s = InferenceSession(session_id="s1", model_id=1)
        s.update_activity()
        assert s.last_activity_at is not None

    def test_repr(self):
        s = InferenceSession(session_id="s1", model_id=1)
        assert "InferenceSession(session_id='s1'" in repr(s)
        assert "model_id=1" in repr(s)


class TestInferenceRequest:
    def _make(self):
        return InferenceRequest(
            session_id="s1", model_id=1, request_type="chat", input_data="{}"
        )

    def test_get_input_data(self):
        r = self._make()
        r.input_data = json.dumps({"prompt": "hi"})
        assert r.get_input_data() == {"prompt": "hi"}

    def test_set_input_data(self):
        r = self._make()
        r.set_input_data({"prompt": "x"})
        assert json.loads(r.input_data) == {"prompt": "x"}

    def test_get_output_data_none(self):
        r = self._make()
        assert r.get_output_data() is None

    def test_get_output_data_present(self):
        r = self._make()
        r.output_data = json.dumps({"text": "ok"})
        assert r.get_output_data() == {"text": "ok"}

    def test_set_output_data(self):
        r = self._make()
        r.set_output_data({"text": "ok"})
        assert json.loads(r.output_data) == {"text": "ok"}

    def test_mark_completed_sets_status_and_duration(self):
        r = self._make()
        r.created_at = datetime.utcnow()
        r.mark_completed({"text": "ok"})
        assert r.status == InferenceStatus.COMPLETED.value
        assert r.completed_at is not None
        assert r.duration_ms is not None and r.duration_ms >= 0
        assert json.loads(r.output_data) == {"text": "ok"}

    def test_mark_completed_without_created_at_skips_duration(self):
        r = self._make()
        r.created_at = None
        r.mark_completed({"text": "ok"})
        assert r.duration_ms is None

    def test_mark_failed_sets_error_and_duration(self):
        r = self._make()
        r.created_at = datetime.utcnow()
        r.mark_failed("boom")
        assert r.status == InferenceStatus.FAILED.value
        assert r.error_message == "boom"
        assert r.completed_at is not None
        assert r.duration_ms is not None

    def test_mark_failed_without_created_at_skips_duration(self):
        r = self._make()
        r.created_at = None
        r.mark_failed("boom")
        assert r.duration_ms is None

    def test_repr(self):
        r = self._make()
        r.id = 7
        assert "InferenceRequest(id=7" in repr(r)
        assert "session_id='s1'" in repr(r)


class TestSystemMetrics:
    def test_memory_usage_percent_with_total(self):
        m = SystemMetrics(memory_used_gb=4, memory_total_gb=8)
        assert m.memory_usage_percent == 50.0

    def test_memory_usage_percent_zero_total(self):
        m = SystemMetrics(memory_used_gb=4, memory_total_gb=0)
        assert m.memory_usage_percent == 0.0

    def test_repr(self):
        m = SystemMetrics(
            memory_used_gb=4, memory_total_gb=8, timestamp=datetime.utcnow()
        )
        assert "SystemMetrics(timestamp=" in repr(m)
        assert "memory_used=4GB" in repr(m)


class TestAppSettings:
    def test_get_typed_value_string(self):
        s = AppSettings(key="k", value="hello", value_type="string")
        assert s.get_typed_value() == "hello"

    def test_get_typed_value_integer(self):
        s = AppSettings(key="k", value="42", value_type="integer")
        assert s.get_typed_value() == 42

    def test_get_typed_value_boolean_true_variants(self):
        for v in ("true", "1", "yes", "on", "TRUE", "Yes"):
            s = AppSettings(key="k", value=v, value_type="boolean")
            assert s.get_typed_value() is True, f"failed for {v}"

    def test_get_typed_value_boolean_false_variants(self):
        for v in ("false", "0", "no", "off", "anything"):
            s = AppSettings(key="k", value=v, value_type="boolean")
            assert s.get_typed_value() is False, f"failed for {v}"

    def test_get_typed_value_json(self):
        s = AppSettings(key="k", value=json.dumps({"a": 1}), value_type="json")
        assert s.get_typed_value() == {"a": 1}

    def test_set_typed_value_bool(self):
        s = AppSettings(key="k")
        s.set_typed_value(True)
        assert s.value == "true"
        assert s.value_type == "boolean"
        s.set_typed_value(False)
        assert s.value == "false"

    def test_set_typed_value_int(self):
        s = AppSettings(key="k")
        s.set_typed_value(42)
        assert s.value == "42"
        assert s.value_type == "integer"

    def test_set_typed_value_dict(self):
        s = AppSettings(key="k")
        s.set_typed_value({"a": 1})
        assert s.value_type == "json"
        assert json.loads(s.value) == {"a": 1}

    def test_set_typed_value_list(self):
        s = AppSettings(key="k")
        s.set_typed_value([1, 2, 3])
        assert s.value_type == "json"
        assert json.loads(s.value) == [1, 2, 3]

    def test_set_typed_value_string_fallback(self):
        s = AppSettings(key="k")
        s.set_typed_value("hello")
        assert s.value == "hello"
        assert s.value_type == "string"

    def test_set_typed_value_float_falls_to_string(self):
        # float isn't bool/int/dict/list → string fallback
        s = AppSettings(key="k")
        s.set_typed_value(1.5)
        assert s.value == "1.5"
        assert s.value_type == "string"

    def test_repr(self):
        s = AppSettings(key="k", value="v", value_type="string")
        assert "AppSettings(key='k'" in repr(s)
        assert "value='v'" in repr(s)
        assert "type='string'" in repr(s)


class TestRequestQueue:
    def _make(self):
        return RequestQueue(session_id="s1", model_id=1, request_data="{}")

    def test_defaults(self):
        # QueueStatus.QUEUED.value sentinel; ORM Column default only applies on flush
        assert QueueStatus.QUEUED.value == "queued"
        q = self._make()
        # priority default=0 also None until flush; skip
        assert q.session_id == "s1"

    def test_get_request_data(self):
        q = self._make()
        q.request_data = json.dumps({"x": 1})
        assert q.get_request_data() == {"x": 1}

    def test_set_request_data(self):
        q = self._make()
        q.set_request_data({"x": 1})
        assert json.loads(q.request_data) == {"x": 1}

    def test_start_processing(self):
        q = self._make()
        q.start_processing()
        assert q.status == QueueStatus.PROCESSING.value
        assert q.started_at is not None

    def test_repr(self):
        q = self._make()
        q.id = 3
        assert "RequestQueue(id=3" in repr(q)
        assert "session_id='s1'" in repr(q)
