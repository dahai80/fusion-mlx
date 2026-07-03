import struct as _struct
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock


def _mock_module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _is_real_mlx_available():
    try:
        import mlx.core as _mx
        _mx.zeros((1,))
        return True
    except Exception:
        return False


_HAS_REAL_MLX = _is_real_mlx_available()

if _HAS_REAL_MLX:
    import mlx.core as _real_mx_core
    import mlx
else:
    _mlx = MagicMock()
    _mlx.__version__ = "0.31.2"
    _mlx.__spec__ = MagicMock()

    _MOCK_FLOAT16 = "mock_float16"
    _MOCK_FLOAT32 = "mock_float32"
    _MOCK_BFLOAT16 = "mock_bfloat16"
    _MOCK_UINT16 = "mock_uint16"

    class _MockMXArray:
        def __init__(self, data, dtype=None):
            if isinstance(data, (list, tuple)):
                self._flat = [float(x) for x in data]
                self._shape = [len(data)]
            else:
                self._flat = []
                self._shape = [1]
            self.dtype = _MOCK_FLOAT32
            self._dtype_name = "float32"
            if dtype is not None:
                self.dtype = dtype

        def __repr__(self):
            return f"<MockMXArray {self._shape} {self._dtype_name}>"

        def __bytes__(self):
            return b"".join(_struct.pack("<f", x) for x in self._flat)

        @property
        def shape(self):
            return tuple(self._shape)

        @property
        def nbytes(self):
            return len(self._flat) * _struct.calcsize("f")

        def view(self, dtype):
            return self

        def tolist(self):
            return list(self._flat)

    def _mock_array(data, dtype=None):
        return _MockMXArray(data, dtype)

    def _mock_eval(*args):
        pass

    _mlx_core = MagicMock()
    _mlx_core.array = _mock_array
    _mlx_core.eval = _mock_eval
    _mlx_core.float16 = _MOCK_FLOAT16
    _mlx_core.float32 = _MOCK_FLOAT32
    _mlx_core.bfloat16 = _MOCK_BFLOAT16
    _mlx_core.int8 = "mock_int8"
    _mlx_core.int16 = "mock_int16"
    _mlx_core.int32 = "mock_int32"
    _mlx_core.int64 = "mock_int64"
    _mlx_core.uint8 = "mock_uint8"
    _mlx_core.uint16 = _MOCK_UINT16
    _mlx_core.uint32 = "mock_uint32"
    _mlx_core.uint64 = "mock_uint64"
    _mlx_core.zeros = MagicMock(return_value=_MockMXArray([]))

    sys.modules["mlx"] = _mlx
    sys.modules["mlx.nn"] = MagicMock()
    sys.modules["mlx.optimizers"] = MagicMock()
    sys.modules["mlx.core"] = _mlx_core
    sys.modules["mlx.random"] = MagicMock()
    sys.modules["mlx.fast"] = MagicMock()
    sys.modules["mlx_dtypes"] = MagicMock()

# MLX-LM: if real package is available, preserve it; otherwise mock.
# Many tests (cache, scheduler, batch_generator) depend on real
# KVCache/ArraysCache objects, so mocking breaks them.
try:
    import mlx_lm as _real_mlx_lm
    sys.modules["mlx_lm"] = _real_mlx_lm
    for _sub in ("generate", "models", "models.cache", "tokenizer_utils",
                 "load", "sample_utils", "utils"):
        _full = f"mlx_lm.{_sub}"
        try:
            __import__(_full)
        except ImportError:
            sys.modules[_full] = MagicMock()
except ImportError:
    _mlx_lm = MagicMock()
    _mlx_lm.load = MagicMock
    _mlx_lm.generate = MagicMock
    _mlx_lm.stream_generate = MagicMock
    sys.modules["mlx_lm"] = _mlx_lm
    sys.modules["mlx_lm.generate"] = MagicMock()
    sys.modules["mlx_lm.models"] = MagicMock()
    sys.modules["mlx_lm.models.cache"] = MagicMock()
    sys.modules["mlx_lm.tokenizer_utils"] = MagicMock()
    sys.modules["mlx_lm.load"] = MagicMock()
    sys.modules["mlx_lm.sample_utils"] = MagicMock()
    sys.modules["mlx_lm.utils"] = MagicMock()

# MLX-VLM: if real package is available, preserve it; otherwise mock.
try:
    import mlx_vlm as _real_mlx_vlm
    sys.modules["mlx_vlm"] = _real_mlx_vlm
    for _sub in ("generate", "models", "utils", "turboquant", "vision_cache"):
        _full = f"mlx_vlm.{_sub}"
        try:
            __import__(_full)
        except ImportError:
            sys.modules[_full] = MagicMock()
except ImportError:
    sys.modules["mlx_vlm"] = MagicMock()
    sys.modules["mlx_vlm.generate"] = MagicMock()
    sys.modules["mlx_vlm.models"] = MagicMock()
    sys.modules["mlx_vlm.utils"] = MagicMock()

# Other MLX ecosystem mocks
sys.modules["mlx_embeddings"] = MagicMock()
sys.modules["mlx_audio"] = MagicMock()
sys.modules["dflash_mlx"] = MagicMock()

# Mock heavy/optional dependencies
_mock_module("transformers")
_hf_hub = MagicMock()
_hf_hub.__spec__ = MagicMock()
_hf_hub.HfApi = MagicMock
_hf_hub.list_models = MagicMock
_hf_hub.model_info = MagicMock
_hf_hub.hf_hub_download = MagicMock
_hf_hub.snapshot_download = MagicMock
sys.modules["huggingface_hub"] = _hf_hub
_hf_hub_utils = MagicMock()
_hf_hub_utils.RepositoryNotFoundError = Exception
_hf_hub_utils.RevisionNotFoundError = Exception
sys.modules["huggingface_hub.utils"] = _hf_hub_utils
_mock_module("tokenizers")
_mock_module("mistral_common")
_mock_module("mistral_common.tokens")
_mock_module("mistral_common.tokens.tokenizers")
_mock_module("sentencepiece")
_mock_module("tiktoken")
_mock_module("socksio")
_mock_module("openai_harmony")

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")


@pytest.fixture(autouse=True)
def mock_mlx():
    pass


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
    tok.decode = MagicMock(return_value="test output")
    tok.apply_chat_template = MagicMock(return_value="<s>test prompt</s>")
    return tok


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.__call__ = MagicMock(return_value=MagicMock())
    return model
