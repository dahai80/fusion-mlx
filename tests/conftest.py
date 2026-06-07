import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

def _mock_module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

# Mock MLX ecosystem (Apple Silicon only - always mock in CI)
# Use MagicMock for attribute auto-creation
_mlx = MagicMock()
_mlx.__version__ = "0.31.2"
# Set __spec__ to prevent transformers find_spec() from crashing
_mlx.__spec__ = MagicMock()
sys.modules["mlx"] = _mlx
sys.modules["mlx.nn"] = MagicMock()
sys.modules["mlx.optimizers"] = MagicMock()
sys.modules["mlx.core"] = MagicMock()
sys.modules["mlx.random"] = MagicMock()
sys.modules["mlx.fast"] = MagicMock()
sys.modules["mlx_dtypes"] = MagicMock()

# MLX-LM mocks
sys.modules["mlx_lm"] = MagicMock()
sys.modules["mlx_lm.generate"] = MagicMock()
sys.modules["mlx_lm.models"] = MagicMock()
sys.modules["mlx_lm.models.cache"] = MagicMock()
sys.modules["mlx_lm.tokenizer_utils"] = MagicMock()
sys.modules["mlx_lm.load"] = MagicMock()
sys.modules["mlx_lm.sample_utils"] = MagicMock()

# MLX-VLM mocks
sys.modules["mlx_vlm"] = MagicMock()
sys.modules["mlx_vlm.generate"] = MagicMock()
sys.modules["mlx_vlm.models"] = MagicMock()

# Other MLX ecosystem mocks
sys.modules["mlx_embeddings"] = MagicMock()
sys.modules["mlx_audio"] = MagicMock()
sys.modules["dflash_mlx"] = MagicMock()

# Mock heavy/optional dependencies to avoid import issues
_mock_module("transformers")
_mock_module("huggingface_hub")
_mock_module("tokenizers")
_mock_module("mistral_common")
_mock_module("mistral_common.tokens")
_mock_module("mistral_common.tokens.tokenizers")
_mock_module("sentencepiece")
_mock_module("tiktoken")
_mock_module("socksio")
_mock_module("openai_harmony")

# Now import pytest after mocks are in place
import pytest

# Ensure fusion_mlx is importable
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
