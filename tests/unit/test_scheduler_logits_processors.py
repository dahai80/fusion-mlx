import pytest

pytestmark = pytest.mark.skip(
    reason="fusion-mlx scheduler does not expose GenerationBatch or logits processor helpers used by omlx tests"
)


def test_placeholder():
    pass
