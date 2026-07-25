# SPDX-License-Identifier: Apache-2.0
"""Minimal ``torch`` stub for the DMG bundle.

xgrammar 0.2.0 declares ``torch>=1.10.0`` as a runtime dep, but fusion-mlx never
exercises its torch-backed code paths: bitmasks are allocated as numpy
``int32`` buffers, the C++ binding fills them, and the MLX kernel applies the
mask. The torch dep is load-bearing only at *import time* — module-level code
in ``xgrammar.matcher``, ``xgrammar.testing``, ``xgrammar.contrib.hf`` and
``tvm_ffi.core`` does ``import torch`` plus a handful of attribute lookups.

Real torch is ~500 MB unpacked on macOS arm64 — too heavy to ship in the DMG.
This stub provides just enough of the torch surface for those modules to
finish loading. Code paths that would actually call into torch raise
``RuntimeError`` from the helpers below; fusion-mlx never reaches them.

When a real torch is installed (pip / Homebrew flow) the stub is a no-op:
``install()`` checks ``importlib.util.find_spec('torch')`` first.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import os
import sys
import threading
import types

logger = logging.getLogger(__name__)

# xgrammar / tvm-ffi versions this stub is known to cover.
# This module is the *single source of truth* — packaging/build.py imports
# these constants to keep the DMG install pin in sync with the stub. Update
# both tuples here when bumping; the build script auto-tracks.
#
# Reachable-but-stubbed torch surface to be aware of when upgrading:
#   - ``torch.full``: ``xgrammar.allocate_token_bitmask`` calls it. fusion-mlx
#     never invokes ``allocate_token_bitmask`` (we use the MLX kernel
#     path), but the symbol is re-exported from ``xgrammar.__init__``.
#     Any future caller that touches it will hit ``_unsupported("full")``
#     and surface a clear RuntimeError.
#   - ``torch.tensor`` returns a ``_StubTensor`` whose attribute access
#     raises a stub-identifying RuntimeError. Module-level
#     ``_FULL_MASK = torch.tensor(-1, ...)`` patterns succeed at import
#     time; any subsequent method call (.fill_, .item, ...) fails.
_TARGET_XGRAMMAR_VERSIONS = ("0.2.0",)
_TARGET_TVM_FFI_VERSIONS = ("0.1.11",)

# Serialize install() across threads. Without this, two threads that both
# pass the "torch" in sys.modules check race to build modules and overwrite
# each other's sys.modules['torch'] entry, leaving threads that already
# dereferenced the loser's module with stale references. Reachable today
# from concurrent HTTP handlers that call install() on first xgrammar use.
_INSTALL_LOCK = threading.Lock()
_INSTALLED = False


class _LazyMockModule(types.ModuleType):
    """Module that auto-creates _LazyMockObject for any attribute access.

    Prevents AttributeError when upstream libs probe attributes we never
    explicitly stubbed. Falls back silently instead of crashing at import time.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._name = name

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        logger.debug("Torch stub intercepted: torch.%s.%s", self._name, name)
        return _LazyMockObject(f"{self._name}.{name}")


class _LazyMockObject:
    """Auto-delegating mock that survives any attribute chain.

    ``x.y.z.w()`` returns another _LazyMockObject so import-time probes
    don't crash. Actual method calls return a mock -- if a code path
    depends on the result, it will fail at the usage site.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, *args, **kwargs) -> _LazyMockObject:
        return _LazyMockObject(f"{self._name}()")

    def __getattr__(self, name: str) -> _LazyMockObject:
        return _LazyMockObject(f"{self._name}.{name}")

    def __mro_entries__(self, bases) -> tuple:
        return (object,)

    def __repr__(self) -> str:
        return f"<_LazyMockObject {self._name}>"


class _StubTensor:
    """Placeholder for ``torch.Tensor`` (annotations + isinstance checks).

    Any attribute access raises a clear RuntimeError so runtime use of a
    stubbed tensor (e.g. ``some_tensor.fill_(...)``) fails loudly with a
    pointer to the cause, rather than at the AttributeError level with a
    generic ``has no attribute 'fill_'`` message.
    """

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        return _StubTensor()

    def __getattr__(self, name: str):
        # Let dunder probes (pickle, copy.deepcopy, descriptor lookups,
        # `hasattr` chains in third-party libs) fall through cleanly as
        # AttributeError — that's the documented `__getattr__` contract.
        # Real torch tensors lack many of these probed dunders anyway, so
        # raising AttributeError is the correct, distinguishable signal.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise RuntimeError(
            f"_StubTensor.{name} is not implemented: fusion-mlx ships a torch "
            "stub for xgrammar's import-time needs only. Reaching a real "
            "tensor method means a code path that needs real torch was "
            "exercised — install torch via pip/Homebrew or report this as "
            "a bug if the call originated inside fusion-mlx."
        )


class _StubDtype:
    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"torch.{self._name}"

    # Some xgrammar/tvm-ffi paths convert dtype to string via ``str(dt)``
    # rather than ``repr(dt)`` (e.g. ``to_cpp_dtype`` strips the "torch."
    # prefix). Match real torch's behaviour where ``str(torch.int32)`` is
    # ``"torch.int32"`` so those paths keep working.
    def __str__(self) -> str:
        return f"torch.{self._name}"


def _stub_tensor_factory(*args, **kwargs) -> _StubTensor:
    """torch.tensor(...) stub: returns a _StubTensor instance.

    Returning a real object (rather than None) means module-globals like
    xgrammar.matcher._FULL_MASK = torch.tensor(-1, dtype=...) succeed at
    import time. Any subsequent method call on the result (.fill_, .item,
    etc.) raises with a clear pointer via _StubTensor.__getattr__.
    """
    return _StubTensor()


def _false(*args, **kwargs) -> bool:
    return False


def _unsupported(qualname: str):
    def _fn(*args, **kwargs):
        raise RuntimeError(
            f"torch.{qualname} is not available: this fusion-mlx build ships a "
            "torch stub for xgrammar's import-time needs only. Install "
            "real torch via pip/Homebrew if you need this code path."
        )

    return _fn


def _stub_tensor_fn(qualname: str):
    """Return a function that returns a _StubTensor — for ops used at import/class-def time."""
    def _fn(*args, **kwargs):
        return _StubTensor()
    return _fn


# (canonical, alias) pairs — real torch aliases torch.int to torch.int32,
# torch.long to torch.int64, etc.; preserve those identities so code that
# does ``torch.int is torch.int32`` keeps working.
_DTYPE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("int32", ("int",)),
    ("int16", ("short",)),
    ("int64", ("long",)),
    ("float16", ("half",)),
    ("float32", ("float",)),
    ("float64", ("double",)),
    ("int8", ()),
    ("uint8", ()),
    ("uint16", ()),
    ("uint32", ()),
    ("uint64", ()),
    ("bfloat16", ()),
    ("bool", ()),
    ("float8_e4m3fn", ()),
    ("float8_e4m3fnuz", ()),
    ("float8_e5m2", ()),
    ("float8_e5m2fnuz", ()),
    ("float8_e8m0fnu", ()),
    ("float4_e2m1fn_x2", ()),
    ("complex64", ()),
    ("complex128", ()),
)

_TENSOR_ALIASES = (
    "Tensor",
    "LongTensor",
    "FloatTensor",
    "IntTensor",
    "ByteTensor",
    "DoubleTensor",
    "HalfTensor",
    "BoolTensor",
    "ShortTensor",
)


# Names that xgrammar / tvm_ffi probe via getattr(torch, name) for
# feature-detection — they catch AttributeError and fall back gracefully.
# Logging WARNING for these floods the log on every model load (one per
# name per process) with diagnostics that aren't actually actionable.
# Demote known-probed names to DEBUG; everything else stays WARNING so
# genuinely-missing attributes surface in operator logs.
_KNOWN_PROBE_NAMES: frozenset[str] = frozenset(
    {
        # Integer dtypes added post-torch-2.0 that tvm_ffi.dtypes enumerates
        "uint16",
        "uint32",
        "uint64",
        # FP8 / FP4 dtypes (probed by tvm_ffi.dtypes' dtype-mapping table)
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
        "float4_e2m1fn_x2",
    }
)


def _make_top_level_torch_getattr() -> callable:
    """Return a ``__getattr__`` for the stub's top-level torch module.

    Real-torch users who reach an unset attribute would get an
    ``AttributeError``; consumers that probe with ``hasattr`` rely on that.
    But we *also* want a clearly-identifiable message when downstream
    libraries (transformers, accelerate, etc.) reach for a torch surface
    we never stubbed — so this raises ``AttributeError`` whose message
    pinpoints the fusion-mlx stub. ``pkgutil.iter_modules(torch.__path__)`` and
    similar discovery paths see the empty ``__path__`` and short-circuit
    before hitting this.
    """

    _missing_attr_logged: set[str] = set()
    _torch_ref = [None]  # mutable holder, set after torch module is built

    def __getattr__(name: str):  # noqa: N807
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"torch.{name}")
        if name not in _missing_attr_logged:
            _missing_attr_logged.add(name)
            level = logging.DEBUG if name in _KNOWN_PROBE_NAMES else logging.WARNING
            logger.log(
                level,
                "fusion-mlx torch stub auto-stubbing: torch.%s "
                "(returning no-op stub)",
                name,
            )
        # Return a no-op that can be called, used as base class, or accessed as module
        # Uses a metaclass so class-level attribute access (torch.autograd.Function) works
        class _AutoStubMeta(type):
            def __getattr__(meta_cls, attr):
                return _auto_sub(f"{meta_cls.__name__}.{attr}")
            def __repr__(meta_cls):
                return f"<metastub:{meta_cls.__name__}>"

        def _auto_sub(attr_name):
            class _AutoStub(metaclass=_AutoStubMeta):
                def __init_subclass__(cls, **kw): pass
                def __init__(self, *a, **kw): pass
                def __call__(self, *a, **kw):
                    return _auto_sub(f"{attr_name}.__call__")
                def __getattr__(self, attr):
                    return _auto_sub(f"{attr_name}.{attr}")
                def __mro_entries__(self, bases):
                    return (object,)
                def __repr__(self):
                    return f"<torch.{attr_name} stub>"
            _AutoStub.__name__ = attr_name
            _AutoStub.__qualname__ = attr_name
            return _AutoStub
        stub = _auto_sub(name)
        # Cache it so the same name always returns the same stub
        if _torch_ref[0] is not None:
            setattr(_torch_ref[0], name, stub)
        return stub

    return __getattr__, _torch_ref


def _build_modules() -> dict[str, types.ModuleType]:
    torch = types.ModuleType("torch")
    for alias in _TENSOR_ALIASES:
        setattr(torch, alias, _StubTensor)
    torch.dtype = _StubDtype
    torch.__version__ = "0.0.0+fusion-mlx-stub"
    # Pin the stub as the source of truth for the xgrammar version it
    # targets; packaging/build.py imports this constant to stay in sync.
    # (Module-level constant lives at the top of this file.)
    for canonical, aliases in _DTYPE_ALIASES:
        dt = _StubDtype(canonical)
        setattr(torch, canonical, dt)
        for a in aliases:
            setattr(torch, a, dt)
    torch.tensor = _stub_tensor_factory
    torch.full = _stub_tensor_fn("full")
    torch.zeros = _stub_tensor_fn("zeros")
    torch.ones = _stub_tensor_fn("ones")
    torch.empty = _stub_tensor_fn("empty")
    torch.empty_like = _stub_tensor_fn("empty_like")
    torch.arange = _stub_tensor_fn("arange")
    torch.linspace = _stub_tensor_fn("linspace")
    torch.randn = _stub_tensor_fn("randn")
    torch.rand = _stub_tensor_fn("rand")
    torch.cat = _stub_tensor_fn("cat")
    torch.stack = _stub_tensor_fn("stack")
    torch.clamp = _stub_tensor_fn("clamp")
    torch.sin = _stub_tensor_fn("sin")
    torch.cos = _stub_tensor_fn("cos")
    torch.mm = _stub_tensor_fn("mm")
    torch.norm = _stub_tensor_fn("norm")
    torch.meshgrid = _stub_tensor_fn("meshgrid")
    torch.from_numpy = _stub_tensor_fn("from_numpy")
    torch.frombuffer = _stub_tensor_fn("frombuffer")
    torch.addcmul = _stub_tensor_fn("addcmul")
    torch.from_dlpack = _unsupported("from_dlpack")

    # No-op context managers / decorators — MLX doesn't use autograd
    class _NoOpContextManager:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __call__(self, fn=None):
            if fn is not None: return fn
            return self
    torch.inference_mode = _NoOpContextManager()
    torch.no_grad = _NoOpContextManager()
    torch.enable_grad = _NoOpContextManager()

    class _Stream:
        pass

    cuda = _LazyMockModule("cuda")
    cuda.is_available = _false
    cuda.current_device = lambda: 0
    cuda.device_count = lambda: 1
    cuda.mem_get_info = lambda device=None: (16 * 1024 * 1024 * 1024, 64 * 1024 * 1024 * 1024)
    cuda.set_device = lambda device: None
    cuda.synchronize = lambda device=None: None
    cuda.empty_cache = lambda: None
    cuda.memory_stats = lambda device=None: {
        "active_bytes.all.current": 0,
        "reserved_bytes.all.current": 0,
        "allocation.all.current": 0,
        "active.all.current": 0,
        "reserved.all.current": 0,
    }
    cuda.get_device_name = lambda device=None: "Apple Metal (MLX)"
    cuda.get_device_properties = lambda device=None: type("props", (), {"gcnArchName": "", "major": 0, "minor": 0, "total_memory": 64 * 1024 * 1024 * 1024})()
    cuda.get_allocator_backend = lambda: "MLX"
    cuda.current_stream = lambda device=None: _Stream()
    cuda.Stream = _Stream
    cuda.stream = _NoOpContextManager()
    cuda.OutOfMemoryError = MemoryError
    cuda.cudart = lambda: type("cudart", (), {"cudaHostRegister": lambda *a, **kw: 0, "cudaHostUnregister": lambda *a, **kw: 0})()

    torch.cuda = cuda

    # Stub device class — represents "mlu:0" or "cpu"
    class _StubDevice:
        def __init__(self, type="mlu", index=0):
            self.type = type
            self.index = index
        def __repr__(self):
            if self.index is not None:
                return f"{self.type}:{self.index}"
            return self.type
        def __hash__(self):
            return hash((self.type, self.index))
        def __eq__(self, other):
            return isinstance(other, _StubDevice) and self.type == other.type and self.index == other.index
    torch.device = _StubDevice

    # Stub xpu / backends / corex — probe-only, return False/None
    xpu = _LazyMockModule("xpu")
    xpu.is_available = _false
    torch.xpu = xpu

    backends = _LazyMockModule("backends")
    backends_cudnn = _LazyMockModule("backends.cudnn")
    backends_cudnn.is_available = _false
    backends_cudnn.version = lambda: None
    backends_mps = _LazyMockModule("backends.mps")
    backends_mps.is_available = _false
    backends_mps.is_built = _false
    backends_cuda = _LazyMockModule("backends.cuda")
    backends_cuda.enable_math_sdp = lambda *a: None
    backends_cuda.enable_flash_sdp = lambda *a: None
    backends_cuda.enable_mem_efficient_sdp = lambda *a: None
    backends_cuda.allow_fp16_bf16_reduction_math_sdp = lambda *a: None
    backends_cuda.matmul = type("matmul", (), {"allow_fp16_accumulation": False})()
    backends.cudnn = backends_cudnn
    backends.mps = backends_mps
    backends.cuda = backends_cuda
    torch.backends = backends
    torch.corex = _LazyMockModule("corex")

    version = _LazyMockModule("version")
    version.cuda = None
    version.hip = None
    torch.version = version

    nn_functional = _LazyMockModule("nn.functional")
    nn_functional.pad = _unsupported("nn.functional.pad")
    nn_functional.linear = _unsupported("nn.functional.linear")
    nn_functional.conv1d = _unsupported("nn.functional.conv1d")
    nn_functional.conv2d = _unsupported("nn.functional.conv2d")
    nn_functional.conv3d = _unsupported("nn.functional.conv3d")
    nn_functional.group_norm = _unsupported("nn.functional.group_norm")
    nn_functional.batch_norm = _unsupported("nn.functional.batch_norm")
    nn_functional.layer_norm = _unsupported("nn.functional.layer_norm")
    nn_functional.rms_norm = _unsupported("nn.functional.rms_norm")
    nn_functional.scaled_dot_product_attention = _unsupported("nn.functional.scaled_dot_product_attention")
    nn_functional.silu = _unsupported("nn.functional.silu")
    nn_functional.gelu = _unsupported("nn.functional.gelu")
    nn_functional.relu = _unsupported("nn.functional.relu")
    nn_functional.interpolate = _unsupported("nn.functional.interpolate")
    nn_functional.embedding = _unsupported("nn.functional.embedding")

    # Proper base classes that can be subclassed
    class _StubModule:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return _StubTensor()
        def forward(self, *a, **kw): return _StubTensor()
        def parameters(self): return []
        def named_parameters(self): return []
        def to(self, *a, **kw): return self
        def cuda(self, *a, **kw): return self
        def cpu(self): return self
        def float(self): return self
        def half(self): return self
        def train(self, mode=True): return self
        def eval(self): return self
        def __repr__(self): return f"{self.__class__.__name__}()"

    class _StubParameter(_StubTensor):
        def __new__(cls, data=None, *a, **kw): return object.__new__(cls)
        def __init__(self, data=None, requires_grad=False): pass

    class _StubLinear(_StubModule):
        def __init__(self, in_features=0, out_features=0, bias=True, **kw): pass

    class _StubConv1d(_StubModule):
        def __init__(self, in_channels=0, out_channels=0, kernel_size=1, **kw): pass

    class _StubConv2d(_StubModule):
        def __init__(self, in_channels=0, out_channels=0, kernel_size=1, **kw): pass

    class _StubConv3d(_StubModule):
        def __init__(self, in_channels=0, out_channels=0, kernel_size=1, **kw): pass

    class _StubGroupNorm(_StubModule):
        def __init__(self, num_groups=1, num_channels=1, **kw): pass

    class _StubBatchNorm2d(_StubModule):
        def __init__(self, num_features=0, **kw): pass

    class _StubLayerNorm(_StubModule):
        def __init__(self, normalized_shape=None, **kw): pass

    class _StubRMSNorm(_StubModule):
        def __init__(self, normalized_shape=None, **kw): pass

    class _StubEmbedding(_StubModule):
        def __init__(self, num_embeddings=0, embedding_dim=0, **kw): pass

    class _StubConvTranspose1d(_StubModule):
        def __init__(self, in_channels=0, out_channels=0, kernel_size=1, **kw): pass

    class _StubConvTranspose2d(_StubModule):
        def __init__(self, in_channels=0, out_channels=0, kernel_size=1, **kw): pass

    class _StubSequential(_StubModule):
        def __init__(self, *args, **kw): pass

    class _StubModuleList(_StubModule):
        def __init__(self, modules=None, **kw): pass

    nn = _LazyMockModule("nn")
    nn.Module = _StubModule
    nn.Parameter = _StubParameter
    nn.Linear = _StubLinear
    nn.Conv1d = _StubConv1d
    nn.Conv2d = _StubConv2d
    nn.Conv3d = _StubConv3d
    nn.GroupNorm = _StubGroupNorm
    nn.BatchNorm2d = _StubBatchNorm2d
    nn.LayerNorm = _StubLayerNorm
    nn.RMSNorm = _StubRMSNorm
    nn.Embedding = _StubEmbedding
    nn.ConvTranspose1d = _StubConvTranspose1d
    nn.ConvTranspose2d = _StubConvTranspose2d
    nn.Sequential = _StubSequential
    nn.ModuleList = _StubModuleList
    nn.functional = nn_functional

    # torch.nn.init — weight initialization functions
    nn_init = _LazyMockModule("nn.init")
    nn_init.normal_ = lambda *a, **kw: None
    nn_init.zeros_ = lambda *a, **kw: None
    nn_init.ones_ = lambda *a, **kw: None
    nn_init.xavier_uniform_ = lambda *a, **kw: None
    nn_init.xavier_normal_ = lambda *a, **kw: None
    nn_init.kaiming_uniform_ = lambda *a, **kw: None
    nn_init.kaiming_normal_ = lambda *a, **kw: None
    nn_init.constant_ = lambda *a, **kw: None
    nn.init = nn_init

    # torch.nn.utils.parametrize
    nn_utils_parametrize = _LazyMockModule("nn.utils.parametrize")
    nn_utils_parametrize.remove_parametrizations = lambda *a, **kw: None
    nn_utils_parametrize.register_parametrization = lambda *a, **kw: None
    nn_utils = _LazyMockModule("nn.utils")
    nn_utils.parametrize = nn_utils_parametrize
    nn_utils.weight_norm = lambda *a, **kw: a[0] if a else None
    nn_utils.remove_weight_norm = lambda *a, **kw: None
    nn_utils.spectral_norm = lambda *a, **kw: a[0] if a else None
    nn_utils.remove_spectral_norm = lambda *a, **kw: None
    nn_utils.clip_grad_norm_ = lambda *a, **kw: 0.0
    nn.utils = nn_utils

    # torch.nn.modules.utils
    nn_modules_utils = _LazyMockModule("nn.modules.utils")
    nn_modules_utils._triple = lambda x: (x, x, x) if isinstance(x, int) else x
    nn_modules_utils._pair = lambda x: (x, x) if isinstance(x, int) else x
    nn_modules = _LazyMockModule("nn.modules")
    nn_modules.utils = nn_modules_utils
    nn.modules = nn_modules

    torch.nn = nn

    # torch.nn.attention for SDPA
    nn_attention = _LazyMockModule("nn.attention")
    nn_attention.SDPBackend = type("SDPBackend", (), {})
    nn_attention.sdpa_kernel = _NoOpContextManager()
    torch.nn.attention = nn_attention

    # torch.autograd submodule — must be a real module for `from torch.autograd import Function`
    class _StubAutogradFunction:
        def __init_subclass__(cls, **kw): pass
        @staticmethod
        def apply(*a, **kw): return _StubTensor()
    autograd = _LazyMockModule("autograd")
    autograd.Function = _StubAutogradFunction
    torch.autograd = autograd

    # torch.nn.modules.module — spandrel does `from torch.nn.modules.module import Module`
    nn_modules_module = _LazyMockModule("nn.modules.module")
    nn_modules_module.Module = _StubModule
    nn_modules.module = nn_modules_module

    # torch.nn.modules.batchnorm — spandrel does `from torch.nn.modules.batchnorm import _BatchNorm`
    nn_modules_batchnorm = _LazyMockModule("nn.modules.batchnorm")
    nn_modules_batchnorm._BatchNorm = _StubBatchNorm2d
    nn_modules.batchnorm = nn_modules_batchnorm

    # torch.nn.modules.conv — spandrel does `from torch.nn.modules.conv import _ConvNd`
    nn_modules_conv = _LazyMockModule("nn.modules.conv")
    nn_modules_conv._ConvNd = _StubConv2d
    nn_modules.conv = nn_modules_conv

    # torch.distributions
    distributions = _LazyMockModule("distributions")
    distributions.Normal = type("Normal", (), {"__init__": lambda self, *a, **kw: None, "sample": lambda self, *a: _StubTensor(), "rsample": lambda self, *a: _StubTensor(), "log_prob": lambda self, *a: _StubTensor()})
    distributions.Independent = type("Independent", (), {"__init__": lambda self, *a, **kw: None})
    distributions.KL = _LazyMockModule("distributions.kl")
    torch.distributions = distributions

    # torch.linalg submodule
    linalg = _LazyMockModule("linalg")
    linalg.norm = _stub_tensor_fn
    linalg.inv = _stub_tensor_fn
    linalg.solve = _stub_tensor_fn
    linalg.eig = _stub_tensor_fn
    linalg.eigh = _stub_tensor_fn
    linalg.svd = _stub_tensor_fn
    linalg.det = _stub_tensor_fn
    linalg.matrix_exp = _stub_tensor_fn
    linalg.multi_dot = _stub_tensor_fn
    linalg.cross = _stub_tensor_fn
    torch.linalg = linalg

    # torch.optim submodule
    optim = _LazyMockModule("optim")
    torch.optim = optim

    # torch.compiler submodule
    compiler = _LazyMockModule("compiler")
    compiler.disable = lambda *a, **kw: (lambda f: f)
    compiler.enable = lambda *a, **kw: (lambda f: f)
    torch.compiler = compiler

    # torch.jit submodule
    jit = _LazyMockModule("jit")
    jit.script = lambda f: f
    jit.trace = lambda *a, **kw: (lambda f: f)
    jit.Final = type("Final", (), {"__class_getitem__": classmethod(lambda cls, item: item)})
    jit.export = lambda f: f
    torch.jit = jit

    # torch.library submodule
    _library = _LazyMockModule("library")
    torch.library = _library

    # torch.mps submodule
    mps = _LazyMockModule("mps")
    mps.is_available = _false
    mps.is_built = _false
    torch.mps = mps

    # torch.hub submodule
    hub = _LazyMockModule("hub")
    hub.load_state_dict_from_url = _stub_tensor_fn
    hub.set_dir = lambda *a, **kw: None
    hub.get_dir = lambda: "/tmp/torch_hub"
    hub.load = _stub_tensor_fn
    torch.hub = hub

    utils_dlpack = _LazyMockModule("utils.dlpack")
    utils_dlpack.to_dlpack = _unsupported("utils.dlpack.to_dlpack")
    utils_checkpoint = _LazyMockModule("utils.checkpoint")
    utils_checkpoint.checkpoint = lambda *a, **kw: a[0] if a else None
    utils = _LazyMockModule("utils")
    utils.dlpack = utils_dlpack
    utils.checkpoint = utils_checkpoint
    utils_data = _LazyMockModule("utils.data")
    utils_data.DataLoader = type("DataLoader", (), {"__init__": lambda self, *a, **kw: None, "__iter__": lambda self: iter([]), "__len__": lambda self: 0})
    utils_data.Dataset = type("Dataset", (), {"__init__": lambda self, *a, **kw: None})
    utils.data = utils_data
    torch.utils = utils

    # torch.fft
    fft = _LazyMockModule("fft")
    fft.fft = _unsupported("fft.fft")
    fft.rfft = _unsupported("fft.rfft")
    fft.irfft = _unsupported("fft.irfft")
    fft.fftfreq = _unsupported("fft.fftfreq")
    fft.rfftfreq = _unsupported("fft.rfftfreq")
    torch.fft = fft

    serialization = _LazyMockModule("serialization")
    serialization.add_safe_globals = lambda *a, **kw: None
    torch.serialization = serialization

    # Top-level __getattr__ — auto-stubs unknown attributes so ComfyUI's
    # deep torch integration doesn't crash on every missing symbol.
    getter, _torch_ref_holder = _make_top_level_torch_getattr()
    _torch_ref_holder[0] = torch
    torch.__getattr__ = getter

    return {
        "torch": torch,
        "torch.cuda": cuda,
        "torch.version": version,
        "torch.nn": nn,
        "torch.nn.functional": nn_functional,
        "torch.nn.attention": nn_attention,
        "torch.utils": utils,
        "torch.utils.dlpack": utils_dlpack,
        "torch.utils.checkpoint": utils_checkpoint,
        "torch.utils.data": utils_data,
        "torch.nn.init": nn_init,
        "torch.nn.utils": nn_utils,
        "torch.nn.utils.parametrize": nn_utils_parametrize,
        "torch.nn.modules": nn_modules,
        "torch.nn.modules.utils": nn_modules_utils,
        "torch.nn.modules.module": nn_modules_module,
        "torch.nn.modules.batchnorm": nn_modules_batchnorm,
        "torch.nn.modules.conv": nn_modules_conv,
        "torch.distributions": distributions,
        "torch.linalg": linalg,
        "torch.fft": fft,
        "torch.autograd": autograd,
        "torch.optim": optim,
        "torch.compiler": compiler,
        "torch.jit": jit,
        "torch.library": _library,
        "torch.mps": mps,
        "torch.hub": hub,
        "torch.serialization": serialization,
        "torch.backends": backends,
        "torch.backends.cudnn": backends_cudnn,
        "torch.backends.mps": backends_mps,
        "torch.backends.cuda": backends_cuda,
    }


def install() -> bool:
    """Install the stub into ``sys.modules`` if no real torch is available.

    Returns True if the stub was installed (or had been installed previously),
    False if a real torch was found and left alone.

    Thread-safe — concurrent callers (e.g. multiple FastAPI handlers hitting
    the xgrammar entry points in parallel) serialize on _INSTALL_LOCK.
    """
    global _INSTALLED
    needs_version_check = False
    with _INSTALL_LOCK:
        if _INSTALLED:
            return True

        if "torch" in sys.modules:
            already_stub = getattr(sys.modules["torch"], "__version__", "").endswith(
                "+fusion-mlx-stub"
            )
            _INSTALLED = already_stub
            return already_stub

        try:
            if importlib.util.find_spec("torch") is not None:
                # Real torch is on the path — leave it alone, install() is
                # a no-op. Don't mark _INSTALLED so a future sys.modules
                # reset (e.g. in tests) re-evaluates. Crucially, also DO
                # NOT touch ``TVM_FFI_DISABLE_TORCH_C_DLPACK`` — the user
                # has real torch and the tvm-ffi/torch-C-DLPack JIT path
                # may be their preferred fast path.
                return False
        except Exception:
            # find_spec can raise on broken parent packages, partial
            # installs, or weird import hooks. Treat as "no torch" — the
            # stub is the safe fallback.
            pass

        # No real torch — disable tvm_ffi's JIT torch-C-DLPack extension
        # before any tvm-ffi / xgrammar import. Without this,
        # tvm_ffi/_optional_torch_c_dlpack tries to JIT a C extension
        # against our stub at first import, spawns a doomed Python
        # subprocess that fails to ``import torch.utils.cpp_extension``
        # (the stub does not provide it), and surfaces a misleading
        # "Failed to JIT torch c dlpack extension" warning to users on
        # every cold start. The guard inside that module honours this
        # env var and skips the JIT path entirely.
        os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")

        for name, mod in _build_modules().items():
            # ``__spec__`` must be a real ModuleSpec (not None) so that
            # ``importlib.util.find_spec`` succeeds when called by
            # transformers and other consumers. ``__version__`` is a
            # clearly-fake value so transformers refuses to take the
            # torch-modeling path.
            mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            mod.__loader__ = None
            if "." not in name:
                mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = mod
        _INSTALLED = True
        needs_version_check = True

    # Fire the version-drift check OUTSIDE the install lock. xgrammar's
    # C++ extension load can be slow on a cold disk; running it under
    # the lock would block every concurrent install() caller behind one
    # cold import. install() is idempotent at this point — _INSTALLED is
    # set and any racing caller short-circuits at the top of the lock.
    if needs_version_check:
        try:
            warn_if_unexpected_versions()
        except Exception:  # pragma: no cover — defensive
            pass
    return True


def warn_if_unexpected_versions() -> None:
    """Log a warning when bundled xgrammar / tvm-ffi versions drift past the
    versions this stub was tested against. Best-effort: silent if the
    imports themselves haven't happened yet, since the stub is installed
    eagerly at startup.
    """
    try:
        import xgrammar  # type: ignore[import-not-found]

        v = getattr(xgrammar, "__version__", None)
        if v and v not in _TARGET_XGRAMMAR_VERSIONS:
            logger.warning(
                "xgrammar %s is not in the torch-stub target set %s; "
                "structured output may fail at runtime. Update the stub "
                "or pin xgrammar back.",
                v,
                _TARGET_XGRAMMAR_VERSIONS,
            )
    except Exception:
        pass
    try:
        import tvm_ffi  # type: ignore[import-not-found]

        v = getattr(tvm_ffi, "__version__", None)
        if v and v not in _TARGET_TVM_FFI_VERSIONS:
            logger.warning(
                "apache-tvm-ffi %s is not in the torch-stub target set %s; "
                "structured output may fail at runtime.",
                v,
                _TARGET_TVM_FFI_VERSIONS,
            )
    except Exception:
        pass
