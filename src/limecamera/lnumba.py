from typing import TYPE_CHECKING

IS_NUMBA = True
IS_CACHE = True

try:
    import numba as nb
except ImportError as exc:
    if not TYPE_CHECKING:
        class Empty:
            def __init__(self, exc: ImportError):
                self._exc = exc
            def __repr__(self):
                return (f'Module "{self.exc.name}" import was attempted, not found,'
                        f' and then later access was attempted. If you need the'
                        f' module, install it.\n'
                        f'{self._exc.__class__} {self._exc}')
            def __getattr__(self, _: str):
                return self.__class__(self._exc)
            # --------------------------------------------------------------
            __getitem__ = __getattr__
            # --------------------------------------------------------------
            def __call__(self, *_, **__):
                return self.__class__(self._exc)
        # ------------------------------------------------------------------
        class MockNumba:

            def __getattr__(self, _):
                return Empty(exc)
            # --------------------------------------------------------------
            def jit(self, function_or_signature = None, **_):
                if callable(function_or_signature):
                    return function_or_signature
                return self.jit
            # --------------------------------------------------------------
            njit = jit
            # --------------------------------------------------------------
            prange = staticmethod(range)
        nb = MockNumba()

if TYPE_CHECKING:
    from typing import TypeAlias
    nbType: TypeAlias = nb.core.types.abstract.Type
else:
    nbType = object
# ======================================================================
# Signatures
def nbA(dim: int = 1, dtype = nb.float64) -> nbType:
    return nb.types.Array(dtype, dim, 'C')
# ----------------------------------------------------------------------
def nbARO(dim: int = 1, dtype = nb.float64) -> nbType:
    return nb.types.Array(dtype, dim, 'C', readonly = True)
# ======================================================================
# Decorators

if IS_NUMBA:
    nbDec = nb.njit
else:
    def nbDec(f = None, **__):
        return f if callable(f) else nbDec

nbDecC = nbDec(cache = IS_CACHE)
nbDecFC = nbDec(fastmath = True, cache = IS_CACHE)
