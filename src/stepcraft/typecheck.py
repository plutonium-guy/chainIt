from __future__ import annotations

import os
from typing import Any, Callable, Optional

try:
    from beartype import BeartypeConf, beartype as _beartype_decorator
    from beartype.roar import BeartypeDecorHintPep585DeprecationWarning

    import warnings

    warnings.filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)
    _BEARTYPE_CONF = BeartypeConf(is_pep484_tower=True)
    _HAS_BEARTYPE = True
except ImportError:  # pragma: no cover - beartype is a declared dependency
    _beartype_decorator = None
    _BEARTYPE_CONF = None
    _HAS_BEARTYPE = False


def beartype_enabled() -> bool:
    """Return True when runtime type-checking is active for step functions."""
    if not _HAS_BEARTYPE:
        return False
    return os.environ.get("STEPCRAFT_NO_BEARTYPE", "").lower() not in (
        "1", "true", "yes",
    )


def apply_step_beartype(func: Callable[..., Any]) -> Callable[..., Any]:
    """Strictly type-check a user step function's inputs and return value."""
    if not beartype_enabled():
        return func
    return _beartype_decorator(conf=_BEARTYPE_CONF)(func)


def resolve_output_schema(
    func: Callable[..., Any],
    schema: Optional[type],
) -> Optional[type]:
    """Return the explicit ``schema=`` output check, if any.

    The function's return-type annotation is intentionally NOT used as a schema:
    beartype already validates the return value on every call — including each
    element of an auto-mapped / parallel run — at the correct granularity.
    Deriving a schema from the annotation duplicated that work and, because an
    auto-mapped step returns a *list*, wrongly failed the whole-list check.
    ``schema=`` remains for explicit, beartype-independent output validation.
    """
    return schema
