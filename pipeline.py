# Backward-compatible re-export — use `import pipecraft` for new code
from pipecraft import *  # noqa: F401,F403
from pipecraft import _POOLS  # noqa: F401 — used by tests
