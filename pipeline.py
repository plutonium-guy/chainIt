# Backward-compatible re-export — use `import stepcraft` for new code
from stepcraft import *  # noqa: F401,F403
from stepcraft import _POOLS  # noqa: F401 — used by tests
