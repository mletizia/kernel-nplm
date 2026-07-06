# Local Data

Use this directory for datasets that should stay on this machine and out of Git.
The repository tracks this README and the local ignore rules, but ignores all
actual files placed here.

Suggested layout:

```text
local-data/
├── dataset-a/
│   └── ...
└── dataset-b/
    └── ...
```

In notebooks and scripts, prefer making the input location configurable:

```python
from pathlib import Path

DATA_DIR = Path("local-data") / "dataset-a"
```
