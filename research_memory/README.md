

# research_memory

A lightweight electronic lab notebook layer for offline RAG assistants.

```
Entry schema (entry table)
+-------------+------------------+
| column      | type             |
+=============+==================+
| id          | TEXT PK          |
| created_at  | DATETIME         |
| type        | TEXT             |
| title       | TEXT             |
| body_md     | TEXT             |
| status      | TEXT             |
| confidence  | REAL             |
| tags        | TEXT             |
| links       | JSON             |
| revises_id  | TEXT FK->entry   |
+-------------+------------------+
```


## Quick Start

```bash
pip install -e .


python -m research_memory.cli new note "My first note" "Body text"
python -m research_memory.cli search "first"
```

All data is stored under `~/research_memory/`.

This package is MIT licensed and works fully offline.


