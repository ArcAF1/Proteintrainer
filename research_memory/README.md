# Research Memory

A minimal electronic lab notebook for storing and searching research notes.

## Quick Start

```bash
pip install -e .
researchmem new note "Initial note" "Testing memory"
researchmem search "Testing"
```

### Schema
```
entry(id PK, created_at, type, title, body_md, status,
      confidence, tags, links, revises_id)
embedding(entry_id FK -> entry.id, vector BLOB)
```

Markdown files are stored under `~/research_memory/entries/`.

MIT License.
