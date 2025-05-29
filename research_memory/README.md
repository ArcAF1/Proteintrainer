# research_memory

Electronic lab notebook backend for RAG assistants.

## Quick Start

```bash
pip install -e .
researchmem new note "My first note"
```

## Schema

```
entry(id PK, created_at, type, title, body_md, status, confidence,
      tags, links, revises_id)
embedding(entry_id FK, vector BLOB)
```
