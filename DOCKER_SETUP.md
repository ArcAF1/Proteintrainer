# Docker & Neo4j Setup Guide

## Quick Start

1. **Create your .env file** (IMPORTANT for security):
   ```bash
   cp env.template .env
   ```
   Then edit `.env` and replace `your-secure-password-here` with an actual password.

2. **Run the start script**:
   ```bash
   ./start.command
   ```

That's it! The script will:
- ✅ Start Docker if needed
- ✅ Create Neo4j container with M1 optimizations
- ✅ Wait for Neo4j to be healthy
- ✅ Deploy the schema
- ✅ Launch the web interface

## What We Improved

### 🚀 **Performance Optimizations**
- Platform specification for ARM64/M1
- Optimized memory settings (2GB heap, 1GB page cache)
- Health checks to ensure Neo4j is ready

### 🔒 **Security**
- Passwords now in `.env` file (not in docker-compose.yml)
- `.env` is gitignored for security

### 🛡️ **Reliability**
- Connection retry logic (5 attempts with 5s delays)
- Docker health checks
- Better error messages

### 📦 **Structure**
- Organized volumes for data, logs, and imports
- Named network for future service expansion

## Troubleshooting

### Neo4j won't start
```bash
# Check logs
docker compose logs neo4j

# Check if port is in use
lsof -i :7687
```

### Out of memory errors
Edit `docker-compose.yml` and reduce:
```yaml
NEO4J_server_memory_heap_max__size: 1G  # Reduce from 2G
NEO4J_server_memory_pagecache_size: 512m  # Reduce from 1G
```

### Reset everything
```bash
# Stop and remove container
docker compose down

# Remove all data (CAREFUL!)
rm -rf data/neo4j

# Start fresh
./start.command
```

## Accessing Neo4j

- **Browser UI**: http://localhost:7474
- **Username**: neo4j
- **Password**: Whatever you set in `.env`

## Memory Guidelines for M1 Macs

- **8GB M1**: Use 1G heap, 512m page cache
- **16GB M1**: Use 2G heap, 1G page cache (default)
- **32GB+ M1**: Use 4G heap, 2G page cache 