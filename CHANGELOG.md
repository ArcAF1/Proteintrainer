# Changelog

## [2024-01-XX] - Security, Unification, and Progress Tracking Update

### 🔒 Security Improvements
- **Fixed Neo4j Authentication**: Removed hardcoded passwords
  - Added proper environment variable support via python-dotenv
  - Created comprehensive `env.example` template
  - Updated all Neo4j connections to use settings from config.py
  - Added warnings when .env file is missing
  - Changed default password from "test" to secure pattern

### 🎯 GUI Consolidation
- **Created Unified GUI**: Merged gui.py, gui_v2.py, and gui_v3.py into gui_unified.py
  - All features now accessible from a single interface
  - Organized into logical tabs: Home, Setup, Chat, Graph Explorer, Research Lab
  - Consistent user experience across all features
  - Single entry point via run_app.py
  - Maintains backward compatibility with all existing features

### 📊 Progress Tracking
- **Added Real-time Progress Indicators**: 
  - Implemented progress callbacks throughout data_ingestion.py
  - Added progress tracking to train_pipeline.py
  - Integrated Gradio Progress component in GUI
  - Shows detailed progress for:
    - Dataset downloads with percentage
    - Archive extraction
    - PubMed/arXiv fetching
    - Relation extraction
    - Index building
    - Graph population
  - Thread-safe progress updates
  - Auto-updating progress display

### 🛠️ Additional Improvements
- Created comprehensive migration guide (MIGRATION_GUIDE.md)
- Updated README with new features and security instructions
- Improved error handling and user feedback
- Added Neo4j connection testing in run_app.py
- Enhanced docker-compose.yml with better defaults

### 📝 Files Modified
- `src/config.py` - Added dotenv support and security warnings
- `src/graph_rag.py` - Updated to use secure settings
- `src/gui_unified.py` - New unified interface
- `src/data_ingestion.py` - Added progress callbacks
- `src/train_pipeline.py` - Integrated progress tracking
- `run_app.py` - Updated to use unified GUI
- `docker-compose.yml` - Better security defaults
- `env.example` - New comprehensive template
- `README.md` - Updated documentation
- `MIGRATION_GUIDE.md` - New migration guide

### 🚀 How to Upgrade
1. Pull latest changes
2. Copy `env.example` to `.env` and set secure password
3. Restart Docker containers: `docker-compose down && docker-compose up -d`
4. Run `python run_app.py` to launch unified GUI

### ⚠️ Breaking Changes
- Old GUI modules (gui.py, gui_v2.py, gui_v3.py) are deprecated
- Neo4j now requires proper authentication setup
- Import paths changed from `src.gui_v3` to `src.gui_unified` 