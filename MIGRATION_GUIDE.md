# Migration Guide: Moving to Unified GUI

This guide helps users migrate from the previous GUI versions (gui.py, gui_v2.py, gui_v3.py) to the new unified interface.

## 🔄 What's Changed

### 1. Single Entry Point
- **Old**: Multiple GUI files (gui.py, gui_v2.py, gui_v3.py)
- **New**: Single unified GUI (`gui_unified.py`)
- **Usage**: Run `python run_app.py` or `python src/gui_unified.py`

### 2. Security Improvements
- **Old**: Hardcoded Neo4j password ("test")
- **New**: Environment-based configuration
- **Action Required**: 
  ```bash
  cp env.example .env
  # Edit .env to set your secure password
  ```

### 3. Progress Tracking
- **Old**: No progress indicators during long operations
- **New**: Real-time progress bars for all operations
- **Benefit**: Know exactly what's happening during training

## 📋 Feature Mapping

| Old Feature | Old Location | New Location |
|------------|--------------|--------------|
| Basic Chat | gui.py | Chat tab |
| Graph Visualization | gui_v2.py | Graph Explorer tab |
| Hypothesis Engine | gui_v3.py | Research Lab tab |
| System Test | gui_v3.py | Setup tab → Test System |
| Training | All versions | Setup tab → Start Training |

## 🚀 Quick Migration Steps

1. **Update your environment**:
   ```bash
   git pull
   cp env.example .env
   # Edit .env with your Neo4j password
   ```

2. **Update Docker configuration**:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

3. **Launch the new GUI**:
   ```bash
   python run_app.py
   ```

## 🔧 Troubleshooting

### "Module 'gui_v3' not found"
The old GUI modules have been replaced. Update your scripts to use:
```python
from src.gui_unified import main
```

### "Neo4j authentication failed"
1. Make sure you've created `.env` from `env.example`
2. Set a secure password in `.env`
3. Update docker-compose: `docker-compose up -d`

### "Missing progress bars"
Clear your browser cache and reload the GUI at http://localhost:7860

## ✨ New Features Available

- **Unified Interface**: All features in one place
- **Progress Tracking**: Real-time updates during operations
- **Better Organization**: Tabbed interface for different workflows
- **Security**: Proper authentication with environment variables
- **Enhanced Chat**: All slash commands work consistently

## 📝 Notes

- Your existing data, indexes, and Neo4j database remain compatible
- Research memory is preserved
- All chat commands work as before
- Performance is unchanged or improved 