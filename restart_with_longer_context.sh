#!/bin/bash
# Restart script to apply longer context configuration

echo "🔄 Restarting Biomedical AI with longer context support..."
echo ""

# Kill any existing instances
echo "📍 Stopping any running instances..."
pkill -f "python run_app.py" 2>/dev/null || true
pkill -f "gradio" 2>/dev/null || true
sleep 2

echo "✅ Configuration updated:"
echo "   • Context window: 2048 tokens (was 512)"
echo "   • Max output: 600 tokens (was 150)"
echo "   • Can now generate ~2400 character responses"
echo ""

echo "🚀 Starting application with new configuration..."
echo ""

# Start the application
./start_optimized.command 