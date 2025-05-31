#!/bin/bash

echo "🔍 Looking for Proteintrainer processes..."

# Find processes on common ports
for port in 7860 7861 7862 7863 7864 7865; do
    pids=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pids" ]; then
        echo "   Found process on port $port (PID: $pids)"
        kill -9 $pids 2>/dev/null && echo "   ✅ Killed process on port $port"
    fi
done

# Find Python processes running Proteintrainer
pids=$(ps aux | grep -E "python.*proteintrainer|python.*run_app.py|python.*gui" | grep -v grep | awk '{print $2}')
if [ ! -z "$pids" ]; then
    echo "   Found Proteintrainer Python processes: $pids"
    echo $pids | xargs kill -9 2>/dev/null && echo "   ✅ Killed Python processes"
fi

# Kill any gradio processes
pids=$(ps aux | grep gradio | grep -v grep | awk '{print $2}')
if [ ! -z "$pids" ]; then
    echo "   Found Gradio processes: $pids"
    echo $pids | xargs kill -9 2>/dev/null && echo "   ✅ Killed Gradio processes"
fi

echo "✅ Cleanup complete. You can now start Proteintrainer." 