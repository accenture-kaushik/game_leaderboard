#!/bin/bash
set -e

echo "=== Sports Leaderboard startup ==="

# Start Flask API on port 5000 (internal, never exposed directly)
echo "Starting Flask API on :5000 ..."
gunicorn \
    --bind 0.0.0.0:5000 \
    --workers 1 \
    --timeout 120 \
    --log-level info \
    app:app &

FLASK_PID=$!

# Give Flask a moment to bind
sleep 3

# Azure sets $PORT; fall back to 8501
APP_PORT="${PORT:-8501}"
echo "Starting Streamlit on :${APP_PORT} ..."

exec streamlit run streamlit_app.py \
    --server.port="${APP_PORT}" \
    --server.address="0.0.0.0" \
    --server.headless=true \
    --browser.gatherUsageStats=false

# Cleanup (only reached if streamlit exits)
kill "$FLASK_PID" 2>/dev/null || true
