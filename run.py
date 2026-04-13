"""
Production entry point for Azure App Service.
Runs Streamlit on the port Azure expects ($WEBSITES_PORT or $PORT).
"""

import os
import subprocess
import sys

port = os.getenv("WEBSITES_PORT", os.getenv("PORT", "8501"))

subprocess.run(
    [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ],
    check=True,
)
