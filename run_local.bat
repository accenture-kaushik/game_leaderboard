@echo off
REM Change to the folder where this .bat file lives
cd /d "%~dp0"

echo ==========================================
echo   Sports Leaderboard - Local Launcher
echo ==========================================
echo.

REM Try "py -m streamlit" (Windows Python launcher - most reliable)
py -m streamlit run streamlit_app.py --server.port 8501
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Could not start Streamlit.
    echo Make sure dependencies are installed:
    echo   py -m pip install -r requirements.txt
    echo.
)

pause
