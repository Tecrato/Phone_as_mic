@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ==== Rutas del venv ====
set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQ_FILE=requirements.txt"

echo.
echo ============================================
echo   Mic remoto - arranque
echo ============================================
echo.

REM ==== 1) Crear venv si no existe ====
if not exist "%VENV_PY%" (
    echo [1/3] Creando entorno virtual en "%VENV_DIR%"...
    where python >nul 2>nul
    if errorlevel 1 (
        echo.
        echo [ERROR] No se encontro Python en el PATH.
        echo Instala Python 3.9+ desde https://www.python.org/downloads/
        echo y marca la casilla "Add Python to PATH" durante la instalacion.
        echo.
        pause
        exit /b 1
    )
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Fallo al crear el venv.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Venv ya existe, se reutiliza.
)

REM ==== 2) Instalar dependencias si faltan ====
"%VENV_PY%" -c "import fastapi, uvicorn, sounddevice, numpy, cryptography" >nul 2>nul
if errorlevel 1 (
    echo [2/3] Instalando dependencias ^(puede tardar unos minutos la primera vez^)...
    "%VENV_PY%" -m pip install --upgrade pip
    if exist "%REQ_FILE%" (
        "%VENV_PY%" -m pip install -r "%REQ_FILE%"
    ) else (
        "%VENV_PY%" -m pip install fastapi "uvicorn[standard]" sounddevice numpy cryptography
    )
    if errorlevel 1 (
        echo [ERROR] Fallo la instalacion de dependencias.
        pause
        exit /b 1
    )
) else (
    echo [2/3] Dependencias ya instaladas.
)

REM ==== 3) Arrancar servidor ====
echo [3/3] Arrancando servidor...
echo.
"%VENV_PY%" server.py

REM Si el servidor se cierra (Ctrl+C o error), no cerramos la ventana
echo.
echo Servidor detenido.
pause