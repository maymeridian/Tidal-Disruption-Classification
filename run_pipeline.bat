@echo off
echo ==========================================
echo      TDE CLASSIFIER PIPELINE 
echo ==========================================

:: Check if a model argument was provided (e.g. "random_forest")
IF "%~1"=="" (
    echo [INFO] No model specified. Using default from config.py.
    set COMMAND=python train.py
) ELSE (
    echo [INFO] Model override detected: %~1
    set COMMAND=python train.py --model %~1
)

:: 1. Run Training
echo.
echo [1/2] Starting Training...
%COMMAND%

:: Check if train.py failed
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Training failed! Aborting pipeline.
    pause
    exit /b %ERRORLEVEL%
)

:: 2. Run Prediction
echo.
echo [2/2] Starting Prediction...
python predict.py

:: Check if predict.py failed
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Prediction failed!
    pause
    exit /b %ERRORLEVEL%
)

:: 3. Success
echo.
echo ==========================================
echo      SUCCESS: Pipeline Complete!
echo ==========================================
pause