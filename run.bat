@echo off
echo ==========================
echo Running Driving DDS Project
echo ==========================

:: Step 1 - Train Random Forest & MLP Models
echo.
echo Training models with train_model.py...
python src\train_model.py

:: Step 2 - Predict with trained models
echo.
echo Running predictions on test data...
python src\predict.py

:: Step 3 - Run DDS Genetic Algorithm
echo.
echo Running DDS with Genetic Algorithm...
python src\dds_ga.py

echo.
echo All steps completed. Models saved in /models folder.
pause
