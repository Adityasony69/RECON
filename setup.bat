@echo off
echo Setting up Project Recon...

python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt

echo.
echo Setup complete.
echo 1. Put Fake.csv and True.csv in the Dataset folder (see README)
echo 2. Add GROQ_API_KEY=your_key to a .env file (optional, for explanations)
echo 3. Run: python app.py
pause
