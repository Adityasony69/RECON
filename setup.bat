@echo off 
echo Setting up Project Environment...

git init
git remote add origin https://github.com/Adityasony69/Project-RECON.git
python -m venv venv
call venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
echo Setup complete!
pause
