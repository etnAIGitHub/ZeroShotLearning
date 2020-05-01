@ECHO OFF 
:: This batch file create a requirements.txt file containing used python packages
TITLE Create Python Requirement File
ECHO requirements.txt created!
pip freeze > requirements.txt