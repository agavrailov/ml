import os

file_path = 'data/raw/nvda_minute.csv'

if os.path.exists(file_path):
    if os.path.getsize(file_path) > 0:
        print("File exists and is not empty.")
    else:
        print("File exists but is empty.")
else:
    print("File does not exist.")