import os
import sys
import shutil

folder_path = sys.argv[1]

files = os.listdir(folder_path)
test_folder_path = "data/test/"
train_folder_path = "data/train/"

count_eval = 0
for file_name in files:
   file_addr = folder_path + '/' + file_name
   shutil.move(file_addr, test_folder_path + file_name)
    
   count_eval += 1
   if (count_eval >= 500):
       break

for file_name in files[500:]:
    file_addr = folder_path + '/' + file_name
    shutil.move(file_addr, train_folder_path + file_name)
