import pandas as pd
import shutil 
import os

categories = os.listdir('images/train')
for cat in categories:
    os.makedirs(os.path.join('images_fraction_1_5', 'train', cat), exist_ok=True)
    os.makedirs(os.path.join('images_fraction_2_5', 'train', cat), exist_ok=True)
    os.makedirs(os.path.join('images_fraction_3_5', 'train', cat), exist_ok=True)
    os.makedirs(os.path.join('images_fraction_4_5', 'train', cat), exist_ok=True)



df1 = pd.read_csv('./data_list/fraction_train_15.csv')
df2 = pd.read_csv('./data_list/fraction_train_25.csv')
df3 = pd.read_csv('./data_list/fraction_train_35.csv')
df4 = pd.read_csv('./data_list/fraction_train_45.csv')


df1.columns = ['id', 'image_path']
src_imgs1 = list(df1['image_path'])
src_imgs1 = [img.strip() for img in src_imgs1]

df2.columns = ['id', 'image_path']
src_imgs2 = list(df2['image_path'])
src_imgs2 = [img.strip() for img in src_imgs2]

df3.columns = ['id', 'image_path']
src_imgs3 = list(df3['image_path'])
src_imgs3 = [img.strip() for img in src_imgs3]

df4.columns = ['id', 'image_path']
src_imgs4 = list(df4['image_path'])
src_imgs4 = [img.strip() for img in src_imgs4]

for img in src_imgs1:
    shutil.copy(src=img, dst=img.replace('images_bkp', 'images_fraction_1_5'))
    
for img in src_imgs2:
    shutil.copy(src=img, dst=img.replace('images_bkp', 'images_fraction_2_5'))
    
for img in src_imgs3:
    shutil.copy(src=img, dst=img.replace('images_bkp', 'images_fraction_3_5'))
    
for img in src_imgs4:
    shutil.copy(src=img, dst=img.replace('images_bkp', 'images_fraction_4_5'))
