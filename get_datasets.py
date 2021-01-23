import os
from PIL import Image


root = './images'
imgs = []
pngs = []
for folder in os.listdir(root):
    sub_folder = os.path.join(root, folder)
    for cat in os.listdir(sub_folder):
        cat_folder = os.path.join(sub_folder, cat)
        for img in os.listdir(cat_folder):
            if img.endswith('png'):
                pngs.append(os.path.join(cat_folder, img))
            else:
                imgs.append(os.path.join(cat_folder, img))
print(len(imgs))
print(pngs)

for img_fn in imgs+pngs:
    try:
        img = Image.open(img_fn)
        #exif_data = img._getexif()
    except ValueError as err:
        print(err)
        print("Error on image: ", img_fn)



