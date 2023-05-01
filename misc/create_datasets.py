import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pandas as pd


image_dataset_dir = "C:/Users/friedele/Repos/DRFM/images/allTgts"
new_dataset_folder = "C:/Users/friedele/Repos/DRFM/images/new_targets"


dataset = {
    "image" :[],
    "label" : []
}
for label in os.listdir(image_dataset_dir):
     images_dir= image_dataset_dir + "/" + label
     # if not os.path.isdir(images_dir):
     #    continue
     for image_file in os.listdir(images_dir):
        if not image_file.endswith(".JPG", ".PNG",".png"):
            continue 
        img = load_img(os.path.join(image_dataset_dir, label, image_file))
        x = img_to_array(img)                  
        

        rel_path = label + "/" + os.path.splitext(image_file)[0] + '.npz'
        os.makedirs(new_dataset_folder + "/" + label, exist_ok=True)
        npz_file = os.path.join(new_dataset_folder, rel_path)
        np.savez(npz_file, x)
        dataset["image"].append(rel_path)
        dataset["label"].append(label)

                         
df = pd.DataFrame(dataset)
df.to_csv(os.path.join(new_dataset_folder, "train.csv"), index=False)

print('Dataset converted to npz and saved here at %s '%new_dataset_folder)

df.head()

