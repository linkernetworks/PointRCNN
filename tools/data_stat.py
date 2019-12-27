# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
print(os.getcwd())
try:
    os.chdir(os.path.join(os.getcwd(), 'tools'))
    print(os.getcwd())
except:
    pass
# %%
from IPython import get_ipython

# %%
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os

# %%
cat_map = []
root = '//home/linker/data_disk/projects/pointrcnn1/data/KITTI/object/training/label_2'
# root = '/home/linker/data_disk/kitti_transform_data/shuttle_3k/train/label_2'
tmp = [x for x in os.listdir(root) if x.endswith('txt')]
for i in tmp:
    filename = os.path.join(root, i)
    with open(filename) as f:
        content = f.readlines()
    for line in content:
        split_line = line.split(' ')
        category = split_line[0]
        cat_map.append([category, split_line[8], 'h'])
        cat_map.append([category, split_line[9], 'w'])
        cat_map.append([category, split_line[10], 'l'])
        cat_map.append([category, split_line[11], 'x'])
        cat_map.append([category, split_line[12], 'y'])
        cat_map.append([category, split_line[13], 'z'])
        # if category == 'car' and (float(split_line[11]) <-100 or float(split_line[13])<-100):
        #     print(filename)
df = pd.DataFrame(cat_map, columns=['Category', 'Size', 'Dimension'])
df['Size'] = df['Size'].astype(float)

# %%
import json

# %%
cat_map = []
root = '/home/linker/data_disk/projects/raw_parser/kitti/shuttle'
tmp = [os.path.join(root, x) for x in os.listdir(root) if x.endswith('json')]
for json_file_path in tmp:
    with open(json_file_path, 'r') as f:
        data = json.loads(f.read())
        for sen in data['frame_list']:
            for label in sen['labels']:
                category = label['category']
                # print(label.keys())
                if 'box3d' in label.keys():
                    if 'box2ds' in label.keys() and label['box2ds']:
                        cat_map.append([category, '2dn3d'])
                    else:
                        cat_map.append([category, '3d'])
                else:
                    cat_map.append([category, '2d'])
df = pd.DataFrame(cat_map, columns=['Category', 'Dimension'])

# %%
df

# %%
df.head()

# %%
sns.set(style="ticks", palette="pastel")

# Load the example tips dataset
# tips = sns.load_dataset("tips")

# Draw a nested boxplot to show bills by day and time
cat_list = df.Category.unique()
h_count = 4
f, axes = plt.subplots(round(len(cat_list) / h_count),
                       h_count,
                       figsize=(20, 20))
for i, key in enumerate(cat_list):
    sns_plot = sns.boxplot(x="Category",
                           y="Size",
                           hue="Dimension",
                           data=df[df["Category"] == key],
                           ax=axes[i // h_count][i % h_count])
sns_plot.figure.savefig("figs/output_roof_size.png")
# df.boxplot(column=['w','h','l'],by="Category")
# sns.despine(offset=10, trim=True)

# %%

# %%
df[df['Dimension'] == '3d'].groupby('Category').describe()
# %%
df.groupby(['Category',
            'Dimension']).describe()  #.to_csv("center_description.csv")
# %%
df[df['Category'] == 'cyclist'].groupby('Dimension').describe()
# %%
pole_df = df[df.Category == 'pedestrian']
pole_df[pole_df.Dimension == 'y'].sort_values(by='Size').tail(10)

# %%
import os

# %%
tmp = [
    x for x in os.listdir('//mnt/data_hdd_2T/bdd_kitti_10_30/train/label_2')
    if x.endswith('txt')
]
tmp.sort()
len(tmp)
