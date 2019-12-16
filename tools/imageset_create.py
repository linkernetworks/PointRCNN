# %%
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'tools'))
    print(os.getcwd())
except:
    pass
    print(os.getcwd())

# %%
tmp = [
    x.replace('.txt', '')
    for x in os.listdir('../data/KITTI/object/training/label_2/')
    if x.endswith('.txt')
]
tmp.sort()
print(tmp[-10:])
print(os.getcwd())
# '%06d'%
# %%
val = np.random.choice(tmp, 2000, replace=False)
train = [seq for seq in tmp if not seq in val]
# %%
print(len(val))
print(len(train))
print(len(tmp))
# %%
with open("../data/KITTI/ImageSets/trainval.txt", "w") as txt_file:
    for line in tmp:
        txt_file.write(str(line) + "\n")
# %%
with open("../data/KITTI/ImageSets/train.txt", "w") as txt_file:
    for line in train:
        txt_file.write(str(line) + "\n")
# %%
with open("../data/KITTI/ImageSets/val.txt", "w") as txt_file:
    for line in val:
        txt_file.write(str(line) + "\n")
# %%
val_set = np.loadtxt("../data/KITTI/ImageSets/val.txt", delimiter='\n')
# %%
val_set = np.sort(val_set.astype(int))
val_set
#%%
with open("../data/KITTI/ImageSets/val.txt", "w") as txt_file:
    for line in val_set:
        txt_file.write('%06d' % line + "\n")
