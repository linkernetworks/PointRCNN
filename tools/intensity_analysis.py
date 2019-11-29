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
import numpy as npP
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from scipy.spatial import cKDTree
from numpy import linalg as LA
from sklearn.linear_model import LinearRegression
from os.path import join as join
# %%
V2C = np.array([[
    0.01590444566507537, -0.999817423558399, 0.010590946942147976,
    -0.003022237050537676
],
                [
                    -0.017388221825025585, -0.010867254373478057,
                    -0.9997897541604163, -0.1973658196996334
                ],
                [
                    0.9997223106192291, 0.015716944086776208,
                    -0.017557884802070593, -0.04413746879679806
                ]])
V2C = np.round(V2C)
# V2C = np.array([[0.0,-1.0,0.0,0.0],[0,0,-1,0],[1,0,0,0]])
# inverse_rigid_trans(V2C)
# C2V = np.array([[ 1.51029500e-02,  1.31259700e-02,  9.99799780e-01, 1.27166100e-01],
#                 [-9.99885670e-01,  9.39210000e-04,  1.50919100e-02, -1.12410492e-02],
#                 [-7.40920000e-04, -9.99913410e-01,  1.31386600e-02, -9.21868710e-02]])

C2V = np.linalg.inv(np.vstack((V2C, [0, 0, 0, 1])))[:3]


#%%
def check_scope(data, scope):
    x_scope, y_scope, z_scope = scope
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    idx = (x_scope[0] <= x) & (x <= x_scope[1]) & (y_scope[0] <= y) & (
        y <= y_scope[1]) & (z_scope[0] <= z) & (z <= z_scope[1])
    return data[idx]


def get_points(points, center, hwl, yaw):
    h, w, l = [float(x) for x in hwl]
    # print(center)
    center = np.array(center, dtype=np.float)
    scope = np.vstack([
        center - np.array([w / 2 * np.cos(yaw), h, l / 2 * np.sin(yaw)]),
        center + np.array([w / 2 * np.cos(yaw), 0, l / 2 * np.sin(yaw)])
    ])
    scope = ([scope[0][0],
              scope[1][0]], [scope[0][1],
                             scope[1][1]], [scope[0][2], scope[1][2]])
    # print(scope)
    return check_scope(points, scope), scope


def get_closest_dist(points):
    tree = cKDTree(points)
    distance = []
    for point in points:
        distance.append(tree.query(point, 2)[0][1])
    return distance


# %%
cat_map = []
# encoding=utf8
root = '/home/linker/data_disk/kitti_transform_data/bdd_kitti_headlamp_qa/train/label_2'
tmp = [x for x in os.listdir(root) if x.endswith('txt')]
tmp.sort()
no_point = 0
for i in tmp:
    filename = os.path.join(root, i)
    lidar_path = filename.replace('txt', 'bin').replace('label_2', 'velodyne')
    Raw = np.fromfile(lidar_path,
                      dtype=np.float32).reshape(-1, 4)  # u, v, d, r, x, y, z
    # print(Raw.shape)
    org_points = Raw[:, :3]
    org_points = org_points[np.any(org_points != [0, 0, 0], axis=1)]
    org_points = np.hstack((org_points, np.ones((org_points.shape[0], 1))))
    # print(org_points.shape)
    org_points = org_points @ V2C.T
    print(org_points.shape)
    break
    with open(filename) as f:
        content = f.readlines()
        for line in enumerate(content):
            line = line[1]
            split_line = line.split(' ')
            category = split_line[0]
            hwl = split_line[8:11]
            yaw = float(split_line[14])
            center = split_line[11:14]
            points, scope = get_points(org_points, center, hwl, yaw)
            if not len(points):
                print(i)
                print(category)
                print(center)
                print(hwl)
                print(scope)
                no_point += 1
                print('No points count: %d' % no_point)
                raise ArithmeticError
                continue
            pair_distance = get_closest_dist(points)
            cat_map.append([
                category,
                LA.norm(np.array(center, dtype=np.float)),
                np.median(pair_distance),
                np.max(pair_distance),
                np.percentile(pair_distance, 90),
                float(split_line[14]) * 180 / np.pi
            ])
            break
    # break
# %%
import pickle
import _init_path
import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.config import cfg

# %%
root = './gt_database_r2'
gt_filepath_list = [
    join(root, rel_path) for rel_path in os.listdir(root)
    if rel_path.endswith('.pkl')
]
print(gt_filepath_list)
gt_database_list = [
    pickle.load(open(file_path, 'rb')) for file_path in gt_filepath_list
]

for gt_database in gt_database_list:
    if not len(gt_database):
        continue
    np_set = [
        gt_data for gt_data in gt_database if
        not gt_data['points'].shape[0]  #> 0 and gt_data['points'].shape[0] < 5
    ].copy()
    print(gt_database[0]['cls_type'])
    print(len(gt_database))
    print(len(np_set))
    lines = open(
        '/home/linker/data_disk/kitti_transform_data/bdd_kitti_headlamp_qa_r2/map.txt'
    ).readlines()
    seperator = ' '
    for np_obj in np_set:
        print(
            seperator.join(lines[int(
                np_obj['sample_id'])].split(' ')[1:]).replace('\n', ''))

# %%
gt_database[-1]
# %%
np_set = [
    gt_data for gt_data in gt_database
    if not gt_data['points'].shape[0]  #> 0 and gt_data['points'].shape[0] < 5
].copy()
print(len(gt_database))
print(len(np_set))
# %%
np_set
# %%
lines = open(
    '/home/linker/data_disk/kitti_transform_data/bdd_kitti_headlamp_qa_r1_ext/map.txt'
).readlines()
seperator = ' '
for np_obj in np_set:
    print(
        seperator.join(lines[int(np_obj['sample_id'])].split(' ')[1:]).replace(
            '\n', ''))
# %%
lines = open(
    '/home/linker/data_disk/kitti_transform_data/bdd_kitti_headlamp_qa_r2/map.txt'
).readlines()
seperator = ' '
print(seperator.join(lines[939].split(' ')[1:]).replace('\n', ''))
# %%
point_count = [gt_data['points'].shape[0] for gt_data in gt_database]
bin_edges = np.arange(0, 500, 20)
vals = plt.hist(point_count, density=False, bins=bin_edges)
plt.ylabel('Cuboid Count')
plt.xlabel('# of Point in Cuboid')
plt.title('# of Point in Cuboid Histogram, binsize 20')

# %%
center_list
# %%
center_list = np.vstack([gt_data['gt_box3d'][:3] for gt_data in gt_database])
size_list = np.vstack([gt_data['gt_box3d'][3:6] for gt_data in gt_database])
center_list = size_list
print(center_list.shape)
df_dim = pd.DataFrame()
df_dim['x'] = center_list[:, 0]
df_dim['y'] = center_list[:, 1]
df_dim['z'] = center_list[:, 2]
print(np.mean(center_list, axis=0))
sns.boxplot(data=center_list[:, 2])
# %%
bin_edges = np.arange(0, 40, 5)
#-size_list[:, 0]
vals = plt.hist(center_list[:, 2], density=True, bins=bin_edges)
np.sum(vals[0][:4]) * 5
# %%
# Center distance average
np.mean(
    np.sqrt(np.sum((center_list - np.mean(center_list, axis=0))**2, axis=1)))
# Axis variance

# %%
sample = gt_database[0]
tmp_points = sample['points']
d_x = np.abs(np.max(tmp_points[:, 0]) - np.min(tmp_points[:, 0]))
d_z = np.abs(np.max(tmp_points[:, 2]) - np.min(tmp_points[:, 2]))
yaw = sample['gt_box3d'][-1]
print(yaw / np.pi * 180)
cos_y = np.abs(np.cos(yaw))
sin_y = np.abs(np.sin(yaw))
print('Deltas: %02f %02f' % (d_x, d_z))
print('Degrees: %02f, %02f' % (cos_y, sin_y))
# %%
if (sin_y < 0.1):
    l = d_x / cos_y
    w = d_z / cos_y
elif cos_y < 0.1:
    l = d_z / sin_y
    w = d_x / sin_y
else:
    l = np.max([d_x / cos_y, d_z / sin_y])
    w = np.max([d_z / cos_y, d_x / sin_y])

print('Width:%02f, length:%02f' % (w, l))
print(sample['gt_box3d'][3:6])
# %%
no_point = 0
cat_map = []
for object in gt_database:
    points = object['points']
    category = object['cls_type']
    if not len(points):
        no_point += 1
        print('No points count: %d' % no_point)
        continue
    pair_distance = get_closest_dist(points)
    cat_map.append([
        category,
        LA.norm(object['gt_box3d'][:3], dtype=np.float),
        np.median(pair_distance),
        np.max(pair_distance),
        np.percentile(pair_distance, 90),
        float(object['gt_box3d'][-1]) * 180 / np.pi
    ])
# %%
import pandas as pd
df = pd.DataFrame(cat_map,
                  columns=[
                      'Category', 'Distance', 'Pairwise_median',
                      'Pairwise_max', 'Pairwise_90', 'yaw'
                  ])

# %%
df = df.sort_values(by=['Distance'])

# %%
df_cat = df[df.Category == 'car']
bin_edges = np.arange(-180, 180, 15)
vals = plt.hist(df_cat.yaw, density=True,
                bins=bin_edges)  # arguments are passed to np.histogram
plt.title("Angle histogram")
# Text(0.5, 1.0, "Histogram with 'auto' bins")
plt.show()

# %%
df.groupby(['Category']).describe()
# %%
cat_list = df.Category.unique()
for cat in cat_list:
    df_cat = df[df.Category == cat]
    plt.figure()
    plt.title('Category : %s' % cat)
    df_cat = df_cat.dropna()
    X = df_cat.Distance.values
    y = df_cat.Pairwise_90.values
    plt.scatter(X, y, s=1)
    reg = LinearRegression().fit(X.reshape(-1, 1), y.reshape(-1, 1))
    X = np.arange(int(np.max(df_cat.Distance)))
    y = reg.predict(X.reshape(-1, 1))
    print('Category : %s' % cat)
    print('Params: %f, %f' % (reg.intercept_, reg.coef_))
    print('Max dist:%f eps:%f' % (X[-1], y[-1]))
    plt.plot(X, y, c='r')
    plt.xlabel('Distance')
    plt.ylabel('90th quantile of closest distance')


# %%
def get_inner_points(points: np.ndarray,
                     center: np.ndarray,
                     hwl: np.ndarray,
                     maxDist=10):
    delta_d = points - center
    h, w, l = hwl
    mask = (np.abs(delta_d[:,0]) < maxDist) & (np.abs(delta_d[:,1]) < h/2) & \
    (np.abs(delta_d[:,2]) < maxDist)
    points = points[mask]
    delta_d = delta_d[mask]
    x_rot = delta_d[:, 0] * np.cos(yaw) + delta_d[:, 2] * (-np.sin(yaw))
    z_rot = delta_d[:, 0] * np.sin(yaw) + delta_d[:, 2] * np.cos(yaw)
    mask = (np.abs(x_rot) <= l / 2) & (np.abs(z_rot) <= w / 2)
    return points[mask]


# %%
def get_inner_points_simple(points: np.ndarray,
                            center: np.ndarray,
                            hwl: np.ndarray,
                            maxDist=10):
    delta_d = points - center
    h, w, l = hwl
    x_rot = delta_d[:, 0] * np.cos(yaw) + delta_d[:, 2] * (-np.sin(yaw))
    z_rot = delta_d[:, 0] * np.sin(yaw) + delta_d[:, 2] * np.cos(yaw)
    mask = (np.abs(x_rot) <= l / 2) & (np.abs(z_rot) <= w / 2) & (np.abs(
        delta_d[:, 1]) < h / 2)
    return points[mask]


# %%
import time
hwl = np.array([1.2, 1.45, 3.95])
center = np.array([48.59, 0.80, 45.12])
yaw = -0.72
rawFile = np.fromfile('../data/KITTI/object/training/velodyne/000000.bin',
                      dtype=np.float32).reshape(-1, 4)
V2C = np.array([[0., -1., 0.], [-0., -0., -1.], [1., 0., -0.]])
lidar_cam = rawFile[:, :3] @ V2C.T
prevTime = time.time()
for i in range(1000):
    get_inner_points_simple(lidar_cam, center, hwl)
print(time.time() - prevTime)
# %%
os.listdir('/home/linker/data_disk/lgecto/linkerkcstorage/')
