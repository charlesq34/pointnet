""" Utility functions for data loading and processing.
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import numpy as np
import h5py

SAMPLING_BIN = os.path.join(BASE_DIR, 'third_party/mesh_sampling/build/pcsample')

SAMPLING_POINT_NUM = 2048
SAMPLING_LEAF_SIZE = 0.005

MODELNET40_PATH = '../datasets/modelnet40'

def export_ply(pc, filename):
	vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
	for i in range(pc.shape[0]):
		vertex[i] = (pc[i][0], pc[i][1], pc[i][2])
	ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])])
	ply_out.write(filename)

# Sample points on the obj shape
def get_sampling_command(obj_filename, ply_filename):
    cmd = SAMPLING_BIN + ' ' + obj_filename
    cmd += ' ' + ply_filename
    cmd += ' -n_samples %d ' % SAMPLING_POINT_NUM
    cmd += ' -leaf_size %f ' % SAMPLING_LEAF_SIZE
    return cmd

# --------------------------------------------------------------
# Following are the helper functions to load MODELNET40 shapes
# --------------------------------------------------------------

# Read in the list of categories in MODELNET40
def get_category_names():
    shape_names_file = os.path.join(MODELNET40_PATH, 'shape_names.txt')
    shape_names = [line.rstrip() for line in open(shape_names_file)]
    return shape_names

# Return all the filepaths for the shapes in MODELNET40 
def get_obj_filenames():
    obj_filelist_file = os.path.join(MODELNET40_PATH, 'filelist.txt')
    obj_filenames = [os.path.join(MODELNET40_PATH, line.rstrip()) for line in open(obj_filelist_file)]
    print('Got %d obj files in modelnet40.' % len(obj_filenames))
    return obj_filenames

# Helper function to create the father folder and all subdir folders if not exist
def batch_mkdir(output_folder, subdir_list):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for subdir in subdir_list:
        if not os.path.exists(os.path.join(output_folder, subdir)):
            os.mkdir(os.path.join(output_folder, subdir))

# ----------------------------------------------------------------
# Following are the helper functions to load save/load HDF5 files
# ----------------------------------------------------------------

# Write numpy array data and label to h5_filename
def save_h5_data_label_normal(h5_filename, data, label, normal, 
		data_dtype='float32', label_dtype='uint8', noral_dtype='float32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'normal', data=normal,
            compression='gzip', compression_opts=4,
            dtype=normal_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()


# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()


# Write numpy array data and label to h5_filename
def save_h5_bbox_label(h5_filename, data, label, bbox_label, data_dtype='uint8', label_dtype='uint8', bbox_label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.create_dataset(
            'bbox_label', data=bbox_label,
            compression='gzip', compression_opts=4,
            dtype=bbox_label_dtype)
    h5_fout.close()


def load_h5_data_bbox_label(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    bbox_label = f['bbox_label'][:]
    return (data, label, bbox_label)



# Read numpy array data and label from h5_filename
def load_h5_data_label_normal(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    normal = f['normal'][:]
    return (data, label, normal)

# Read numpy array data and label from h5_filename
def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)

# Read numpy array data and label from h5_filename
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

# ----------------------------------------------------------------
# Following are the helper functions to load save/load PLY files
# ----------------------------------------------------------------

# Load PLY file
def load_ply_data(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data[:point_num]
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

# Load PLY file
def load_ply_normal(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['normal'].data[:point_num]
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

# Make up rows for Nxk array
# Input Pad is 'edge' or 'constant'
def pad_arr_rows(arr, row, pad='edge'):
    assert(len(arr.shape) == 2)
    assert(arr.shape[0] <= row)
    assert(pad == 'edge' or pad == 'constant')
    if arr.shape[0] == row:
        return arr
    if pad == 'edge':
        return np.lib.pad(arr, ((0, row-arr.shape[0]), (0, 0)), 'edge')
    if pad == 'constant':
        return np.lib.pad(arr, ((0, row-arr.shape[0]), (0, 0)), 'constant', (0, 0))


# ---------------------------------------------------------
# Point Cloud Pre-processing
# ----------------------------------------------------------



# Shuffle data and labels
def shuffle_data(data, labels):
	idx = np.arange(len(labels))
	np.random.shuffle(idx)
	return data[idx, ...], labels[idx], idx

# print both to console and the file flog
def printout(flog, data):
	print data
	flog.write(data + '\n')

# shuffle point cloud input order
# current_data: BxNx3 a batch of 3d points
def change_point_cloud_order(current_data, random_order=True, sorted_order=False):
    assert(random_order != sorted_order)
    # Naive shuffle begin
    if random_order:
        for k in range(current_data.shape[0]):
            seq = np.arange(current_data.shape[1])
            np.random.shuffle(seq)
            current_data[k, ...] = current_data[k, seq, ...]
    # Sort the points according to x first, then y, then z
    elif sorted_order:
        for k in range(current_data.shape[0]):
            pc = current_data[k, :, :]
            seq = np.lexsort((pc[:, 2], pc[:, 1], pc[:, 0]))
            current_data[k, ...] = current_data[k, seq, ...]
    return current_data

# randomly rotate the point clouds to augument the dataset
# rotation is per shape based
# batch data is BxNx3
def rotate_point_cloud(batch_data):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in xrange(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

# Add small elevation/tilt perturbation 10 degree
def rotate_point_cloud_v2(batch_data, rad_range=10/180.0*np.pi):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in xrange(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rot_angle = np.random.uniform(-rad_range, rad_range)
        cval = np.cos(rot_angle)
        sval = np.sin(rot_angle)
        rotation_matrix = np.dot(rotation_matrix, np.array([[1,0,0],[0,cval,-sval],[0,sval,cval]]))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_with_rotation_degrees(batch_data, rot_range=2*np.pi):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotation_degrees = np.zeros(batch_data.shape[0], dtype=np.float32)
    for k in xrange(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * rot_range - rot_range / 2
	rotation_degrees[k] = rotation_angle
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data, rotation_degrees


def rotate_point_cloud_with_normal(batch_data, batch_normal):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotated_normal = np.zeros(batch_normal.shape, dtype=np.float32)
    for k in xrange(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        shape_normal = batch_normal[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_normal[k, ...] = np.dot(shape_normal.reshape((-1, 3)), np.linalg.inv(rotation_matrix.T))
    return rotated_data, rotated_normal

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def insert_random_noise_point(batch_data, ratio=0.05):
    """ batch data: BxNx3 """
    num_noise_pt = int(batch_data.shape[1]*ratio)
    if num_noise_pt == 0: return batch_data
    noise_pts = np.random.random((batch_data.shape[0],num_noise_pt, 3)) * 2 - 1
    batch_data[:,0:num_noise_pt,:] = noise_pts
    return batch_data

def random_shift_point_cloud(batch_data, shift_range=0.1):
    """ batch_data is BxNxC. globally shift each point """
    B, N, C = batch_data.shape
    shifted_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in xrange(B):
        shift = np.array([np.random.uniform(-shift_range, shift_range) for _ in range(C)])
        shifted_data[k, ...] += shift
    return shifted_data

def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ batch_data is BxNx3, rotation_angle is single float number (rad) """    
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in xrange(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_2d_point_cloud(batch_data, angle_range=0.25*np.pi):
    """ batch_data is BxNx2 """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in xrange(batch_data.shape[0]):
        rotation_angle = (np.random.uniform()*2 - 1) * angle_range
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval],
                                     [sinval, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 2)), rotation_matrix)
    return rotated_data

def point_cloud_dropout(batch_data, keep_prob=0.9):
    """ batch_data is BxNxC """
    B, N, C = batch_data.shape
    output_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in xrange(B):
        keep = np.array([np.random.uniform()<=keep_prob for _ in range(N)])
        output_data[k, keep, :] = batch_data[k, keep, :]
    return output_data


def cut_point_cloud_by_plane(batch_data, plane=None):
    """ batch_data is BxNx3 with centroid at around (0,0,0)
        we will cut each point cloud by a random plane passing the origin:
        aX + bY + cZ = 0, and mark the cut out points as one of the 
        left point (assume we use MAX pooling for aggregation).
    """
    B, N, C = batch_data.shape
    output_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in xrange(B):
        if plane is None:
            plane = np.random.random(3) * 2 - 1.0
        keep = np.sum(batch_data[k, :, :] * plane, 1) > 0
        if len(keep) == 0:
            output_data[k, ...] = batch_data[k, 0, :]
        else:
            output_data[k, 0:np.sum(keep), :] = batch_data[k, keep, :]
            output_data[k, np.sum(keep):, :] = output_data[k, 0, :]
    return output_data

# For a new incoming/testing object
# First, convert it from obj to ply
# Then, do some pre-processing (e.g. normalization)
def obj2point_cloud(obj_filename):
    # Convert obj to normalized point cloud
    output_ply = '/tmp/tmp.ply'
    cmd = get_sampling_command(obj_filename, output_ply)
    os.system(cmd)
    point_cloud = pad_arr_rows(load_ply_data(output_ply, SAMPLING_POINT_NUM), SAMPLING_POINT_NUM)
    
    # Normalize the point cloud
    centroid = np.mean(point_cloud, axis=0)
    # Move the pc to the center
    point_cloud -= centroid

    # Normalize scale to make it within a unit sphere
    furthest_distance = np.max(np.sqrt(np.sum(abs(point_cloud)**2,axis=-1)))
    point_cloud /= furthest_distance

    return point_cloud

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def loadDataFile_with_bbox_label(filename):
    return load_h5_data_bbox_label(filename)

def loadDataFile(filename):
    return load_h5(filename)

def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)

def loadDataFile_with_normal(filename):
    return load_h5_data_label_normal(filename)

def dump_data_normal_into_pts_files(outdir, data, normal, gt_normal):
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	for i in range(data.shape[0]):
                dist = np.sqrt(np.sum(normal[i, :, :]**2, axis=1)).reshape((data.shape[1], 1)) * np.ones((1, 3))
		res = np.concatenate((data[i, ...], normal[i, ...]/dist), axis=1)
		np.savetxt(os.path.join(outdir, str(i)+'_pred.pts'), res)
                dist = np.sqrt(np.sum(gt_normal[i, :, :]**2, axis=1)).reshape((data.shape[1], 1)) * np.ones((1, 3))
		res = np.concatenate((data[i, ...], gt_normal[i, ...]/dist), axis=1)
		np.savetxt(os.path.join(outdir, str(i)+'_gt.pts'), res)

