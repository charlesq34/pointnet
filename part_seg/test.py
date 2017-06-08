import argparse
import tensorflow as tf
import json
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import provider
import pointnet_part_seg as model

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='train_results/trained_models/epoch_190.ckpt', help='Model checkpoint path')
FLAGS = parser.parse_args()


# DEFAULT SETTINGS
pretrained_model_path = FLAGS.model_path # os.path.join(BASE_DIR, './pretrained_model/model.ckpt')
hdf5_data_dir = os.path.join(BASE_DIR, './hdf5_data')
ply_data_dir = os.path.join(BASE_DIR, './PartAnnotation')
gpu_to_use = 0
output_dir = os.path.join(BASE_DIR, './test_results')
output_verbose = True   # If true, output all color-coded part segmentation obj files

# MAIN SCRIPT
point_num = 3000            # the max number of points in the all testing data shapes
batch_size = 1

test_file_list = os.path.join(BASE_DIR, 'testing_ply_file_list.txt')

oid2cpid = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))

object2setofoid = {}
for idx in range(len(oid2cpid)):
    objid, pid = oid2cpid[idx]
    if not objid in object2setofoid.keys():
        object2setofoid[objid] = []
    object2setofoid[objid].append(idx)

all_obj_cat_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
fin = open(all_obj_cat_file, 'r')
lines = [line.rstrip() for line in fin.readlines()]
objcats = [line.split()[1] for line in lines]
objnames = [line.split()[0] for line in lines]
on2oid = {objcats[i]:i for i in range(len(objcats))}
fin.close()

color_map_file = os.path.join(hdf5_data_dir, 'part_color_mapping.json')
color_map = json.load(open(color_map_file, 'r'))

NUM_OBJ_CATS = 16
NUM_PART_CATS = 50

cpid2oid = json.load(open(os.path.join(hdf5_data_dir, 'catid_partid_to_overallid.json'), 'r'))

def printout(flog, data):
	print(data)
	flog.write(data + '\n')

def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

def output_color_point_cloud_red_blue(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            if seg[i] == 1:
                color = [0, 0, 1]
            elif seg[i] == 0:
                color = [1, 0, 0]
            else:
                color = [0, 0, 0]

            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def placeholder_inputs():
    pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, point_num, 3))
    input_label_ph = tf.placeholder(tf.float32, shape=(batch_size, NUM_OBJ_CATS))
    return pointclouds_ph, input_label_ph

def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

def load_pts_seg_files(pts_file, seg_file, catid):
    with open(pts_file, 'r') as f:
        pts_str = [item.rstrip() for item in f.readlines()]
        pts = np.array([np.float32(s.split()) for s in pts_str], dtype=np.float32)
    with open(seg_file, 'r') as f:
        part_ids = np.array([int(item.rstrip()) for item in f.readlines()], dtype=np.uint8)
        seg = np.array([cpid2oid[catid+'_'+str(x)] for x in part_ids])
    return pts, seg

def pc_augment_to_point_num(pts, pn):
    assert(pts.shape[0] <= pn)
    cur_len = pts.shape[0]
    res = np.array(pts)
    while cur_len < pn:
        res = np.concatenate((res, pts))
        cur_len += pts.shape[0]
    return res[:pn, :]

def convert_label_to_one_hot(labels):
    label_one_hot = np.zeros((labels.shape[0], NUM_OBJ_CATS))
    for idx in range(labels.shape[0]):
        label_one_hot[idx, labels[idx]] = 1
    return label_one_hot

def predict():
    is_training = False
    
    with tf.device('/gpu:'+str(gpu_to_use)):
        pointclouds_ph, input_label_ph = placeholder_inputs()
        is_training_ph = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, seg_pred, end_points = model.get_model(pointclouds_ph, input_label_ph, \
                cat_num=NUM_OBJ_CATS, part_num=NUM_PART_CATS, is_training=is_training_ph, \
                batch_size=batch_size, num_point=point_num, weight_decay=0.0, bn_decay=None)
        
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        flog = open(os.path.join(output_dir, 'log.txt'), 'w')

        # Restore variables from disk.
        printout(flog, 'Loading model %s' % pretrained_model_path)
        saver.restore(sess, pretrained_model_path)
        printout(flog, 'Model restored.')
        
        # Note: the evaluation for the model with BN has to have some statistics
        # Using some test datas as the statistics
        batch_data = np.zeros([batch_size, point_num, 3]).astype(np.float32)

        total_acc = 0.0
        total_seen = 0
        total_acc_iou = 0.0

        total_per_cat_acc = np.zeros((NUM_OBJ_CATS)).astype(np.float32)
        total_per_cat_iou = np.zeros((NUM_OBJ_CATS)).astype(np.float32)
        total_per_cat_seen = np.zeros((NUM_OBJ_CATS)).astype(np.int32)

        ffiles = open(test_file_list, 'r')
        lines = [line.rstrip() for line in ffiles.readlines()]
        pts_files = [line.split()[0] for line in lines]
        seg_files = [line.split()[1] for line in lines]
        labels = [line.split()[2] for line in lines]
        ffiles.close()

        len_pts_files = len(pts_files)
        for shape_idx in range(len_pts_files):
            if shape_idx % 100 == 0:
                printout(flog, '%d/%d ...' % (shape_idx, len_pts_files))

            cur_gt_label = on2oid[labels[shape_idx]]

            cur_label_one_hot = np.zeros((1, NUM_OBJ_CATS), dtype=np.float32)
            cur_label_one_hot[0, cur_gt_label] = 1

            pts_file_to_load = os.path.join(ply_data_dir, pts_files[shape_idx])
            seg_file_to_load = os.path.join(ply_data_dir, seg_files[shape_idx])

            pts, seg = load_pts_seg_files(pts_file_to_load, seg_file_to_load, objcats[cur_gt_label])
            ori_point_num = len(seg)

            batch_data[0, ...] = pc_augment_to_point_num(pc_normalize(pts), point_num)

            label_pred_val, seg_pred_res = sess.run([pred, seg_pred], feed_dict={
                        pointclouds_ph: batch_data,
                        input_label_ph: cur_label_one_hot, 
                        is_training_ph: is_training,
                    })

            label_pred_val = np.argmax(label_pred_val[0, :])
            
            seg_pred_res = seg_pred_res[0, ...]

            iou_oids = object2setofoid[objcats[cur_gt_label]]
            non_cat_labels = list(set(np.arange(NUM_PART_CATS)).difference(set(iou_oids)))

            mini = np.min(seg_pred_res)
            seg_pred_res[:, non_cat_labels] = mini - 1000

            seg_pred_val = np.argmax(seg_pred_res, axis=1)[:ori_point_num]

            seg_acc = np.mean(seg_pred_val == seg)

            total_acc += seg_acc
            total_seen += 1

            total_per_cat_seen[cur_gt_label] += 1
            total_per_cat_acc[cur_gt_label] += seg_acc

            mask = np.int32(seg_pred_val == seg)

            total_iou = 0.0
            iou_log = ''
            for oid in iou_oids:
                n_pred = np.sum(seg_pred_val == oid)
                n_gt = np.sum(seg == oid)
                n_intersect = np.sum(np.int32(seg == oid) * mask)
                n_union = n_pred + n_gt - n_intersect
                iou_log += '_' + str(n_pred)+'_'+str(n_gt)+'_'+str(n_intersect)+'_'+str(n_union)+'_'
                if n_union == 0:
                    total_iou += 1
                    iou_log += '_1\n'
                else:
                    total_iou += n_intersect * 1.0 / n_union
                    iou_log += '_'+str(n_intersect * 1.0 / n_union)+'\n'

            avg_iou = total_iou / len(iou_oids)
            total_acc_iou += avg_iou
            total_per_cat_iou[cur_gt_label] += avg_iou
            
            if output_verbose:
                output_color_point_cloud(pts, seg, os.path.join(output_dir, str(shape_idx)+'_gt.obj'))
                output_color_point_cloud(pts, seg_pred_val, os.path.join(output_dir, str(shape_idx)+'_pred.obj'))
                output_color_point_cloud_red_blue(pts, np.int32(seg == seg_pred_val), 
                        os.path.join(output_dir, str(shape_idx)+'_diff.obj'))

                with open(os.path.join(output_dir, str(shape_idx)+'.log'), 'w') as fout:
                    fout.write('Total Point: %d\n\n' % ori_point_num)
                    fout.write('Ground Truth: %s\n' % objnames[cur_gt_label])
                    fout.write('Predict: %s\n\n' % objnames[label_pred_val])
                    fout.write('Accuracy: %f\n' % seg_acc)
                    fout.write('IoU: %f\n\n' % avg_iou)
                    fout.write('IoU details: %s\n' % iou_log)

        printout(flog, 'Accuracy: %f' % (total_acc / total_seen))
        printout(flog, 'IoU: %f' % (total_acc_iou / total_seen))

        for cat_idx in range(NUM_OBJ_CATS):
            printout(flog, '\t ' + objcats[cat_idx] + ' Total Number: ' + str(total_per_cat_seen[cat_idx]))
            if total_per_cat_seen[cat_idx] > 0:
                printout(flog, '\t ' + objcats[cat_idx] + ' Accuracy: ' + \
                        str(total_per_cat_acc[cat_idx] / total_per_cat_seen[cat_idx]))
                printout(flog, '\t ' + objcats[cat_idx] + ' IoU: '+ \
                        str(total_per_cat_iou[cat_idx] / total_per_cat_seen[cat_idx]))

                
with tf.Graph().as_default():
    predict()
