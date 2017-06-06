## Semantic Segmentation of Indoor Scenes

### Dataset

Donwload prepared HDF5 data for training:

    sh download_data.sh

(optional) Download 3D indoor parsing dataset (<a href="http://buildingparser.stanford.edu/dataset.html">S3DIS Dataset</a>) for testing and visualization. Version 1.2 of the dataset is used in this work.


To prepare your own HDF5 data, you need to firstly download 3D indoor parsing dataset and then use `python collect_indoor3d_data.py` for data re-organization and `python gen_indoor3d_h5.py` to generate HDF5 files.

### Training

Once you have downloaded prepared HDF5 files or prepared them by yourself, to start training:

    python train.py --log_dir log6 --test_area 6
    
In default a simple model based on vanilla PointNet is used for training. Area 6 is used for test set.

### Testing

Testing requires download of 3D indoor parsing data and preprocessing with `collect_indoor3d_data.py`

After training, use `batch_inference.py` command to segment rooms in test set. In our work we use 6-fold training that trains 6 models. For model1 , area2-6 are used as train set, area1 is used as test set. For model2, area1,3-6 are used as train set and area2 is used as test set... Note that S3DIS dataset paper uses a different 3-fold training, which was not publicly announced at the time of our work.

For example, to test model6, use command:

    python batch_inference.py --model_path log6/model.ckpt --dump_dir log6/dump --output_filelist log6/output_filelist.txt --room_data_filelist meta/area6_data_label.txt --visu

Some OBJ files will be created for prediciton visualization in `log6/dump`.

To evaluate overall segmentation accuracy, we evaluate 6 models on their corresponding test areas and use `eval_iou_accuracy.py` to produce point classification accuracy and IoU as reported in the paper. 


