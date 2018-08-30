#!bin/sh
#yanyu 2018.03.03
#usage:
#let we note models-master $MODELS_DIR
#1. set the DATASET to dataset you want to train and test,for voc it's pascal
#2. copy $MODELS_DIR/research/object_detection/ssd_mobilenet_v1_$DATASET.config to $DATASET/ssd_mobilenet_v1_$DATASET.config, and change the num_classes to be the num of classes, for voc it's 21
# search _label_map.pbtxt and set them to be $DATASET/$DATASET_label_map.pbtxt,search .record and set itt to be ${DATASET}/{DATASET}_train.record and {DATASET}/{DATASET}_val.record 
#3 open a new terminal and change directory to the dir of the file, run sh train.sh

#0. set the vars
DATASET=car
#DATASET=pascal
#DATASET=pet

#YEAR=VOC2012
#DATASET_DIR=/home/yanyu/data/voc/VOCdevkit

YEAR=DETRAC
DATASET_DIR=home/yanyu/data

OB_DIR=/home/yanyu/models/research/object_detection

#you do't have to change the following ones
PIPELINE_CONFIGPATH=$DATASET/ssd_mobilenet_v1_$DATASET.config
TRAIN_DIR=$DATASET/train_logs
EVAL_DIR=$DATASET/eval

#stage 1, generating tfrecords
if [ ! -e ${DATASET}/${DATASET}_train.record ]; then
echo ${DATASET}" tfrecords not exist, start converting"
python $DATASET/create_${DATASET}_tf_record.py --label_map_path=${DATASET}/${DATASET}_label_map.pbtxt --data_dir=$DATASET_DIR --year=$YEAR --set=train --output_path=${DATASET}/${DATASET}_train.record
python $DATASET/create_${DATASET}_tf_record.py --label_map_path=${DATASET}/${DATASET}_label_map.pbtxt --data_dir=$DATASET_DIR --year=$YEAR --set=val --output_path=${DATASET}/${DATASET}_val.record
else
echo ${DATASET}" already converted, using existed one"
fi

#stage 2, training
echo "start training"
python $OB_DIR/train.py --logtostderr --pipeline_config_path=$PIPELINE_CONFIGPATH --train_dir=$TRAIN_DIR --num_clones=2 2>&1 | tee $DATASET/train_logs.txt

#stage 3,evaluation
echo "start evaluation"
python $OB_DIR/eval.py --logtostderr --pipeline_config_path=$PIPELINE_CONFIGPATH --checkpoint_dir=$TRAIN_DIR --eval_dir=$EVAL_DIR

#stage 4,frozen models
echo "start frozen models"
python $OB_DIR/export_inference_graph.py --input_type image_tensor --pipeline_config_path $PIPELINE_CONFIGPATH --trained_checkpoint_prefix $TRAIN_DIR/model.ckpt-200000 --output_directory $DATASET
echo "all done"

python demo.py