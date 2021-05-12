#!/bin/bash
set -x
set -e

MODE=$1

OUT_PATH="../data/avenue"
N_OBJ=10 # number of object categories
N_REL=10 # number of relationship categories
H5=avenue_${MODE}_-SGG.h5
JSON=avenue_${MODE}_-SGG-dicts.json
FRAC=1

IMDB=/mnt/sda/sc/prepared_code/scene-graph-TF-release-master/scene-graph-TF-release-master/data/avenue/avenue_${MODE}_imdb_1800.h5
OBJ=/mnt/sda/sc/prepared_code/scene-graph-TF-release-master/scene-graph-TF-release-master/data/avenue/avenue_video_${MODE}_objects.json
REL=/mnt/sda/sc/prepared_code/scene-graph-TF-release-master/scene-graph-TF-release-master/data/avenue/avenue_video_${MODE}_relationships.json
META=/mnt/sda/sc/prepared_code/scene-graph-TF-release-master/scene-graph-TF-release-master/data/avenue/avenue_video_${MODE}_image_data.json
python avenue_to_roidb.py \
    --imdb $IMDB \
    --mode $MODE \
    --json_file $OUT_PATH/$JSON \
    --h5_file $OUT_PATH/$H5 \
    --load_frac $FRAC \
    --num_objects $N_OBJ \
    --num_predicates $N_REL \
    --object_input $OBJ \
    --relationship_input $REL \
    --metadata_input $META