MODE=$1
INPUT_IMAGES_DIR=../data/avenue/${MODE}_video_images
INPUR_IMAGES_JSON=../data/avenue/avenue_video_${MODE}_image_data.json
python avenue_to_imdb.py \
    --mode ${MODE}\
    --image_dir ${INPUT_IMAGES_DIR} \
    --image_size 1800 \
    --metadata_input ${INPUR_IMAGES_JSON}