set -x
set -e

export PYTHONUNBUFFERED="True"

DATASET=$1
NUM_IM=$2
NET=$3
INFERENCE_ITER=$4
WEIGHT_FN=$5
TEST_MODE=$6
GPU_ID=$7

CFG_FILE=experiments/cfgs/sparse_graph.yml


case $DATASET in
    ucfcrime_test)
        ROIDB=ucfcrime_test_-SGG
        RPNDB=ucfcrime_video_test_proposals
        IMDB=ucfcrime_test_imdb_1800
        ;;
    avenue_test)
        ROIDB=avenue_test_-SGG
        RPNDB=avenue_video_test_proposals
        IMDB=avenue_test_imdb_1800
        ;;
    shanghaitech_test)
        ROIDB=shanghaitech_test_-SGG
        RPNDB=shanghaitech_video_test_proposals
        IMDB=shanghaitech_test_imdb_1800
        ;;
    *)
        echo "Wrong dataset"
        exit
        ;;
esac


export CUDA_VISIBLE_DEVICES=$GPU_ID

time ./tools/test_net.py --gpu $GPU_ID \
  --weights ${WEIGHT_FN} \
  --imdb ${IMDB}.h5 \
  --roidb ${ROIDB} \
  --rpndb ${RPNDB}.h5 \
  --cfg ${CFG_FILE} \
  --network ${NET} \
  --inference_iter ${INFERENCE_ITER} \
  --test_size ${NUM_IM} \
  --test_mode ${TEST_MODE} 
