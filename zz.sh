export CUDA_VISIBLE_DEVICES=$1

python -c 'import torch; print(torch.__version__)'

MODE="unsupervised"
DATA="as"
DATA_PATH=data/$DATA
MAX_SEG_LEN="4"
EXTRY="bert_seg"
SEED="42"
NAME="zz_test"


MODEL_PATH=models_$EXTRY/$MODE-$DATA-$MAX_SEG_LEN

if [ ! -z "$NAME" ];
then
MODEL_PATH=models_"$EXTRY"_"$NAME"/$MODE-$DATA-$MAX_SEG_LEN
fi
echo $NAME
echo $MODEL_PATH


TRAINING_WORDS=$DATA_PATH/words.txt
UNSEGMENT_DATA=$DATA_PATH/unsegmented.txt
SEGMENT_DATA=$DATA_PATH/segmented.txt

TEST_DATA=$DATA_PATH/test.txt
GOLD_TEST=$DATA_PATH/test_gold.txt
TEST_OUTPUT=$MODEL_PATH/prediction.txt
TEST_SCORE=$MODEL_PATH/score.txt

VALID_DATA=$TEST_DATA
GOLD_VALID=$GOLD_TEST
VALID_OUTPUT=$MODEL_PATH/valid_prediction.txt
VALID_SCORE=$MODEL_PATH/valid_score.txt

VALID_OUTPUT_CLS=$MODEL_PATH/valid_prediction_cls.txt
VALID_SCORE_CLS=$MODEL_PATH/valid_score_cls.txt

CONFIG_FILE=models/slm_"$DATA"_"$MAX_SEG_LEN"_config.json
INIT_EMBEDDING_PATH=data/vocab/embedding.npy
VOCAB_FILE=data/vocab/vocab.txt

CONFIG_FILE=models/slm_"$DATA"_"$MAX_SEG_LEN"_config_bert_seg.json
echo $CONFIG_FILE


mkdir -p $MODEL_PATH
rm -rf $MODEL_PATH/*
# cp models/checkpoint $MODEL_PATH

python -u -m codes.run \
    --use_cuda \
    --do_unsupervised \
    --do_valid \
    --do_predict \
    --unsegmented $UNSEGMENT_DATA \
    --predict_input $TEST_DATA \
    --predict_output $TEST_OUTPUT \
    --valid_inputs $VALID_DATA \
    --valid_output $VALID_OUTPUT \
    --vocab_file $VOCAB_FILE \
    --config_file $CONFIG_FILE \
    --save_path "$MODEL_PATH" \
    --sgd_learning_rate 8.0 \
    --adam_learning_rate 0.0005 \
    --warm_up_steps 1000 \
    --train_steps 6000 \
    --unsupervised_batch_size 8000 \
    --predict_batch_size 1000 \
    --valid_batch_size 1000 \
    --segment_token "  " \
    --hug_name "bert-base-chinese" \
    --encoder_mask_type "seg_mask" \
    --seed $SEED

