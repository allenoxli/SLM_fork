
export CUDA_VISIBLE_DEVICES=$1

python -c 'import torch; print(torch.__version__)'

MODE="unsupervised"
DATA="msr"
MAX_SEG_LEN="4"
EXTRY="gpt"
SEED="42"
NAME="zz_test"

DATA_PATH=data/$DATA


warm_up_steps_list=('800 1500')
lr_list=("5e-4 1e-3 5e-3 8e-3 1e-2")
hug_name='ckiplab/gpt2-base-chinese'


for warm_ratio in $warm_up_steps_list;
do
    for lr_rate in $lr_list;
    do

NAME=warm"$warm_ratio"_lr"$lr_rate"
MODEL_PATH=models_"$EXTRY"_"$NAME"/$MODE-$DATA-$MAX_SEG_LEN

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

CONFIG_FILE=models/slm_"$MAX_SEG_LEN"_config_gpt2.json
if [ $hug_name == 'bert-base-chinese' ]
then
CONFIG_FILE=models/slm_"$DATA"_"$MAX_SEG_LEN"_config_bert_seg.json
fi

echo $CONFIG_FILE


mkdir -p $MODEL_PATH
rm -rf $MODEL_PATH/*

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
    --sgd_learning_rate 16.0 \
    --adam_learning_rate $lr_rate \
    --warm_up_steps $warm_ratio \
    --train_steps 6000 \
    --unsupervised_batch_size 2000 \
    --predict_batch_size 2000 \
    --valid_batch_size 2000 \
    --segment_token "  " \
    --hug_name $hug_name \
    --seed $SEED

    done
done

