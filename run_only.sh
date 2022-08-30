export CUDA_VISIBLE_DEVICES=$5

python -c 'import torch; print(torch.__version__)'

COMMAND=$1
MODE=$2
DATA=$3
DATA_PATH=data/$DATA
MAX_SEG_LEN=$4
EXTRY=$6
SEED=$7
NAME=$8

MODEL_PATH=models/$MODE-$DATA-$MAX_SEG_LEN
MODEL_PATH=models_"$EXTRY"_"$SEED"/$MODE-$DATA-$MAX_SEG_LEN
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

if [ $EXTRY == "bert" ]
then
CONFIG_FILE=models/slm_"$DATA"_"$MAX_SEG_LEN"_config_bert.json

elif [ $EXTRY == "bert_seg" ]
then
CONFIG_FILE=models/slm_"$DATA"_"$MAX_SEG_LEN"_config_bert_seg.json
fi
echo $CONFIG_FILE


if [ $COMMAND == "train" ] && [ $MODE == "unsupervised" ] && [ $EXTRY == "normal" ]
then
echo "Start Unsupervised Training......"

mkdir -p $MODEL_PATH
rm -rf $MODEL_PATH/*
cp models/checkpoint $MODEL_PATH

python -u -m codes.run \
    --use_cuda \
    --do_unsupervised \
    --do_valid \
    --do_predict \
    --unsegmented $UNSEGMENT_DATA  \
    --predict_input $TEST_DATA \
    --predict_output $TEST_OUTPUT \
    --valid_inputs $VALID_DATA \
    --valid_output $VALID_OUTPUT \
    --vocab_file $VOCAB_FILE \
    --config_file $CONFIG_FILE \
    --init_embedding_path $INIT_EMBEDDING_PATH \
    --save_path "$MODEL_PATH" \
    --sgd_learning_rate 16.0 \
    --adam_learning_rate 0.0005 \
    --warm_up_steps 800 \
    --train_steps 6000 \
    --unsupervised_batch_size 16000 \
    --predict_batch_size 500 \
    --valid_batch_size 500 \
    --segment_token "  " \
    --seed $SEED


rm $MODEL_PATH/checkpoint

elif [ $COMMAND == "train" ] && [ $MODE == "unsupervised" ] && [ $EXTRY == "gpt" ]
then
echo "Start Unsupervised Training......"

mkdir -p $MODEL_PATH
rm -rf $MODEL_PATH/*
# cp models/checkpoint $MODEL_PATH

CONFIG_FILE=models/slm_"$MAX_SEG_LEN"_config_gpt2.json

python -u -m codes.run \
    --use_cuda \
    --do_unsupervised \
    --do_valid \
    --do_predict \
    --unsegmented $UNSEGMENT_DATA  \
    --predict_input $TEST_DATA \
    --predict_output $TEST_OUTPUT \
    --valid_inputs $VALID_DATA \
    --valid_output $VALID_OUTPUT \
    --vocab_file $VOCAB_FILE \
    --config_file $CONFIG_FILE \
    --save_path "$MODEL_PATH" \
    --sgd_learning_rate 16.0 \
    --adam_learning_rate 0.0005 \
    --warm_up_steps 0 \
    --train_steps 6000 \
    --unsupervised_batch_size 8000 \
    --predict_batch_size 500 \
    --valid_batch_size 500 \
    --segment_token "  " \
    --hug_name "ckiplab/gpt2-base-chinese" \
    --seed $SEED


elif [ $COMMAND == "train" ] && [ $MODE == "unsupervised" ] && [ $EXTRY == "bert" ]
then
echo "Start Unsupervised Training......"


CONFIG_FILE=models/slm_"$MAX_SEG_LEN"_config_bert.json

mkdir -p $MODEL_PATH
rm -rf $MODEL_PATH/*
# cp models/checkpoint $MODEL_PATH

python -u -m codes.run \
    --use_cuda \
    --do_unsupervised \
    --do_valid \
    --do_predict \
    --unsegmented $UNSEGMENT_DATA  \
    --predict_input $TEST_DATA \
    --predict_output $TEST_OUTPUT \
    --valid_inputs $VALID_DATA \
    --valid_output $VALID_OUTPUT \
    --vocab_file $VOCAB_FILE \
    --config_file $CONFIG_FILE \
    --save_path "$MODEL_PATH" \
    --sgd_learning_rate 16.0 \
    --adam_learning_rate 0.0005 \
    --warm_up_steps 800 \
    --train_steps 6000 \
    --unsupervised_batch_size 8000 \
    --predict_batch_size 2000 \
    --valid_batch_size 2000 \
    --segment_token "  " \
    --hug_name "bert-base-chinese" \
    --seed $SEED

elif [ $COMMAND == "train" ] && [ $MODE == "unsupervised" ] && [ $EXTRY == "bert_impact" ]
then
echo "Start Unsupervised Training......"

CONFIG_FILE=models/slm_"$MAX_SEG_LEN"_config_bert.json

mkdir -p $MODEL_PATH
rm -rf $MODEL_PATH/*

python -u -m codes.run \
    --use_cuda \
    --do_unsupervised \
    --do_valid \
    --do_predict \
    --unsegmented $UNSEGMENT_DATA  \
    --predict_input $TEST_DATA \
    --predict_output $TEST_OUTPUT \
    --valid_inputs $VALID_DATA \
    --valid_output $VALID_OUTPUT \
    --vocab_file $VOCAB_FILE \
    --config_file $CONFIG_FILE \
    --save_path "$MODEL_PATH" \
    --sgd_learning_rate 16.0 \
    --adam_learning_rate 0.0005 \
    --warm_up_steps 800 \
    --train_steps 6000 \
    --unsupervised_batch_size 8000 \
    --predict_batch_size 2000 \
    --valid_batch_size 2000 \
    --segment_token "  " \
    --hug_name "bert-base-chinese" \
    --is_impacted \
    --seed $SEED

elif [ $COMMAND == "train" ] && [ $MODE == "unsupervised" ] && [ $EXTRY == "bert_impact_mlm" ]
then
echo "Start Unsupervised Training......"

CONFIG_FILE=models/slm_"$MAX_SEG_LEN"_config_bert.json

mkdir -p $MODEL_PATH
rm -rf $MODEL_PATH/*

python -u -m codes.run \
    --use_cuda \
    --do_unsupervised \
    --do_valid \
    --do_predict \
    --unsegmented $UNSEGMENT_DATA  \
    --predict_input $TEST_DATA \
    --predict_output $TEST_OUTPUT \
    --valid_inputs $VALID_DATA \
    --valid_output $VALID_OUTPUT \
    --vocab_file $VOCAB_FILE \
    --config_file $CONFIG_FILE \
    --save_path "$MODEL_PATH" \
    --sgd_learning_rate 16.0 \
    --adam_learning_rate 0.0005 \
    --warm_up_steps 800 \
    --train_steps 6000 \
    --unsupervised_batch_size 8000 \
    --predict_batch_size 2000 \
    --valid_batch_size 2000 \
    --segment_token "  " \
    --hug_name "bert-base-chinese" \
    --is_impacted \
    --upper_bound 10 \
    --do_masked_lm \
    --mlm_train_steps 0 \
    --mask_ratio 0.15 \
    --seed $SEED

elif [ $COMMAND == "train" ] && [ $MODE == "unsupervised" ] && [ $EXTRY == "bert_no_single" ]
then
echo "Start Unsupervised Training......"


CONFIG_FILE=models/slm_"$MAX_SEG_LEN"_config_bert.json

mkdir -p $MODEL_PATH
rm -rf $MODEL_PATH/*
# cp models/checkpoint $MODEL_PATH

python -u -m codes.run \
    --use_cuda \
    --do_unsupervised \
    --do_valid \
    --do_predict \
    --unsegmented $UNSEGMENT_DATA  \
    --predict_input $TEST_DATA \
    --predict_output $TEST_OUTPUT \
    --valid_inputs $VALID_DATA \
    --valid_output $VALID_OUTPUT \
    --vocab_file $VOCAB_FILE \
    --config_file $CONFIG_FILE \
    --save_path "$MODEL_PATH" \
    --sgd_learning_rate 16.0 \
    --adam_learning_rate 0.0005 \
    --warm_up_steps 800 \
    --train_steps 6000 \
    --unsupervised_batch_size 2000 \
    --predict_batch_size 2000 \
    --valid_batch_size 2000 \
    --segment_token "  " \
    --hug_name "bert-base-chinese" \
    --no_single \
    --seed $SEED


# rm $MODEL_PATH/checkpoint

elif [ $COMMAND == "train" ] && [ $MODE == "unsupervised" ] && [ $EXTRY == "bert_only_test" ]
then
echo "Start Unsupervised Training......"


CONFIG_FILE=models/slm_"$MAX_SEG_LEN"_config_bert.json

mkdir -p $MODEL_PATH
rm -rf $MODEL_PATH/*
# cp models/checkpoint $MODEL_PATH

python -u -m codes.run \
    --use_cuda \
    --do_unsupervised \
    --do_valid \
    --do_predict \
    --unsegmented $TEST_DATA  \
    --predict_input $TEST_DATA \
    --predict_output $TEST_OUTPUT \
    --valid_inputs $VALID_DATA \
    --valid_output $VALID_OUTPUT \
    --vocab_file $VOCAB_FILE \
    --config_file $CONFIG_FILE \
    --save_path "$MODEL_PATH" \
    --sgd_learning_rate 16.0 \
    --adam_learning_rate 0.0005 \
    --warm_up_steps 800 \
    --train_steps 6000 \
    --unsupervised_batch_size 8000 \
    --predict_batch_size 2000 \
    --valid_batch_size 2000 \
    --segment_token "  " \
    --hug_name "bert-base-chinese" \
    --seed $SEED


# rm $MODEL_PATH/checkpoint

elif [ $COMMAND == "train" ] && [ $MODE == "unsupervised" ] && [ $EXTRY == "bert_seg" ]
then
echo "Start Unsupervised Training......"

mkdir -p $MODEL_PATH
rm -rf $MODEL_PATH/*
# cp models/checkpoint $MODEL_PATH

python -u -m codes.run \
    --use_cuda \
    --do_unsupervised \
    --do_valid \
    --do_predict \
    --unsegmented $UNSEGMENT_DATA  \
    --predict_input $TEST_DATA \
    --predict_output $TEST_OUTPUT \
    --valid_inputs $VALID_DATA \
    --valid_output $VALID_OUTPUT \
    --vocab_file $VOCAB_FILE \
    --config_file $CONFIG_FILE \
    --save_path "$MODEL_PATH" \
    --sgd_learning_rate 16.0 \
    --adam_learning_rate 0.0005 \
    --warm_up_steps 1000 \
    --train_steps 6000 \
    --unsupervised_batch_size 8000 \
    --predict_batch_size 2000 \
    --valid_batch_size 2000 \
    --segment_token "  " \
    --hug_name "bert-base-chinese" \
    --encoder_mask_type "seg_mask" \
    --seed $SEED


# rm $MODEL_PATH/checkpoint


elif [ $COMMAND == "train" ] && [ $MODE == "unsupervised" ] && [ $EXTRY == "bert_mlm" ]
then
echo "Start Unsupervised Training......"

mkdir -p $MODEL_PATH
rm -rf $MODEL_PATH/*
# cp models/checkpoint $MODEL_PATH

CONFIG_FILE=models/slm_"$DATA"_"$MAX_SEG_LEN"_config_bert.json

python -u -m codes.run_mlm \
    --use_cuda \
    --do_unsupervised \
    --do_valid \
    --do_predict \
    --unsegmented $UNSEGMENT_DATA  \
    --predict_input $TEST_DATA \
    --predict_output $TEST_OUTPUT \
    --valid_inputs $VALID_DATA \
    --valid_output $VALID_OUTPUT \
    --vocab_file $VOCAB_FILE \
    --config_file $CONFIG_FILE \
    --save_path "$MODEL_PATH" \
    --sgd_learning_rate 16.0 \
    --adam_learning_rate 0.0005 \
    --warm_up_steps 800 \
    --train_steps 6000 \
    --unsupervised_batch_size 8000 \
    --predict_batch_size 2000 \
    --valid_batch_size 2000 \
    --segment_token "  " \
    --hug_name "bert-base-chinese" \
    --do_masked_lm \
    --mlm_train_steps 500 \
    --mask_ratio 0.15 \
    --seed $SEED

# rm $MODEL_PATH/checkpoint


elif [ $COMMAND == "train" ] && [ $MODE == "unsupervised" ] && [ $EXTRY == "bert_seg_mlm" ]
then
echo "Start Unsupervised Training......"

mkdir -p $MODEL_PATH
rm -rf $MODEL_PATH/*
# cp models/checkpoint $MODEL_PATH

CONFIG_FILE=models/slm_"$DATA"_"$MAX_SEG_LEN"_config_bert_seg.json

python -u -m codes.run_mlm \
    --use_cuda \
    --do_unsupervised \
    --do_valid \
    --do_predict \
    --unsegmented $UNSEGMENT_DATA  \
    --predict_input $TEST_DATA \
    --predict_output $TEST_OUTPUT \
    --valid_inputs $VALID_DATA \
    --valid_output $VALID_OUTPUT \
    --vocab_file $VOCAB_FILE \
    --config_file $CONFIG_FILE \
    --save_path "$MODEL_PATH" \
    --sgd_learning_rate 16.0 \
    --adam_learning_rate 0.0005 \
    --warm_up_steps 1000 \
    --train_steps 6000 \
    --unsupervised_batch_size 8000 \
    --predict_batch_size 500 \
    --valid_batch_size 1000 \
    --segment_token "  " \
    --hug_name "bert-base-chinese" \
    --encoder_mask_type "seg_mask" \
    --do_masked_lm \
    --mlm_train_steps 500 \
    --mask_ratio 0.15 \
    --seed $SEED


# rm $MODEL_PATH/checkpoint


elif [ $COMMAND == "train" ] && [ $MODE == "unsupervised" ] && [ $EXTRY == "classifier" ]
then
echo "Start Unsupervised Training......"

mkdir -p $MODEL_PATH
rm -rf $MODEL_PATH/*
# cp models/checkpoint $MODEL_PATH

python -u -m codes.run \
    --use_cuda \
    --do_unsupervised \
    --do_valid \
    --do_predict \
    --unsegmented $UNSEGMENT_DATA  \
    --predict_input $TEST_DATA \
    --predict_output $TEST_OUTPUT \
    --valid_inputs $VALID_DATA \
    --valid_output $VALID_OUTPUT \
    --vocab_file $VOCAB_FILE \
    --config_file $CONFIG_FILE \
    --init_embedding_path $INIT_EMBEDDING_PATH \
    --save_path "$MODEL_PATH" \
    --sgd_learning_rate 16.0 \
    --adam_learning_rate 0.005 \
    --warm_up_steps 800 \
    --train_steps 4000 \
    --unsupervised_batch_size 16000 \
    --predict_batch_size 500 \
    --valid_batch_size 500 \
    --segment_token "  " \
    --cls_train_steps 4000 \
    --do_classifier \
    --seed $SEED


# rm $MODEL_PATH/checkpoint

elif [ $COMMAND == "train" ] && [ $MODE == "unsupervised" ] && [ $EXTRY == "iterative" ]
then
echo "Start Iterative Training......"

# rm -rf $MODEL_PATH/*
cp models/checkpoint $MODEL_PATH

python -u -m codes.run \
    --use_cuda \
    --do_unsupervised \
    --do_valid \
    --do_predict \
    --unsegmented $UNSEGMENT_DATA  \
    --predict_input $TEST_DATA \
    --predict_output $TEST_OUTPUT \
    --valid_inputs $VALID_DATA \
    --valid_output $VALID_OUTPUT \
    --vocab_file $VOCAB_FILE \
    --config_file $CONFIG_FILE \
    --init_embedding_path $INIT_EMBEDDING_PATH \
    --save_path "$MODEL_PATH" \
    --sgd_learning_rate 16.0 \
    --adam_learning_rate 0.005 \
    --warm_up_steps 800 \
    --train_steps 8000 \
    --unsupervised_batch_size 16000 \
    --predict_batch_size 500 \
    --valid_batch_size 500 \
    --segment_token "  " \
    --cls_train_steps 4000 \
    --iterative_train_steps 4000 \
    --iterative_train \
    --seed $SEED

# rm $MODEL_PATH/checkpoint

elif [ $COMMAND == "train" ] && [ $MODE == "unsupervised" ] && [ $EXTRY == "iterative_lm" ]
then
echo "Start Iterative Training......"

# rm -rf $MODEL_PATH/*
cp models/checkpoint $MODEL_PATH

python -u -m codes.run_lm \
    --use_cuda \
    --do_unsupervised \
    --do_valid \
    --do_predict \
    --unsegmented $UNSEGMENT_DATA  \
    --predict_input $TEST_DATA \
    --predict_output $TEST_OUTPUT \
    --valid_inputs $VALID_DATA \
    --valid_output $VALID_OUTPUT \
    --vocab_file $VOCAB_FILE \
    --config_file $CONFIG_FILE \
    --init_embedding_path $INIT_EMBEDDING_PATH \
    --save_path "$MODEL_PATH" \
    --sgd_learning_rate 16.0 \
    --adam_learning_rate 0.005 \
    --warm_up_steps 800 \
    --train_steps 8000 \
    --unsupervised_batch_size 16000 \
    --predict_batch_size 500 \
    --valid_batch_size 500 \
    --segment_token "  " \
    --cls_train_steps 4000 \
    --iterative_train_steps 4000 \
    --iterative_train \
    --label_smoothing 0.0 \
    --lm_train_steps 2000 \
    --seed $SEED

# rm $MODEL_PATH/checkpoint

elif [ $COMMAND == "train" ] && [ $MODE == "unsupervised" ] && [ $EXTRY == "circular" ]
then
echo "Start Circular Training......"

mkdir -p $MODEL_PATH
rm -rf $MODEL_PATH/*
cp models/checkpoint $MODEL_PATH

python -u -m codes.run_circular \
    --use_cuda \
    --do_unsupervised \
    --do_valid \
    --do_predict \
    --unsegmented $UNSEGMENT_DATA  \
    --predict_input $TEST_DATA \
    --predict_output $TEST_OUTPUT \
    --valid_inputs $VALID_DATA \
    --valid_output $VALID_OUTPUT \
    --vocab_file $VOCAB_FILE \
    --config_file $CONFIG_FILE \
    --init_embedding_path $INIT_EMBEDDING_PATH \
    --save_path "$MODEL_PATH" \
    --sgd_learning_rate 16.0 \
    --adam_learning_rate 0.005 \
    --warm_up_steps 800 \
    --train_steps 10000 \
    --unsupervised_batch_size 16000 \
    --predict_batch_size 500 \
    --valid_batch_size 500 \
    --segment_token "  " \
    --cls_train_steps 4000 \
    --circular_train_steps 7000 \
    --label_smoothing 0.0 \
    --seed $SEED


rm $MODEL_PATH/checkpoint

elif [ $COMMAND == "train" ] && [ $MODE == "unsupervised" ] && [ $EXTRY == "cls" ]
then
echo "Start Unsupervised Training......"

# rm -rf $MODEL_PATH/*
cp models/checkpoint $MODEL_PATH

python -u codes/run_cls.py \
    --use_cuda \
    --do_unsupervised \
    --do_valid \
    --do_predict \
    --unsegmented $UNSEGMENT_DATA  \
    --predict_input $TEST_DATA \
    --predict_output $TEST_OUTPUT \
    --valid_inputs $VALID_DATA \
    --valid_output $VALID_OUTPUT \
    --vocab_file $VOCAB_FILE \
    --config_file $CONFIG_FILE \
    --init_embedding_path $INIT_EMBEDDING_PATH \
    --save_path "$MODEL_PATH" \
    --sgd_learning_rate 16.0 \
    --adam_learning_rate 0.005 \
    --warm_up_steps 800 \
    --train_steps 4000 \
    --unsupervised_batch_size 16000 \
    --predict_batch_size 500 \
    --valid_batch_size 500 \
    --segment_token "  " \
    --cls_train_steps 4000 \
    --do_classifier


rm $MODEL_PATH/checkpoint

elif [ $COMMAND == "train" ] && [ $MODE == "supervised" ]
then
echo "Start Supervised Training......"
python -u codes/run.py \
    --use_cuda \
    --do_supervised \
    --do_valid \
    --do_predict \
    --segmented $SEGMENT_DATA \
    --predict_input $TEST_DATA \
    --predict_output $TEST_OUTPUT \
    --valid_inputs $VALID_DATA \
    --valid_output $VALID_OUTPUT \
    --vocab_file $VOCAB_FILE \
    --config_file $CONFIG_FILE \
    --init_embedding_path $INIT_EMBEDDING_PATH \
    --save_path "$MODEL_PATH" \
    --sgd_learning_rate 4.0 \
    --adam_learning_rate 0.005 \
    --warm_up_steps 800 \
    --train_steps 4000 \
    --supervised_batch_size 16000 \
    --predict_batch_size 500 \
    --valid_batch_size 500 \
    --segment_token "  "

rm $MODEL_PATH/checkpoint

elif [ $COMMAND == "predict" ]
then
echo "Start Predicting......"
python -u codes/run.py \
    --use_cuda \
    --do_predict \
    --predict_input $TEST_DATA \
    --predict_output $TEST_OUTPUT \
    --vocab_file $VOCAB_FILE \
    --config_file $CONFIG_FILE \
    --init_checkpoint "$MODEL_PATH" \
    --predict_batch_size 500 \
    --segment_token "  "

elif [ $COMMAND == "valid" ]
then

perl data/score.pl $TRAINING_WORDS $GOLD_VALID $VALID_OUTPUT > $VALID_SCORE

tail -14 $VALID_SCORE | head -13

echo "Examples:"

head -10 $VALID_OUTPUT


elif [ $COMMAND == "valid_cls" ]
then

perl data/score.pl $TRAINING_WORDS $GOLD_VALID $VALID_OUTPUT_CLS > $VALID_SCORE_CLS

tail -14 $VALID_SCORE_CLS | head -13

echo "Examples:"

head -10 $VALID_OUTPUT_CLS


elif [ $COMMAND == "eval" ]
then

perl data/score.pl $TRAINING_WORDS $GOLD_TEST $TEST_OUTPUT > $TEST_SCORE

tail -14 $TEST_SCORE | head -13

echo "Examples:"

head -10 $TEST_OUTPUT

fi
