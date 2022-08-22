#!/bin/zsh


# exp_list=("38_new_layer_norm_sigmoid_small_hug")


exp_list=("normal_10000step_1234" "normal_10000step_42" "normal_10000step_87" "normal_1234" "normal_42" "normal_8000step_1234" "normal_8000step_42" "normal_8000step_87" "normal_87")
exp_list=("gpt_slm42_800warm_8k_b" "gpt_slm42_no_warm" "gpt_slm87_800warm_8k_b")

exp_list=("bert_mlm_slm42" "bert_seg_slm1234_1kwarm" "bert_seg_slm1234_800warm" "bert_seg_slm42_1kwarm" "bert_seg_slm42_800warm" "bert_seg_slm87_1kwarm" "bert_seg_slm87_800warm" "bert_slm1234_800warm" "bert_slm1234_800warm_8k_b" "bert_slm42_800kwarm" "bert_slm42_800warm" "bert_slm42_800warm_8k_b" "bert_slm87_800warm" "bert_slm87_800warm_8k_b")

bert_seg_mlm_slm1234_real
bert_seg_slm1234_real
bert_slm1234_real

"bert_mlm_slm1234_real"
"bert_mlm_slm42_real"
"bert_mlm_slm87_real"

"bert_seg_mlm_slm42_real"
"bert_seg_mlm_slm87_real"

"bert_seg_slm42_real"
"bert_seg_slm87_real"

"bert_slm42_real"
"bert_slm87_real"

"bert_slm1234_real_2k_b"
"bert_slm42_real_2k_b"
"bert_slm87_real_2k_b"

"bert_slm42_real_4k_b"
"bert_slm42_real_4k_b_no_warm"

for exp in $exp_list;
do
    mkdir log/bert/$exp&rsync -va --exclude='*checkpoint' models_"$exp"/* log/bert/$exp
done


for exp in $exp_list;
do
    rm -rf models_"$exp"
done