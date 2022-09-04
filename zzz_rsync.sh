#!/bin/bash


# exp_list=("38_new_layer_norm_sigmoid_small_hug")




# exp_list=("bert_warm800_narrow128" "bert_warm800_narrow16" "bert_warm800_narrow256" "bert_warm800_narrow32" "bert_warm800_narrow512" "bert_warm800_narrow64")
# exp_list=("bert_no_single_42")
exp_list=("gpt_warm1500_lr1e-2" "gpt_warm1500_lr1e-3" "gpt_warm1500_lr5e-3" "gpt_warm1500_lr8e-3" "gpt_warm800_lr1e-2" "gpt_warm800_lr1e-3" "gpt_warm800_lr5e-3" "gpt_warm800_lr8e-3")
dir="no_single"

for exp in $exp_list;
do
    mkdir -p log/$dir/$exp&rsync -va --exclude='*checkpoint' models_"$exp"/* log/$dir/$exp
done


for exp in $exp_list;
do
    rm -rf models_"$exp"
done

