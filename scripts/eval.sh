#!/usr/bin/env bash
echo "Dataset:" $1  "Backbone:" $2  "GPU index:" $3 "Tag:" $4 "logits_coeffs" $5

# ####################
# # 初始化：Way & Shot
# ####################
way=5
shot1=1
shot2=5
# #####################
python exp2_test_few_shot.py --way $way --shot $shot1 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos'  --feat_source_list='1,2' --branch_list='1,2'
python exp2_test_few_shot.py --way $way --shot $shot2 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos'  --feat_source_list='1,2' --branch_list='1,2'




# # ######################
# # Multi-way & Multi-shot
# # ######################

# #1.Multi-shot
# way=5
# python exp2_test_few_shot.py --way $way --shot 1 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos'  --feat_source_list='1,2' --branch_list='1,2'
# python exp2_test_few_shot.py --way $way --shot 5 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos'  --feat_source_list='1,2' --branch_list='1,2'

# way=5
# python exp2_test_few_shot.py --way $way --shot 10 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos'  --feat_source_list='1,2' --branch_list='1,2'
# python exp2_test_few_shot.py --way $way --shot 15 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos'  --feat_source_list='1,2' --branch_list='1,2'

# way=5
# python exp2_test_few_shot.py --way $way --shot 20 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos'  --feat_source_list='1,2' --branch_list='1,2'
# python exp2_test_few_shot.py --way $way --shot 25 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos'  --feat_source_list='1,2' --branch_list='1,2'


# #2.Multi-way
# way=5
# python exp2_test_few_shot.py --way $way --shot 1 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos'  --feat_source_list='1,2' --branch_list='1,2'

# way=6
# python exp2_test_few_shot.py --way $way --shot 1 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos'  --feat_source_list='1,2' --branch_list='1,2'

# way=7
# python exp2_test_few_shot.py --way $way --shot 1 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos'  --feat_source_list='1,2' --branch_list='1,2'

# way=8
# python exp2_test_few_shot.py --way $way --shot 1 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos'  --feat_source_list='1,2' --branch_list='1,2'

# way=9
# python exp2_test_few_shot.py --way $way --shot 1 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos'  --feat_source_list='1,2' --branch_list='1,2'

# way=10
# python exp2_test_few_shot.py --way $way --shot 1 --gpu $3 --config ./configs/$1/test_few_shot_$1.yaml --load=./checkpoints/$2-$1_$4/max-f-va.pth --method='cos'  --feat_source_list='1,2' --branch_list='1,2'
