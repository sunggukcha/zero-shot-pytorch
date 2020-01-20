#
# Zero-shot Learning
#
task='segmentation'
ignore=255
dataset='pascal'
test_set='unseen'
backbone='resnet101'
model='deeplabv2'
norm='gn16'

#
# Hyper-parameters
#
lr=0.01
epoch=50
bs=1

#
#
#
python -W ignore run.py --task $task --ignore $ignore --dataset $dataset --test-set $test_set --backbone $backbone --model $model --lr $lr --epoch $epoch --batch-size $bs
