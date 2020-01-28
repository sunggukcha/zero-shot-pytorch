#
# Zero-shot Learning
#
task='segmentation'
ignore=255
dataset='pascal'
test_set='unseen'
backbone='resnet101'
model='deeplabv3+'
norm='gn16'

#
# Hyper-parameters
#
lr=0.01
epoch=50
bs=1

#
# checkpoints
#
call='./dataloaders/splits/pascal-ft-common-all.pth'
cseen='./dataloaders/splits/pascal-ft-common-seen.pth'
cunseen='./dataloaders/splits/pascal-ft-common-unseen.pth'

clear
python -W ignore run.py --task $task --ignore $ignore --dataset $dataset --test-set $test_set --backbone $backbone --model $model --lr $lr --epoch $epoch --batch-size $bs --call $call --cseen $cseen --cunseen $cunseen
