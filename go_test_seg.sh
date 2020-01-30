#
# Zero-shot Learning
#
task='segmentation'
ignore=255
dataset='pascal'
test_set='seen'
backbone='resnet101'
model='deeplabv3+'
norm='gn16'

#
# Hyper-parameters
#
bs=1

#
# checkpoints
#
resume='./ckpts/exp0.tar'
call='./dataloaders/splits/pascal-ft-common-all.pth'
cseen='./dataloaders/splits/pascal-ft-common-seen.pth'
cunseen='./dataloaders/splits/pascal-ft-common-unseen.pth'

#
# dirs
#
savedir='./prd'

clear
python -W ignore run.py --task $task --ignore $ignore --dataset $dataset --test-set $test_set --backbone $backbone --model $model --batch-size $bs --call $call --cseen $cseen --cunseen $cunseen --save-dir $savedir --test --resume $resume --test-val
