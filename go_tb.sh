clear

logdir='./run/pascal/'

echo "tensorboard summary @$logdir"

tensorboard --logdir $logdir --port 7541

