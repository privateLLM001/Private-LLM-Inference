# for DP_LEVEL in 0.001 0.005
# do
#     python3 train_torch.py -p cifar10_sphinx -n ${DP_LEVEL} -C 8 > logs/torch_cifar10_sphinx_dp${DP_LEVEL}.log
# done

for DP_LEVEL in 0.005 0.010 0.020 0.050 0.100
do
    python3 tmp_resnet50_train.py -p resnet50 -n ${DP_LEVEL} -C 8 > logs/torch_resnet50_classifier_dp${DP_LEVEL}.log
done