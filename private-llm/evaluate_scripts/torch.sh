python3 train_torch.py -p mnist_aby3      > logs/torch_mnist_aby3.log
python3 train_torch.py -p mnist_chameleon > logs/torch_mnist_chameleon.log
python3 train_torch.py -p cifar10_lenet5  > logs/torch_cifar10_lenet5.log
python3 train_torch.py -p cifar10_sphinx  > logs/torch_cifar10_sphinx.log

python3 tmp_resnet50_train.py -p resnet50 > logs/torch_resnet50_classifier.log
python3 tmp_resnet50_train.py -p alexnet  > logs/torch_alexnet_classifier.log