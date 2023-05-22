python3 train_priv_plain.py -p cifar10_sphinx -s 1 > logs/plain_cifar10_sphinx_split1.log
python3 train_priv_plain.py -p cifar10_sphinx -s 2 > logs/plain_cifar10_sphinx_split2.log
python3 train_priv_plain.py -p cifar10_sphinx -s 3 > logs/plain_cifar10_sphinx_split3.log
python3 train_priv_plain.py -p cifar10_sphinx -s 4 > logs/plain_cifar10_sphinx_split4.log

python3 train_priv_plain.py -p resnet50_classifier -s 1 > logs/plain_resnet50_classifier_split1.log
python3 train_priv_plain.py -p resnet50_classifier -s 2 > logs/plain_resnet50_classifier_split2.log
python3 train_priv_plain.py -p resnet50_classifier -s 3 > logs/plain_resnet50_classifier_split3.log
python3 train_priv_plain.py -p resnet50_classifier -s 4 > logs/plain_resnet50_classifier_split4.log