This folder contains some utility scripts for substituting BERT-family models with privacy-friendly modules.

Usage: (take roberta-base as an example)

First finetune on a specific dataset

```
python finetune.py \
    --model_name "roberta-base" \
    --task_name "mrpc" \
    --output_dir "roberta-base-mrpc"
```

Then run substitution script:
first substitute relu with gelu

```
python substitute.py \
    --model_name "roberta-base" \
    --task_name "mrpc" \
    --checkpoint_path "./outputs/roberta-base-mrpc/" \
    --layer_adapt_epochs 10 \
    --replace_layer "gelu" \
    --replace_method "relu"
```
The output folder name auto append "replace_layer" and "replace_method"

second, replace softmax with relu+normalization

```
python substitute.py \
    --model_name "roberta-base" \
    --task_name "mrpc" \
    --checkpoint_path "./outputs/roberta-base-mrpc-gelu-relu/" \
    --layer_adapt_epochs 10 \
    --replace_layer "softmax" \
    --replace_method "relun1"
```

third, replace layernorm1 with affine

```
python substitute.py \
    --model_name "roberta-base" \
    --task_name "mrpc" \
    --checkpoint_path "./outputs/roberta-base-mrpc-gelu-relu-softmax-relun1/" \
    --layer_adapt_epochs 10 \
    --replace_layer "ln1" \
    --replace_method "affine"
```

fourth, replace layernorm2 with affine

```
python substitute.py \
    --model_name "roberta-base" \
    --task_name "mrpc" \
    --checkpoint_path "./outputs/roberta-base-mrpc-gelu-relu-softmax-relun1-ln1-affine/" \
    --layer_adapt_epochs 10 \
    --replace_layer "ln2" \
    --replace_method "affine"
```

You can check the output folder's "modification.txt" to see which modifications have been accepted, for example
```
11,gelu,relu
10,gelu,relu
9,gelu,relu
8,gelu,relu
7,gelu,relu
6,gelu,relu
5,gelu,relu
4,gelu,relu
3,gelu,relu
2,gelu,relu
1,gelu,relu
0,gelu,relu
11,softmax,relun1
10,softmax,relun1
9,softmax,relun1
8,softmax,relun1
7,softmax,relun1
6,softmax,relun1
4,softmax,relun1
3,softmax,relun1
2,softmax,relun1
0,softmax,relun1
11,ln1,affine
10,ln1,affine
9,ln1,affine
8,ln1,affine
7,ln1,affine
6,ln1,affine
5,ln1,affine
4,ln1,affine
2,ln1,affine
1,ln1,affine
0,ln1,affine
```
