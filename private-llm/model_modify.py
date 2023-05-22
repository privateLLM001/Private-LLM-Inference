from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import *
from transformers.models.bert.modeling_bert import *
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification
)
import sys
sys.path.append('..')
import torch_models
import transformers.activations
import types



def get_model_layer_count(bert):
    if isinstance(bert, RobertaForSequenceClassification):
        bert = bert.roberta
    if isinstance(bert, BertForSequenceClassification):
        bert = bert.bert
    bert = bert.encoder
    return len(bert.layer)

def modify_model_layer(bert, layer_id, target_layer, replace_method):
    ''' 
    target_layer = "gelu" or "softmax" or "ln1" or "ln2"
    
    replace_method =
        if target_layer == "gelu": "relu"
        if target_layer == "softmax": "relu", "relusq"
        if target_layer == "ln1" or "ln2": "affine", "1fc", "2fc"

    return trainable params
    '''
    if isinstance(bert, RobertaForSequenceClassification):
        bert = bert.roberta
    if isinstance(bert, BertForSequenceClassification):
        bert = bert.bert
    bert = bert.encoder
    layers = bert.layer
    if layer_id > len(layers):
        raise Exception('layer_id out of range')
    layer = layers[layer_id]
    if target_layer == "gelu":
        if replace_method == "relu":
            layer.intermediate.intermediate_act_fn = nn.ReLU()
            return []
        else:
            raise Exception('Cannot replace GELU with {}'.format(replace_method))
        
    elif target_layer == "softmax":
        layer.attention.self = torch_models.BertSelfAttentionSubstitute(layer.attention.self, replace_method)
        return list(layer.attention.self.parameters())
        
    elif target_layer == "ln1" or target_layer == "ln2":
        target = layer.output.LayerNorm if target_layer == "ln2" else layer.attention.output.LayerNorm
        dims = target.normalized_shape[0]
        if replace_method == "affine":
            sub = torch_models.LayerNormSubstitute(dims)
            sub.to(target.weight.device)
            sub.gamma.data = target.weight.data
            sub.beta.data = target.bias.data
        elif replace_method == "centeraffine":
            sub = torch.nn.Sequential(
                torch_models.Centralize(),
                torch_models.LayerNormSubstitute(dims)
            )
            sub.to(target.weight.device)
        elif replace_method == "1fc":
            sub = torch.nn.Linear(dims, dims)
            sub.to(target.weight.device)
        elif replace_method == "center1fc":
            sub = torch.nn.Sequential(
                torch_models.Centralize(),
                torch.nn.Linear(dims, dims),
            )
            sub.to(target.weight.device)
        elif replace_method == "2fc":
            sub = torch.nn.Sequential(
                torch.nn.Linear(dims, dims),
                torch.nn.ReLU(),
                torch.nn.Linear(dims, dims)
            )
            sub.to(target.weight.device)
        else:
            raise Exception('Cannot replace LayerNorm with {}'.format(replace_method))
        if target_layer == "ln2":
            layer.output.LayerNorm = sub
        else:
            layer.attention.output.LayerNorm = sub
        return list(sub.parameters())
    
    else:
        raise Exception('target_layer not implemented: {}'.format(target_layer))

def modify_model(bert, target_layer, replace_method):
    layer_count = get_model_layer_count(bert)
    params = []
    for i in range(layer_count):
        params += modify_model_layer(bert, i, target_layer, replace_method)
    return params

def modify_model_with_instruction(bert, instruction):
    for layer_id, target_layer, replace_method in instruction:
        modify_model_layer(bert, layer_id, target_layer, replace_method)

def set_trainable_layer(bert, layer_index):
    if isinstance(bert, RobertaForSequenceClassification):
        bert = bert.roberta
    if isinstance(bert, BertForSequenceClassification):
        bert = bert.bert
    bert = bert.encoder
    layers = bert.layer
    if layer_index > len(layers):
        raise Exception('layer_id out of range')
    layer = layers[layer_index]
    for param in layer.parameters():
        param.requires_grad = True

def set_trainable_classifier(bert):
    for param in bert.classifier.parameters():
        param.requires_grad = True

if __name__ == "__main__":
    config = AutoConfig.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', config=config)
    converted = RobertaForSequenceClassification.from_pretrained('roberta-base', config=config)
    modify_relu_next(converted, True, True)

    x = torch.randint(0, 30000, (1, 128))
    y1 = model(x).logits
    y2 = converted(x).logits
    
    print(y1)
    print(y2)

    # config = AutoConfig.from_pretrained('prajjwal1/bert-tiny')
    # model = AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', config=config)
    # converted = AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', config=config)

    # x = torch.randint(0, 30000, (1, 128))
    # y1 = model(x).logits
    # y2 = converted(x).logits
    
    # print(y1)
    # print(y2)