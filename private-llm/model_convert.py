from collections import OrderedDict
import torch.nn.functional as F
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
import dataloader

VERBOSE = True


def convert_RobertaEmbeddings(roberta_embeddings, depth=0, idx=0):
    if VERBOSE:
        print('{}RobertaEmbeddings is not converted: {}-{}'.format(''.join(['\t']*depth), depth, idx))
    return roberta_embeddings

def convert_RobertaLayer(roberta_layer: RobertaLayer, depth=0, idx=0):
    if VERBOSE:
        print('{}Converting RobertaLayer: {}-{}'.format(''.join(['\t']*depth), depth, idx))
    hidden_size = roberta_layer.attention.self.all_head_size
    num_attention_heads = roberta_layer.attention.self.num_attention_heads
    attention_probs_dropout_prob = roberta_layer.attention.self.dropout.p
    act_fn = roberta_layer.intermediate.intermediate_act_fn
    if "relu" in str(type(act_fn)).lower():
        act_fn = nn.ReLU()
    elif "gelu" in str(type(act_fn)).lower():
        act_fn = nn.GELU()
    else:
        raise Exception(f'act_fn {act_fn} is not implemented!')
    
    l = torch_models.TransformerEncoderLayer(
        hidden_size, num_attention_heads, roberta_layer.intermediate.dense.out_features, 
        "relu" if isinstance(act_fn, nn.ReLU) else "gelu", 
        attention_probs_dropout_prob)
    
    if isinstance(roberta_layer.attention.self, torch_models.BertSelfAttentionSubstitute):
        l.set_replace_method(roberta_layer.attention.self.replace_method)
        if l.replace_method == 'reluaffine':
            # The weights are not copied
            raise Exception('Not implemented!')

    l.self_attention.inner.in_proj_weight.data = torch.cat((
        roberta_layer.attention.self.query.weight,
        roberta_layer.attention.self.key.weight,
        roberta_layer.attention.self.value.weight,
    ), dim=0).clone()
    l.self_attention.inner.in_proj_bias.data = torch.cat((
        roberta_layer.attention.self.query.bias,
        roberta_layer.attention.self.key.bias,
        roberta_layer.attention.self.value.bias,
    ), dim=0).clone()
    l.self_attention.inner.out_proj.weight.data = roberta_layer.attention.output.dense.weight.clone()
    l.self_attention.inner.out_proj.bias.data = roberta_layer.attention.output.dense.bias.clone()

    l.dropout1 = roberta_layer.attention.output.dropout
    l.norm1 = roberta_layer.attention.output.LayerNorm

    # print(l.norm1.weight, l.norm1.bias)

    l.linear1 = roberta_layer.intermediate.dense
    l.activation = act_fn
    l.linear2 = roberta_layer.output.dense
    l.dropout2 = roberta_layer.output.dropout
    l.norm2 = roberta_layer.output.LayerNorm


    convert_roberta_layer = l
    
    # correctness check
    convert_roberta_layer.eval()
    roberta_layer.eval()
    # [B, N, E]
    src = torch.rand(10, 32, roberta_layer.attention.self.all_head_size)
    mask = []
    for i in range(10):
        x = torch.randint(1, 32, (1,))
        mask_single = int(x) * [0,] + (32 - int(x)) * [1,]
        mask.append(mask_single)
    mask = torch.tensor(mask, dtype=torch.bool)
    float_mask = torch.where(mask > 0, -10000, 0)[:, None, None, :]
    mask = mask.reshape((10, 1, 32)).repeat((1, 32, 1))
    out1 = convert_roberta_layer(src, mask)
    out2 = roberta_layer(src, attention_mask=float_mask)[0]
    if torch.max(torch.abs(out1 - out2)) >= 1e-3:
        if VERBOSE:
            print('Failed in converting RobertaLayer: {}-{}!'.format(depth, idx))
    
    return convert_roberta_layer
        
def convert_RobertaEncoder(roberta_encoder, depth=0, idx=0):
    if VERBOSE:
        print('{}Converting RobertaEncoder: {}-{}'.format(''.join(['\t']*depth), depth, idx))
    converted_roberta_layers = []
    for idx_, roberta_layer in enumerate(roberta_encoder.layer):
        converted_roberta_layers.append(convert_RobertaLayer(roberta_layer, depth+1, idx_+1))
    converted_roberta_encoder = torch_models.TransformerEncoderStack(
        len(converted_roberta_layers),
        converted_roberta_layers[0].embed_dim,
        converted_roberta_layers[0].num_heads,
        converted_roberta_layers[0].dim_feedforward,
        converted_roberta_layers[0].activation_type,
        converted_roberta_layers[0].dropout,
    )
    converted_roberta_encoder.layers = torch.nn.ModuleList(converted_roberta_layers)

    # correctness check
    converted_roberta_encoder.eval()
    roberta_encoder.eval()
    src = torch.rand(10, 32, roberta_layer.attention.self.all_head_size)
    mask = []
    for i in range(10):
        x = torch.randint(1, 32, (1,))
        mask_single = int(x) * [1,] + (32 - int(x)) * [0,]
        mask.append(mask_single)
    mask = torch.tensor(mask, dtype=torch.int8)
    float_mask = torch.where(mask > 0, 0, -10000)[:, None, None, :]
    out1 = converted_roberta_encoder(src, mask)
    out2 = roberta_encoder(src, attention_mask=float_mask)[0]
    if torch.max(torch.abs(out1 - out2)) >= 1e-3:
        if VERBOSE:
            print('Failed in converting RobertaEncoder: {}-{}!'.format(depth, idx))
    
    return converted_roberta_encoder
        
def convert_RobertaPooler(roberta_pooler, depth=0, idx=0):
    if VERBOSE:
        print('{}Converting RobertaPooler: {}-{}'.format(''.join(['\t']*depth), depth, idx))
    converted_roberta_pooler = nn.Sequential(
        torch_models.BertPooler(),  # TODO: check this
        roberta_pooler.dense,
        nn.Tanh(),  # TODO: check this
    )

    # correctness check
    src = torch.rand(10, 32, roberta_pooler.dense.in_features)
    out1 = converted_roberta_pooler(src)
    out2 = roberta_pooler(src)
    if torch.max(torch.abs(out1 - out2)) >= 1e-3:
        raise Exception('Failed in converting RobertaPooler: {}-{}!'.format(depth, idx))
    return converted_roberta_pooler

def convert_RobertaModel(roberta_model, depth=0, idx=0):
    if VERBOSE:
        print('{}Converting RobertaModel: {}-{}'.format(''.join(['\t']*depth), depth, idx))
    
    # NOTE: the embeddings is not converted
    roberta_embeddings = convert_RobertaEmbeddings(roberta_model.embeddings, depth+1, 1)
    
    stack = convert_RobertaEncoder(roberta_model.encoder, depth+1, 2)
    pooler = None
    if roberta_model.pooler is not None:
        pooler = convert_RobertaPooler(roberta_model.pooler, depth+1, 3)

    return roberta_embeddings, stack, pooler

def convert_Roberta(roberta, save: str = ""):
    if isinstance(roberta, RobertaModel):
        embeddings, stack, pooler = convert_RobertaModel(roberta)
    elif isinstance(roberta, BertModel):
        embeddings, stack, pooler = convert_RobertaModel(roberta)
    elif isinstance(roberta, (RobertaForSequenceClassification, BertForSequenceClassification)):
        if VERBOSE:
            print('Converting RobertaForSequenceClassification:')
        if isinstance(roberta, RobertaForSequenceClassification):
            embeddings, stack, pooler = convert_RobertaModel(roberta.roberta, 1, 1)
        else:
            embeddings, stack, pooler = convert_RobertaModel(roberta.bert, 1, 1)
        
        if VERBOSE:
            print('\tConverting RobertaClassificationHead: 1-2')
        if isinstance(roberta, RobertaForSequenceClassification):
            classifier = nn.Sequential(
                torch_models.TakeCLSToken(),  # TODO: check this
                roberta.classifier.dropout,
                roberta.classifier.dense,
                nn.Tanh(), # TODO: check this
                roberta.classifier.dropout,
                roberta.classifier.out_proj,
            )
        else:
            classifier = nn.Sequential(
                roberta.dropout,
                roberta.classifier,
            )

        # correctness check
        src = torch.rand(10, 32, roberta.config.hidden_size)
        roberta.eval()
        classifier.eval()
        out1 = classifier(src)
        if isinstance(roberta, RobertaForSequenceClassification):
            out2 = roberta.classifier(src)
        else:
            out2 = roberta.classifier(roberta.dropout(src))
        if torch.max(torch.abs(out1 - out2)) >= 1e-3:
            if VERBOSE:
                print('Failed in converting RobertaClassificationHead: 1-2!')
    else:
        raise Exception('not implemented!')
    
    if save != "":
        torch.save(embeddings, save)
    
    return embeddings, torch_models.BertWithoutEmbedding(stack, pooler, classifier)
    

if __name__ == "__main__":

    # model_name = "roberta-base"
    model_name = "prajjwal1/bert-tiny"

    print("Loading config")

    config = AutoConfig.from_pretrained(model_name)

    print("Loading model")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    print("Converting model")

    embeddings, converted = convert_Roberta(model, False)

    torch.save(embeddings, "logs/embedding.pth")
    embeddings = torch.load("logs/embedding.pth")

    x = torch.randint(0, 30000, (10, 128))
    mask = []
    for i in range(10):
        r = torch.randint(1, 128, (1,))
        mask_single = int(r) * [1,] + (128 - int(r)) * [0,]
        mask.append(mask_single)
    mask = torch.tensor(mask, dtype=torch.int8)

    y1 = model(x, attention_mask=mask).logits
    y2 = converted(embeddings(x), mask)
    
    print(torch.max(torch.abs(y1 - y2)))

    # train_dataset, eval_dataset = dataloader.load_dataset("glue-cola", tokenizer_name="roberta-base")
    # print(eval_dataset[0][0]) # First 0: sample; Second 0: input, rather than label

    # x = torch.reshape(eval_dataset[0][0], (0, -1))
    # y1 = model(x).logits
    # y2 = converted(embeddings(x))
    
    # print(torch.max(torch.abs(y1 - y2)), y1, y2)