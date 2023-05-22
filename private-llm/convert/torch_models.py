import torch
import torch.nn
import torchvision
import math

from transformers.models.roberta.modeling_roberta import RobertaLayer
from transformers.models.bert.modeling_bert import BertSelfAttention
import types
from typing import List, Optional, Tuple, Union

import torch.nn as nn

class AlexNetFeatureExtractor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fe = torchvision.models.alexnet(True).features
        self.pool = torch.nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        y = self.pool(self.fe(x))
        return y

class ResNet50FeatureExtractor(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(True)
        self.fe = torch.nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool
        )

    def forward(self, x):
        y = torch.flatten(self.fe(x), 1)
        return y

class DenseNet121FeatureExtractor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        model = torchvision.models.densenet121(True)
        self.fe = model.features

    def forward(self, x):
        y = torch.flatten(self.fe(x), 1)[:, :4096] # 50176 dims total. 
        return y


def load_model(preset: str):
    ret = None
    if preset == "alexnet-fe":
        ret = AlexNetFeatureExtractor()
    if preset == "resnet50-fe":
        return ResNet50FeatureExtractor()
    if preset == "densenet121-fe":
        return DenseNet121FeatureExtractor()
    ret.eval()
    return ret

class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropout)

    # mask: floats, directly added to q*k
    def forward(self, qkv, attention_mask):
        # [B, N, E]
        embed_dim = qkv.shape[-1] // 3
        q = qkv[:, :, :embed_dim]
        k = qkv[:, :, embed_dim:2*embed_dim]
        v = qkv[:, :, 2*embed_dim:]
        q = q / math.sqrt(q.shape[-1])
        print(q.shape, k.shape)
        qk = torch.bmm(q, k.transpose(-2, -1))
        qk += attention_mask
        qk = torch.nn.functional.softmax(qk, dim=-1)
        qk = self.dropout_layer(qk)
        qkv = torch.bmm(qk, v)
        return qkv
    
# The original torch.nn.MultiheadAttention require input to have q, k, v 
# and they are projected respectively.
# This edited version accepts x and make q=k=v=x as input, also uses B,N,E instead of original N,B,E
class MultiheadAttention(torch.nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.inner = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.replace_method = "none"

    def set_replace_method(self, method):
        self.replace_method = method

    # mask: bools, with False means unchanged, True means zeroed
    def forward(self, x, attention_mask = None):
        # [B, N, E], attention_mask: [B, N, N]
        # ret = self.inner(x, x, x, attn_mask = attention_mask.repeat_interleave(self.num_heads, dim=0))[0]

        batchsize = x.shape[0]
        seqlen = x.shape[1]

        in_linear = torch.nn.Linear(self.embed_dim, 3 * self.embed_dim)
        in_linear.weight = self.inner.in_proj_weight
        in_linear.bias = self.inner.in_proj_bias
        qkv = in_linear(x)
        q = qkv[:, :, :self.embed_dim]
        k = qkv[:, :, self.embed_dim:2*self.embed_dim]
        v = qkv[:, :, 2*self.embed_dim:]
        head_dimension = self.embed_dim // self.num_heads
        q = q.transpose(0, 1).contiguous().view(seqlen, batchsize * self.num_heads, head_dimension).transpose(0, 1)
        k = k.transpose(0, 1).contiguous().view(seqlen, batchsize * self.num_heads, head_dimension).transpose(0, 1)
        v = v.transpose(0, 1).contiguous().view(seqlen, batchsize * self.num_heads, head_dimension).transpose(0, 1)

        scale = math.sqrt(q.shape[-1])
        q = q / scale
        p = torch.bmm(q, k.transpose(-2, -1))

        mask_repeat = attention_mask.repeat_interleave(self.num_heads, dim=0)
        add_mask = mask_repeat * -10000
        p += add_mask

        if self.replace_method == "none":
            a = torch.nn.functional.softmax(p, dim=-1)
        elif self.replace_method == "relun1":
            a = torch.nn.functional.relu(p)
            a = a / torch.sum(a, dim=-1, keepdim=True)
        else:
            raise Exception("Invalid replace method")
        a = torch.nn.functional.dropout(a, p=self.dropout, training=self.training)
        y = torch.bmm(a, v)

        y = torch.reshape(y.transpose(0, 1), (seqlen, batchsize, self.num_heads * head_dimension)).transpose(0, 1)
        
        out_linear = torch.nn.Linear(self.embed_dim, self.embed_dim)
        out_linear.weight = self.inner.out_proj.weight
        out_linear.bias = self.inner.out_proj.bias
        y = out_linear(y)

        return y
    
class TransformerEncoderLayer(torch.nn.Module):

    def __init__(self, d_model, num_heads, dim_feedforward, activation = "relu", dropout=0.1):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.activation_type = activation

        self.self_attention = MultiheadAttention(d_model, num_heads, dropout)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)

        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.activation = torch.nn.ReLU() if activation == "relu" else torch.nn.GELU()
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.norm2 = torch.nn.LayerNorm(d_model)

        self.replace_method = "none"

    def set_replace_method(self, method:str):
        self.replace_method = method
        self.self_attention.set_replace_method(method)

    def forward(self, x, attention_mask):
        x1 = self.self_attention(x, attention_mask)
        # print("max attention", torch.max(torch.abs(x1)))

        x1 = self.dropout1(x1)
        x1 = x1 + x
        # print("max res1", torch.max(torch.abs(x1)))
        x1 = self.norm1(x1)
        # print("max norm1", torch.max(torch.abs(x1)))

        x2 = self.linear1(x1)
        # print("max linear1", torch.max(torch.abs(x2)))
        x2 = self.linear2(self.activation(x2))
        # print("max linear2", torch.max(torch.abs(x2)))
        x2 = self.dropout2(x2)
        x2 = x2 + x1
        # print("max res2", torch.max(torch.abs(x2)))
        x2 = self.norm2(x2)
        # print("max norm2", torch.max(torch.abs(x2)))

        return x2
    
class TransformerEncoderStack(torch.nn.Module):

    def __init__(self, layer_count, d_model, num_heads, dim_feedforward, activation = "relu", dropout=0.1):
        super().__init__()
        self.layer_count = layer_count
        self.embed_dim = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.activation = activation

        self.layers = torch.nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, activation, dropout) 
            for _ in range(layer_count)])

    def forward(self, x, attention_mask):
        # B, N, E; attention_mask: B, N
        if attention_mask is None:
            seqlen = x.shape[1]
            attention_mask = torch.zeros((seqlen, seqlen), dtype=torch.bool)
        else: 
            assert(len(attention_mask.shape) == 2)
            seqlen = x.shape[1]
            assert(attention_mask.shape[1] == seqlen)
            attention_mask = torch.where(attention_mask > 0, False, True)[:, None, :].repeat((1, seqlen, 1))

        for layer in self.layers:
            x = layer(x, attention_mask)
        
        return x
    
class BertWithoutEmbedding(torch.nn.Module):

    def __init__(self, stack, pooler, classifier):
        super().__init__()
        self.stack = stack
        self.pooler = pooler
        self.classifier = classifier

    def forward(self, x, attention_mask):
        x = self.stack(x, attention_mask)
        if self.pooler is not None:
            x = self.pooler(x)
        x = self.classifier(x)
        return x

class Residual(torch.nn.Module):
    
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)
    
class Truncate(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class TorchNative(torch.nn.Module):
    
    def __init__(self, preset):
        super().__init__()
        self.preset = preset
        self.model = load_model(self.preset)
    
    def forward(self, x):
        return self.model(x)
    
class BertPooler(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        return hidden_states[:, 0]

class TakeCLSToken(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features):
        return features[:, 0, :]  # take <s> token (equiv. to [CLS])
    
class LayerNormSubstitute(torch.nn.Module):
    # Create gamma and beta as [1, 1, E] but do not compute statistics
    def __init__(self, embed_dim):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(1, 1, embed_dim))
        self.beta = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        x = x - mean
        return self.gamma * x + self.beta
    
class Centralize(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        x = x - mean
        return x

class BertSelfAttentionSubstitute(torch.nn.Module):
    def __init__(self, copy_from: BertSelfAttention, replace_method: str = 'relusq'):
        super().__init__()

        self.num_attention_heads = copy_from.num_attention_heads
        self.attention_head_size = copy_from.attention_head_size
        self.all_head_size = copy_from.all_head_size

        self.query = nn.Linear(copy_from.query.in_features, self.all_head_size)
        self.key = nn.Linear(copy_from.key.in_features, self.all_head_size)
        self.value = nn.Linear(copy_from.value.in_features, self.all_head_size)

        # copy weights and bias
        self.query.weight.data = copy_from.query.weight.data
        self.query.bias.data = copy_from.query.bias.data
        self.key.weight.data = copy_from.key.weight.data
        self.key.bias.data = copy_from.key.bias.data
        self.value.weight.data = copy_from.value.weight.data
        self.value.bias.data = copy_from.value.bias.data

        self.dropout = nn.Dropout(copy_from.dropout.p)
        self.position_embedding_type = copy_from.position_embedding_type
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            raise NotImplementedError
            self.max_position_embeddings = copy_from.max_position_embeddings
            self.distance_embedding = copy_from.distance_embedding

        self.is_decoder = copy_from.is_decoder

        self.replace_method = replace_method
        if self.replace_method == "reluaffine":
            self.softmax_replaced = nn.Linear(1, 1)

        

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        if self.replace_method == "none":
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        elif self.replace_method == "relu":
            attention_probs = nn.functional.relu(attention_scores)
        elif self.replace_method == "relun1":
            attention_probs = nn.functional.relu(attention_scores)
            attention_probs = attention_probs / (attention_probs.sum(dim=-1, keepdim=True) + 1e-6)
        elif self.replace_method == "relusq":
            attention_probs = nn.functional.relu(attention_scores) ** 2
        elif self.replace_method == "relusq_n1":
            attention_probs = nn.functional.relu(attention_scores) ** 2
            attention_probs = attention_probs / (attention_probs.sum(dim=-1, keepdim=True) + 1e-6)
        elif self.replace_method == "reluaffine":
            attention_probs = nn.functional.relu(attention_scores)
            attention_probs = attention_probs.view(*attention_probs.shape, 1)
            attention_probs = self.softmax_replaced(attention_probs)
            attention_probs = attention_probs.view(*attention_scores.shape)
        else:
            raise ValueError("Unknown replace method: {}".format(self.replace_method))

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
    

if __name__ == "__main__":

    import types
    config = types.SimpleNamespace()
    config.num_attention_heads = 2
    config.hidden_size = 128
    config.chunk_size_feed_forward = 1
    config.attention_probs_dropout_prob = 0
    config.is_decoder = False
    config.layer_norm_eps = 1e-5
    config.hidden_dropout_prob = 0
    config.add_cross_attention = False
    config.intermediate_size = 512
    config.hidden_act = "gelu"

    roberta_layer = RobertaLayer(config)
    l = TransformerEncoderLayer(128, 2, 512, "gelu", 0)
    stack = TransformerEncoderStack(1, 128, 2, 512, "gelu", 0)
    stack.layers[0] = l

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

    l.linear1 = roberta_layer.intermediate.dense
    act_fn = roberta_layer.intermediate.intermediate_act_fn
    if "relu" in str(type(act_fn)).lower():
        act_fn = torch.nn.ReLU()
    elif "gelu" in str(type(act_fn)).lower():
        act_fn = torch.nn.GELU()
    l.activation = act_fn
    l.linear2 = roberta_layer.output.dense
    l.dropout2 = roberta_layer.output.dropout
    l.norm2 = roberta_layer.output.LayerNorm

    l.eval()
    roberta_layer.eval()
    x = torch.randn((2, 4, 128))
    attention_mask = torch.tensor(
        [
            [1,1,1,0],
            [1,1,0,0]
        ]
    )
    float_attention_mask = torch.where(attention_mask > 0, 0, -10000)[:, None, None, :]
    
    print("extended_mask", float_attention_mask)
    yrl = roberta_layer(x, attention_mask=float_attention_mask)[0]
    print("----------------- simulation -------------------")
    yl = stack(x, attention_mask=attention_mask)
    print("------------diff--------------")
    print(yl - yrl)