import server_modules
import communication_server
from crypto_switch import cryptography as crypto
import numpy as np
import time
import torch
import torch.nn
import torch.nn.functional
import torch.optim
import optimizer
import model as modelloader
import torch_models
import torchinfo
import transformers
import model_convert
import os
import model_modify

def load_previous_modification(file):
    if not os.path.exists(file):
        return []
    # load from a file, every line has 3 parts: layer_id, replace_layer, replace_method
    with open(file, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split(",") for line in lines]
    lines = [(int(line[0].strip()),line[1].strip(),line[2].strip()) for line in lines]
    return lines

def create_shares(x):
    m = np.max(x)
    r = np.random.uniform(-m, m, x.shape)
    return r, x-r

def absmax(x):
    return np.max(np.abs(x))

def absmean(x):
    return np.mean(np.abs(x))

def to_tensor(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype)

def compare(x, y, prompt="y"):
    y = y.detach().numpy()
    d = x - y
    print("{0:>7}: md {1:.6f}, mv [t {2:>6.3f}, p {3:>6.3f}], ~d {4:.6f}".format(prompt, absmax(d), absmax(y), absmax(x), absmean(d)))

def test_modules():

    comm = communication_server.ServerCommunication()
    comm.listen()
    comm.accept_connection()
    cryp = crypto.EvaluationUtils()
    cryp.receive_keys(comm.recv())

    def share(x, scale):
        x = cryp.to_field(x, scale)
        x_a = cryp.field_random_mask(x.size)
        x_b = cryp.field_add(x, cryp.field_negate(x_a))
        comm.send(x_b)
        return x_a

    SCALE = cryp.default_scale()
    DOUBLE_SCALE = SCALE * SCALE

    start_scale = SCALE
    forward_only = True

    # input_shape = (2, 128, 64 * 3) # [B, N, 3E]
    # model_name, model_torch = "", torch_models.ScaledDotProductAttention()

    # input_shape = (1, 128, 128) # [B, N, E]
    # model_name, model_torch = "", torch_models.MultiheadAttention(128, 2)

    # input_shape = (1, 128, 128) # [B, N, E]
    # model_name, model_torch = "", torch_models.MultiheadAttention(128, 2)
    # model_torch.set_replace_method("relun1")

    # input_shape = (1, 128, 128) # [B, N, E]
    # model_name, model_torch = "", torch_models.TransformerEncoderLayer(128, 2, 128 * 4, "gelu")
    # model_torch.set_replace_method("relun1")
    # model_torch.norm1 = torch_models.LayerNormSubstitute(128)
    # model_torch.norm2 = torch_models.LayerNormSubstitute(128)
    # 1 layer of bert-tiny

    # input_shape = (1, 128, 512) # [B, N, E]
    # model_name, model_torch = "", torch_models.TransformerEncoderLayer(512, 8, 512 * 4, "relu") 
    # model_torch.set_replace_method("relun1")
    # model_torch.norm1 = torch_models.LayerNormSubstitute(512)
    # model_torch.norm2 = torch_models.LayerNormSubstitute(512)
    # # 1 layer of bert-medium

    input_shape = (1, 128, 64 * 12) # [B, N, E]
    model_name, model_torch = "", torch_models.TransformerEncoderLayer(64 * 12, 8, 64 * 12 * 4, "relu") 
    model_torch.set_replace_method("relun1")
    model_torch.norm1 = torch_models.LayerNormSubstitute(64 * 12)
    model_torch.norm2 = torch_models.LayerNormSubstitute(64 * 12)
    # 1 layer of bert-base

    # input_shape = (1, 128, 128) # [B, N, E]
    # model_name, model_torch = "", torch_models.TransformerEncoderLayer(128, 2, 512, "gelu")
    
    # input_shape = (1, 128, 128) # [B, N, E]
    # model_name, model_torch = "", torch_models.TransformerEncoderStack(2, 128, 2, 512, "relu")
    # for layer in model_torch.layers:
    #     layer.set_replace_method("relun1")
    #     layer.norm1 = torch_models.LayerNormSubstitute(128)
    #     layer.norm2 = torch_models.LayerNormSubstitute(128)

    # input_shape = (1, 128, 128) # [B, N, E]
    # model_torch = torch_models.BertWithoutEmbedding(
    #     torch_models.TransformerEncoderStack(2, 128, 2, 512, "relu"),
    #     torch.nn.Sequential(torch_models.BertPooler(), torch.nn.Linear(128, 128), torch.nn.Tanh()),
    #     torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(128, 2))
    # ) # Bert-tiny but GELU replaced by ReLU
    
    # input_shape = (1, 128, 512) # [B, N, E]
    # model_torch = torch_models.BertWithoutEmbedding(
    #     torch_models.TransformerEncoderStack(8, 512, 8, 2048, "relu"),
    #     torch.nn.Sequential(torch_models.BertPooler(), torch.nn.Linear(512, 512), torch.nn.Tanh()),
    #     torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(512, 2))
    # ) # Bert-medium but GELU replaced by ReLU

    # input_shape = (1, 32, 128) # [B, N, E]
    # _embeddings, model_torch = modelloader.get_model_bert_tiny_seq_classification()

    # input_shape = (1, 128, 128) # [B, N, E]
    # pretrained_model_path = "../LoRA/convert/outputs/bert-tiny-mrpc-gelu-relu-softmax-relun1-ln1-affine-ln2-affine"
    # model_base_name = "prajjwal1/bert-tiny"
    # config = transformers.AutoConfig.from_pretrained(
    #     model_base_name,
    #     num_labels = 2,
    #     finetuning_task = "mrpc",
    # )

    # model_torch = transformers.AutoModelForSequenceClassification.from_pretrained(
    #     model_base_name,
    #     config=config,
    # )
    # previous_modification = load_previous_modification(os.path.join(pretrained_model_path, "modification.txt"))
    # model_modify.modify_model_with_instruction(model_torch, previous_modification)
    # model_torch.load_state_dict(torch.load(os.path.join(pretrained_model_path, "model.pt")))
    # _, model_torch = model_convert.convert_Roberta(model_torch, '')

    # torchinfo.summary(model_torch, depth=10)
    model_torch.eval()

    comm.send(input_shape)
    
    model = server_modules.server_model_from_pytorch(model_torch, cryp, comm)
    model.eval()

    comm.send(model.describe())
    
    model.prepare(input_shape)
    print("Prepared")

    SCALE = cryp.default_scale()
    DOUBLE_SCALE = SCALE * SCALE

    inputs = (np.random.rand(*input_shape) - 0.5)

    inputs_a = share(inputs, start_scale)

    timed = time.time()

    batchsize = input_shape[0]

    # try:

    if isinstance(model, server_modules.ScaledDotProductAttention):
        seqlen = input_shape[1]
        mask = np.where(np.random.randint(0, 2, (batchsize, seqlen, seqlen)) > 0, 0, -200)
        # both privacy-module and torch-module uses float masks directly added to q*k
        mask_a = share(mask, SCALE)
        outputs_a = model.forward(inputs_a, mask_a)
        outputs = cryp.to_decimal(cryp.field_add(outputs_a, comm.recv()), scale=DOUBLE_SCALE)
        outputs_torch = model_torch(to_tensor(inputs), to_tensor(mask))
        outputs = np.reshape(outputs, outputs_torch.shape)
        compare(outputs, outputs_torch)

    elif isinstance(model, (server_modules.MultiheadAttention, server_modules.TransformerEncoderLayer)):
        seqlen = input_shape[1]
        mask = np.where(np.random.randint(0, 2, (batchsize, seqlen, seqlen)) > 0, False, True)
        # torch-module uses bool masks, privacy-module uses float masks added to qk.
        mask_float = np.where(mask, -200, 0)
        mask_a = share(mask_float, SCALE)
        outputs_a = model.forward(inputs_a, mask_a)
        outputs = cryp.to_decimal(cryp.field_add(outputs_a, comm.recv()), scale=SCALE)
        outputs_torch = model_torch(to_tensor(inputs), to_tensor(mask, torch.bool))
        outputs = np.reshape(outputs, outputs_torch.shape)
        compare(outputs, outputs_torch)

    elif isinstance(model, (server_modules.TransformerEncoderStack, server_modules.BertWithoutEmbedding)):
        seqlen = input_shape[1]
        mask = []
        for i in range(batchsize):
            x = torch.randint(1, seqlen, (1,))
            mask_single = int(x) * [0,] + (seqlen - int(x)) * [1,]
            mask.append(mask_single)
        mask = np.array(mask)
        # torch-module uses 0/1 masks (1 for pass, 0 for masked),
        # privacy-module uses float masks added to qk.
        mask_float = np.where(mask, 0, -200)
        mask_a = share(mask_float, SCALE)
        outputs_a = model.forward(inputs_a, mask_a)
        decode_scale = SCALE
        if isinstance(model, server_modules.BertWithoutEmbedding):
            decode_scale = DOUBLE_SCALE
        outputs = cryp.to_decimal(cryp.field_add(outputs_a, comm.recv()), scale=decode_scale)
        outputs_torch = model_torch(to_tensor(inputs), to_tensor(mask, torch.bool))
        outputs = np.reshape(outputs, outputs_torch.shape)
        compare(outputs, outputs_torch)

    print("forward time =", time.time() - timed)
    
    # except Exception as e:
    #     print(e)

    comm.close_connection()

def test_real():

    comm = communication_server.ServerCommunication()
    comm.listen()
    comm.accept_connection()
    cryp = crypto.EvaluationUtils()
    cryp.receive_keys(comm.recv())

    test_count = comm.recv()
    pretrained_model_path = comm.recv()
    check_correct = comm.recv()
    check_layerwise: bool = comm.recv()
    use_token_type_ids = comm.recv()

    model_base_name = comm.recv()
    model_filename = comm.recv()
    num_labels = comm.recv()
    task_name = comm.recv()
    
    config = transformers.AutoConfig.from_pretrained(
        model_base_name,
        num_labels = num_labels,
        finetuning_task = task_name,
    )

    model_torch = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_base_name,
        config=config,
    )
    # load checkpoint
    previous_modification = load_previous_modification(os.path.join(pretrained_model_path, "modification.txt"))
    model_modify.modify_model_with_instruction(model_torch, previous_modification)
    model_torch.load_state_dict(torch.load(os.path.join(pretrained_model_path, model_filename)))
    model_torch.eval()

    embedding_path = os.path.join(pretrained_model_path, "embedding.pt")
    print("Saved embedding to {0}".format(embedding_path))

    torchinfo.summary(model_torch, depth=100)

    _, converted_model = model_convert.convert_Roberta(model_torch, embedding_path)

    comm.send("embedding.pt")
    
    converted_model.eval()

    model_torch.eval()
    model: server_modules.BertWithoutEmbedding = server_modules.server_model_from_pytorch(converted_model, cryp, comm)
    model.eval()

    comm.send(model.describe())

    input_shape = comm.recv()
    print("Input shape (prepare) = {0}".format(input_shape))
    model.prepare(input_shape)
    print("Prepare done.")

    for step in range(test_count):
        if test_count != 1:
            print("[step = {0}]".format(step))

        embedded_share = np.zeros(input_shape).flatten()
        attention_mask_share = np.zeros((input_shape[0], input_shape[1])).flatten()

        if not check_layerwise:
            output_share = model.forward(embedded_share, attention_mask_share)
        else:
            print("---- Compare layerwise ----")
            def compare_merge(output_share, output_torch, prompt, double_scale=False):
                scale = cryp.default_scale()
                if double_scale:
                    scale = scale**2
                output_merged = cryp.to_decimal(cryp.field_add(output_share, comm.recv()), scale=scale, shape=output_torch.shape)
                compare(output_merged, output_torch, prompt)

            embeddings_torch = comm.recv()
            attention_mask_torch = comm.recv()
            # encoder stack
            batchsize = input_shape[0]
            seqlen = input_shape[1]
            attention_mask_share = np.reshape(attention_mask_share, (batchsize, 1, seqlen)).repeat(seqlen, 1) # [B, N, N]
            attention_mask_torch = torch.where(attention_mask_torch > 0, False, True)[:, None, :].repeat((1, seqlen, 1))
            output_share = embedded_share
            output_torch = embeddings_torch
            for i, encoder_layer in enumerate(model.stack.layers):
                encoder_layer_torch = converted_model.stack.layers[i]
                # multihead attention
                output_share_r = encoder_layer.mha.forward(output_share, attention_mask_share)
                output_torch_r = encoder_layer_torch.self_attention(output_torch, attention_mask_torch)
                compare_merge(output_share_r, output_torch_r, "l{0}.mha".format(i))
                # residual 1
                output_share_r = encoder_layer.dropout1.forward(output_share_r)
                output_torch_r = encoder_layer_torch.dropout1(output_torch_r)
                output_share = encoder_layer.add(output_share_r, output_share)
                output_torch = output_torch_r + output_torch
                compare_merge(output_share, output_torch, "l{0}.re1".format(i))
                # norm1
                output_share = encoder_layer.norm1.forward(output_share)
                output_share = encoder_layer.comm.truncate(output_share)
                output_torch = encoder_layer_torch.norm1(output_torch)
                compare_merge(output_share, output_torch, "l{0}.ln1".format(i))
                # linear 1
                output_share_r = encoder_layer.linear1.forward(output_share)
                output_torch_r = encoder_layer_torch.linear1(output_torch)
                compare_merge(output_share_r, output_torch_r, "l{0}.ff1".format(i), double_scale=True)
                # activation
                output_share_r = encoder_layer.activation.forward(output_share_r)
                output_torch_r = encoder_layer_torch.activation(output_torch_r)
                compare_merge(output_share_r, output_torch_r, "l{0}.act".format(i))
                # linear 2
                output_share_r = encoder_layer.linear2.forward(output_share_r)
                output_share_r = encoder_layer.comm.truncate(output_share_r)
                output_torch_r = encoder_layer_torch.linear2(output_torch_r)
                compare_merge(output_share_r, output_torch_r, "l{0}.ff2".format(i))
                # residual 2
                output_share_r = encoder_layer.dropout2.forward(output_share_r)
                output_torch_r = encoder_layer_torch.dropout2(output_torch_r)
                output_share = encoder_layer.add(output_share_r, output_share)
                output_torch = output_torch_r + output_torch
                compare_merge(output_share, output_torch, "l{0}.re2".format(i))
                # norm2
                output_share = encoder_layer.norm2.forward(output_share)
                output_share = encoder_layer.comm.truncate(output_share)
                output_torch = encoder_layer_torch.norm2(output_torch)
                compare_merge(output_share, output_torch, "l{0}.ln2".format(i))
            # pooler
            if model.pooler is not None:
                output_share = model.pooler.forward(output_share)
                output_torch = converted_model.pooler(output_torch)
                compare_merge(output_share, output_torch, "pool")
            # classifier
            output_share = model.classifier.forward(output_share)
            output_torch = converted_model.classifier(output_torch)
            compare_merge(output_share, output_torch, "clsfr", double_scale=True)
        
        comm.send(output_share)

        if check_correct:
            print("---- Check correctness ----")
            inputs = comm.recv()
            mask = comm.recv()
            if use_token_type_ids:
                token_type_ids = comm.recv()
                output_torch = model_torch(inputs, mask, token_type_ids=token_type_ids)
            else:
                output_torch = model_torch(inputs, mask)
            print("output torch =", output_torch.logits.detach().cpu().numpy())

            embeddings = comm.recv()
            print("converted output torch", converted_model(embeddings, mask).detach().cpu().numpy())

    comm.close_connection()

def test_convert():
    
    pretrained_model_path = "../LoRA/convert/outputs/20230331-085313-bert-tiny-mrpc-relu-model/runs/Mar31_08-53-13_ig114/best_model"
    embedding_path = pretrained_model_path + "/embedding.pth"
    modify_relu = True

    config = transformers.AutoConfig.from_pretrained(pretrained_model_path)
    
    model_torch = transformers.AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_path, config=config)
    if modify_relu:
        import model_modify
        model_modify.modify_relu_all(model_torch, False)

    _, converted_model = model_convert.convert_Roberta(model_torch, False)

    converted_model.eval()
    model_torch.eval()

    embeddings = torch.load(embedding_path)
    
    x = torch.randint(0, 30000, (10, 128))
    mask = []
    for i in range(10):
        r = torch.randint(1, 128, (1,))
        mask_single = int(r) * [1,] + (128 - int(r)) * [0,]
        mask.append(mask_single)
    mask = torch.tensor(mask, dtype=torch.int8)
    
    y1 = model_torch(x, attention_mask=mask).logits
    y2 = converted_model(embeddings(x), mask)
    print(torch.max(torch.abs(y1 - y2)))

def test_nonlinear():
    
    comm = communication_server.ServerCommunication()
    comm.listen()
    comm.accept_connection()
    cryp = crypto.EvaluationUtils()
    cryp.receive_keys(comm.recv())

    # (relu^2) normalized to 1
    n = 128
    xoriginal = np.random.randn(1, n, n)
    x = cryp.to_field(xoriginal) # [n*n] scale
    x = comm.relu(x, False)[0] # [n*n]
    x = comm.elementwise_multiply(x, x) # [n*n] scale^2
    # xt = comm.truncate(x) # [n*n] scale
    # xt = np.reshape(xt, (1, n, n)) # [1,n,n] scale
    # s = cryp.field_mod(np.sum(xt, axis=-1, keepdims=True)) # [1,n,1] scale
    # s = np.repeat(s, n, axis=-1).flatten() # [n*n] scale
    # x = comm.divide_element(xt, s) # scale*scale
    # x = comm.truncate(x)
    # x = cryp.to_decimal(cryp.field_add(x, comm.recv()))
    # print(x)

    comm.close_connection()




if __name__ == "__main__":
    test_real()