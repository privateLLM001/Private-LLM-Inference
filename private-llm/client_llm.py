import client_modules
import communication_client
from crypto_switch import cryptography as crypto
import numpy as np
import model
import time
import torch 
import dataloader
import transformers
from tqdm import tqdm
import os
import model_modify

single_mrpc = [
    {
        'sentence1': "He said the foodservice pie business doesn 't fit the company 's long-term growth strategy .", 
        'sentence2': '" The foodservice pie business does not fit our long-term growth strategy .', 
        'label': 1, 
        'idx': 9, 
        'input_ids': [101, 2002, 2056, 1996, 9440, 2121, 7903, 
                        2063, 11345, 2449, 2987, 1005, 1056, 4906, 
                        1996, 2194, 1005, 1055, 2146, 1011, 2744, 
                        3930, 5656, 1012, 102, 1000, 1996, 9440, 
                        2121, 7903, 2063, 11345, 2449, 2515, 2025,
                        4906, 2256, 2146, 1011, 2744, 3930, 5656, 1012, 
                        102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                        0, 0, 0], 
        'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0], 
        'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0]
    },
    {
        'sentence1': 'Magnarelli said Racicot hated the Iraqi regime and looked forward to using his long years of training in the war .', 
        'sentence2': 'His wife said he was " 100 percent behind George Bush " and looked forward to using his years of training in the war .', 
        'label': 0, 
        'idx': 18, 
        'input_ids': [101, 20201, 22948, 2056, 10958, 19053, 4140, 6283, 1996, 8956, 6939, 1998, 2246, 2830, 2000, 2478, 2010, 2146, 2086, 1997, 2731, 1999, 1996, 2162, 1012, 102, 2010, 2564, 2056, 2002, 2001, 1000, 2531, 3867, 2369, 2577, 5747, 1000, 1998, 2246, 2830, 2000, 2478, 2010, 2086, 1997, 2731, 1999, 1996, 2162, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
]

def load_previous_modification(file):
    if not os.path.exists(file):
        return []
    # load from a file, every line has 3 parts: layer_id, replace_layer, replace_method
    with open(file, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split(",") for line in lines]
    lines = [(int(line[0].strip()),line[1].strip(),line[2].strip()) for line in lines]
    return lines

def test_modules():

    comm = communication_client.ClientCommunication()
    comm.connect()
    cryp = crypto.EncryptionUtils()
    comm.send(cryp.generate_keys())

    def share():
        return comm.recv()

    input_shape = comm.recv()

    description = comm.recv()
    model = client_modules.client_model_from_description(description, cryp, comm)
    
    model.prepare(input_shape)
    print("Prepared")

    inputs_b = share()
    comm.clear_accumulation()
    comm.counter = 1
    
    timed = time.time()

    # try:
    
    if isinstance(model, client_modules.ScaledDotProductAttention):
        mask_b = share()
        outputs_b = model.forward(inputs_b, mask_b)
        comm.send(outputs_b)

    elif isinstance(model, (client_modules.MultiheadAttention, client_modules.TransformerEncoderLayer)):
        mask_b = share()
        outputs_b = model.forward(inputs_b, mask_b)
        comm.send(outputs_b)

    elif isinstance(model, (client_modules.TransformerEncoderStack, client_modules.BertWithoutEmbedding)):
        mask_b = share()
        outputs_b = model.forward(inputs_b, mask_b)
        comm.send(outputs_b)

    print("forward time =", time.time() - timed)
    print("forward trans = ", comm.get_transmission())

    # except Exception as e:
    #     print(e)


    comm.close_connection()


def test_real():


    dataset_name = "glue-qnli"
    model_base_name = f"prajjwal1/bert-medium" # prajjwal1/bert-tiny

    # pretrained_model_path = f"./convert/outputs/roberta-large-mrpc-gelu-relu-singlelayer-softmax-relun1-ln1-affine-ln2-affine" 
    pretrained_model_path = f"./convert/outputs/bert-medium-qnli-gelu-relu-softmax-relun1-ln1-affine-ln2-affine" 
    use_token_type_ids = model_base_name.find("roberta") == -1 # BERT uses token_type_ids, RoBERTa does not
    model_filename = "model.pt"
    test_count = 0 # 0 for all test data
    check_correct = True
    check_layerwise = False

    if dataset_name == "glue-mrpc" and test_count <= 2 and test_count != 0:
        num_labels = 2
        task_name = "mrpc"
        eval_dataset = single_mrpc[:test_count]
    else:
        _, eval_dataset, num_labels, task_name = dataloader.load_dataset(dataset_name, tokenizer_name=model_base_name)

    if check_correct:
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

    print("[Setting]")
    print("dataset_name =", dataset_name)
    print("model_base =", model_base_name)
    print("pretrained_model_path =", pretrained_model_path)
    print("model_filename =", model_filename)
    print("test_count =", test_count)
    print("check_correct =", check_correct)
    print("[HE/MPC Params]")
    print("Scale bits =", crypto.SCALE_BITS)
    print("Modulus bits =", crypto.PLAIN_BITS)
    print("Poly degree =", crypto.BFV_POLY_DEGREE)
    print("Q bit sizes =", crypto.BFV_Q_BITS)
    print("------------")

    if test_count == 0 or test_count > len(eval_dataset):
        test_count = len(eval_dataset)

    if check_correct:
        # First test on torch model
        print("Testing on torch model")
        correct = 0
        total = 0   
        for data_point in eval_dataset:
            data_inputs = data_point["input_ids"]
            data_attention_mask = data_point["attention_mask"]
            data_label = data_point["label"]

            inputs = torch.LongTensor(data_inputs).reshape((1, -1))
            attention_mask = torch.LongTensor(data_attention_mask).reshape((1, -1))
            if use_token_type_ids:
                token_type_ids = torch.LongTensor(data_point["token_type_ids"]).reshape((1, -1))
                output_torch = model_torch(inputs, attention_mask, token_type_ids=token_type_ids)
            else:
                output_torch = model_torch(inputs, attention_mask)
            output_torch = output_torch[0].detach().numpy()
            prediction = np.argmax(output_torch)
            if prediction == data_label:
                correct += 1
            total += 1
        print("Correct = {0}/{1} = {2}".format(correct, total, correct / total))

    comm = communication_client.ClientCommunication()
    comm.connect()
    cryp = crypto.EncryptionUtils()
    comm.send(cryp.generate_keys())

    comm.send(test_count)
    comm.send(pretrained_model_path)
    comm.send(check_correct)
    comm.send(check_layerwise)
    comm.send(use_token_type_ids)

    comm.send(model_base_name)
    comm.send(model_filename)
    comm.send(num_labels)
    comm.send(task_name)

    embedding_path = os.path.join(pretrained_model_path, comm.recv())

    embeddings = torch.load(embedding_path)
    
    description = comm.recv()
    model = client_modules.client_model_from_description(description, cryp, comm)

    # get input shape
    data_point = eval_dataset[0]
    inputs = torch.LongTensor(data_point["input_ids"]).reshape((1, -1))
    embedded = embeddings(inputs).detach().numpy()
    input_shape = embedded.shape
    comm.send(input_shape)

    model.prepare(input_shape)

    step = 0
    correct = 0
    total = 0
    different = 0
    comm.clear_accumulation()
    start_time = time.time()
    for data_point in tqdm(eval_dataset, total=test_count):
        # print(data_point)
        if test_count != 1:
            print("[step = {0}]".format(step))

        single_transmission = comm.get_transmission()

        data_inputs = data_point["input_ids"]
        data_attention_mask = data_point["attention_mask"]
        data_attention_mask = np.array(data_attention_mask)
        # print(data_attention_mask)
        data_label = data_point["label"]

        # Inputs and attention_mask are 1d numpy array
        # Need to reshape to [B, N] = [1, -1]
        inputs = torch.LongTensor(data_inputs).reshape((1, -1))
        attention_mask = np.array(data_attention_mask, dtype=np.int8).reshape((1, -1))
        if use_token_type_ids:
            token_type_ids = torch.LongTensor(data_point["token_type_ids"]).reshape((1, -1))
            embedded = embeddings(inputs, token_type_ids=token_type_ids).detach()
        else:
            embedded = embeddings(inputs).detach()
        
        embedded_share = cryp.to_field(embedded.numpy())
        attention_mask_share = cryp.to_field(np.where(attention_mask > 0, 0, -200))

        time_single = time.time()
        
        if not check_layerwise:
            output_share = model.forward(embedded_share, attention_mask_share)
        else:
            comm.send(embedded)
            comm.send(torch.LongTensor(data_attention_mask).reshape((1, -1)))
            # encoder stack
            batchsize = input_shape[0]
            seqlen = input_shape[1]
            attention_mask_share = np.reshape(attention_mask_share, (batchsize, 1, seqlen)).repeat(seqlen, 1) # [B, N, N]
            output_share = embedded_share
            for i, encoder_layer in enumerate(model.stack.layers):
                # multihead attention
                output_share_r = encoder_layer.mha.forward(output_share, attention_mask_share)
                comm.send(output_share_r)
                # residual 1
                output_share_r = encoder_layer.dropout1.forward(output_share_r)
                output_share = encoder_layer.add(output_share_r, output_share)
                comm.send(output_share)
                # norm1
                output_share = encoder_layer.norm1.forward(output_share)
                output_share = encoder_layer.comm.truncate(output_share)
                comm.send(output_share)
                # linear 1
                output_share_r = encoder_layer.linear1.forward(output_share)
                comm.send(output_share_r)
                # activation
                output_share_r = encoder_layer.activation.forward(output_share_r)
                comm.send(output_share_r)
                # linear 2
                output_share_r = encoder_layer.linear2.forward(output_share_r)
                output_share_r = encoder_layer.comm.truncate(output_share_r)
                comm.send(output_share_r)
                # residual 2
                output_share_r = encoder_layer.dropout2.forward(output_share_r)
                output_share = encoder_layer.add(output_share_r, output_share)
                comm.send(output_share)
                # norm2
                output_share = encoder_layer.norm2.forward(output_share)
                output_share = encoder_layer.comm.truncate(output_share)
                comm.send(output_share)
            # pooler
            if model.pooler is not None:
                output_share = model.pooler.forward(output_share)
                comm.send(output_share)
            # classifier
            output_share = model.classifier.forward(output_share)
            comm.send(output_share)

        output = cryp.field_add(output_share, comm.recv())
        output = cryp.to_decimal(output, scale=cryp.default_scale() ** 2)

        time_single = time.time() - time_single
        print("Single inference time =", time_single)
        print("Single transmission =", comm.get_transmission() - single_transmission)
        
        prediction = np.argmax(output)
        if prediction == data_label:
            correct += 1
        total += 1

        print("Current accuracy = {0}/{1} = {2}".format(correct, total, correct / total))

        if check_correct:
            print("label =", data_label)
            print("output =", output)
            if use_token_type_ids:
                output_torch = model_torch(inputs, torch.LongTensor(data_attention_mask).reshape((1, -1)), token_type_ids=token_type_ids)
            else:
                output_torch = model_torch(inputs, torch.LongTensor(data_attention_mask).reshape((1, -1)))
            output_torch = output_torch[0][0].detach().numpy()
            print("output torch =", output_torch)
            print("difference max =", np.max(np.abs(output - output_torch)))
            prediction_torch = np.argmax(output_torch)
            if prediction != prediction_torch:
                different += 1
                print("Different! private = {0}, torch = {1}".format(prediction, prediction_torch))


            comm.send(inputs)
            comm.send(torch.LongTensor(data_attention_mask).reshape((1, -1)))
            if use_token_type_ids:
                comm.send(token_type_ids)

            comm.send(embedded)

        step += 1
        if step == test_count:
            break

    comm.close_connection()
    print("total time = {0}".format(time.time() - start_time))
    print("linear transmission = ", comm.get_transmission())

    print("[Summary]")
    print("correct =", correct)
    print("total =", total)
    print("accuracy =", correct / total)
    if check_correct:
        print("different =", different)

def test_nonlinear():

    comm = communication_client.ClientCommunication()
    comm.connect()
    cryp = crypto.EncryptionUtils()
    comm.send(cryp.generate_keys())

    # (relu^2) normalized to 1
    n = 128
    xoriginal = np.zeros((1, n, n))
    x = cryp.to_field(xoriginal) # [n*n] scale
    x = comm.relu(x, False)[0] # [n*n]
    x = comm.elementwise_multiply(x, x) # [n*n] scale^2
    # xt = comm.truncate(x) # [n*n] scale
    # xt = np.reshape(xt, (1, n, n)) # [1,n,n] scale
    # s = cryp.field_mod(np.sum(xt, axis=-1, keepdims=True)) # [1,n,1] scale
    # s = np.repeat(s, n, axis=-1).flatten() # [n*n] scale
    # x = comm.divide_element(xt, s) # scale*scale
    # x = comm.truncate(x)
    # comm.send(x)

    comm.close_connection()

if __name__ == "__main__":
    test_real()