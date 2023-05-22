import model
import client_modules
import communication_client
from crypto_switch import cryptography
import dataloader
from tqdm import tqdm
import numpy as np
import torch.nn.functional
import time
import argparse

def one_hot(x):
    return torch.nn.functional.one_hot(torch.tensor(x), num_classes=10).numpy()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", type=int, default=0) # split count, 0 to disable
    parser.add_argument("-e", type=int, default=10) # epochs
    args = parser.parse_args()

    print(f"Split   = {args.s}")
    print(f"Epoch   = {args.e}")

    SPLIT = args.s != 0
    SPLIT_JOIN_COUNT = args.s

    # establish connection
    comm = communication_client.ClientCommunication()
    comm.connect()
    print("Client: Connection established.")

    # create crypto
    crypto = cryptography.EncryptionUtils()
    comm.send(crypto.generate_keys())
    print("Client: Cryptography context created.")
    
    # load model
    model_description = comm.recv()
    model = client_modules.client_model_from_description(model_description, crypto, comm)
    print("Client: Model created from description successfully.")

    # prepare
    dataset_name, input_shape = comm.recv()
    batchsize = input_shape[0]
    model.prepare(input_shape)
    print("Client: Model preprocessing finished.")

    # load data
    train_data, test_data = dataloader.load_dataset(dataset_name)
    # train_data, test_data = dataloader.load_dataset("cifar10-32")
    test_dataloader = dataloader.BatchLoader(test_data, batchsize, True)
    if SPLIT:
        train_dataloaders = [
            dataloader.BatchLoader(list(filter(lambda sample: sample[1] >= i*2 and sample[1] <= (i*2+1), train_data)), batchsize, True)
            for i in range(5)
        ]
        train_dataloader = dataloader.BatchLoader(train_data, batchsize, True)
    else:
        train_dataloader = dataloader.BatchLoader(train_data, batchsize, True)
    print("Client: Train dataset loaded")

    # train
    for epoch in range(args.e):
        epoch_time = 0
        print(f"Epoch = {epoch}")
        for step in tqdm(range(len(train_dataloader))):
            # if step > 50: break
            timer = time.time()
            skip_train_step = False
            if SPLIT:
                if step % 5 >= SPLIT_JOIN_COUNT: skip_train_step = True
                inputs, labels = train_dataloaders[step % 5].get()
            else:
                inputs, labels = train_dataloader.get()
            if not skip_train_step:
                comm.send("train")
                inputs = inputs.numpy()
                labels = labels.numpy()
                x = crypto.to_field(inputs)
                output = model.forward(x)
                output_another_share = comm.recv()
                output = crypto.field_mod(output + output_another_share)
                output = crypto.to_decimal(output, crypto.default_scale() ** 2, (batchsize, -1))
                loss = torch.nn.functional.cross_entropy(torch.tensor(output), torch.tensor(labels))
                
                pred = np.argmax(output, axis=1)
                correct = np.sum(pred == labels)
                total = len(labels)

                print(f"Loss = {loss:.6f}", "logit max =", np.max(np.abs(output)), f"correct={correct}/{total}")
                output_softmax = np.exp(output) / np.reshape(np.sum(np.exp(output), axis=1), (-1, 1))
                output_grad = (output_softmax - one_hot(labels)) / len(labels)
                output_grad = crypto.to_field(output_grad)
                model.backward(output_grad)
                epoch_time += time.time() - timer
            if (step + 1) % (len(train_dataloader) // 5) == 0:
                comm.send("anneal")
                print("testing")
                comm.send("test")
                correct, total = comm.recv()
                print(f"correct/total = {correct}/{total} = {correct/total}")
                comm.send("save")
                name = f"epoch{epoch}-step{step}"
                if SPLIT: name += f"-split-join{SPLIT_JOIN_COUNT}"
                comm.send(name)
        print("Epoch time =", epoch_time)
    comm.send("finish")

    # close connection
    comm.close_connection()
