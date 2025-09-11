import socket
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import FlexibleNN
from utils import deserialize_model, recv_msg, send_msg

def fedprox_loss(outputs, labels, model, global_params):
    criterion = nn.CrossEntropyLoss()
    ce_loss = criterion(outputs, labels)

    # adaptive mu based on drift
    drift = 0.0
    for param, global_param in zip(model.parameters(), global_params):
        drift += torch.norm(param - global_param).item()
    mu = min(0.1, max(0.001, drift / 1000))  # dynamic regularization

    # proximal term
    prox_term = 0.0
    for param, global_param in zip(model.parameters(), global_params):
        prox_term += ((param - global_param) ** 2).sum()

    return ce_loss + (mu / 2.0) * prox_term

def connect(SERVER_IP="127.0.0.1", PORT=5050):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((SERVER_IP, PORT))
    print(f"[CLIENT] Connected to server {SERVER_IP}:{PORT}")

    payload1 = recv_msg(s)  # client_id
    payload2 = recv_msg(s)  # model
    payload3 = recv_msg(s)  # arch
    payload4 = recv_msg(s)  # dataset
    payload5 = recv_msg(s)  # epochs
    payload6 = recv_msg(s)  # training type

    client_id = payload1["client_id"]
    model_bytes = payload2["model"]
    arch = payload3["arch"]
    data_list = payload4["data"]
    local_epochs = payload5["epochs"]
    training_type = payload6["training_type"]

    print(f"[CLIENT {client_id}] Config: {len(data_list)} samples, epochs={local_epochs}, training={training_type}")

    model = FlexibleNN(arch["input_dim"], arch["hidden_dims"], arch["output_dim"])
    model = deserialize_model(model, model_bytes)

    x, y = zip(*data_list)
    x, y = torch.stack(x), torch.tensor(y)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    global_params = [p.clone().detach() for p in model.parameters()]

    for epoch in range(local_epochs):
        for i, (inputs, labels) in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(inputs)

            if training_type == "fedprox":
                loss = fedprox_loss(outputs, labels, model, global_params)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if i % 5 == 0:
                msg = {
                    "type": "progress",
                    "client_id": client_id,
                    "epoch": epoch,
                    "progress": int(100 * i / len(loader)),
                    "loss": loss.item(),
                }
                send_msg(s, msg)

    msg = {"type": "weights", "client_id": client_id, "weights": model.state_dict()}
    send_msg(s, msg)
    print(f"[CLIENT {client_id}] Training done, sent weights")
    s.close()
