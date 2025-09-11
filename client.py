import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import FlexibleNN
from utils import deserialize_model,recv_msg, send_msg

def connect(SERVER_IP="0.0.0.0",PORT=5050):
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect((SERVER_IP,PORT))
    s.settimeout(60)
    print(f"[CLIENT] Connected to server {SERVER_IP}:{PORT}")

    try:
        payload1 = recv_msg(s)
        print("[CLIENT] Got client_id:", payload1)
        client_id=payload1["client_id"]
        payload2 = recv_msg(s)
        print("[CLIENT] Got model")
        model_bytes=payload2["model"]
        payload3 = recv_msg(s)
        print("[CLIENT] Got arch")
        arch=payload3["arch"]
        payload4 = recv_msg(s)
        print("[CLIENT] Got dataset")
        data_list=payload4["data"]
    except Exception as e:
        print(f"[CLIENT] Error while receiving: {e}")
        s.close()
    print(f"[CLIENT] Got dataset with {len(data_list)} samples")

    model=FlexibleNN(arch["input_dim"],arch["hidden_dims"],arch["output_dim"])
    model=deserialize_model(model,model_bytes)

    x,y=zip(*data_list)
    x,y=torch.stack(x),torch.tensor(y)
    dataset=TensorDataset(x,y)
    loader=DataLoader(dataset,batch_size=32,shuffle=True)

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=0.01)

    print(f"[CLIENT {client_id}] Training on {len(dataset)} samples")

    for epoch in range(20):
        for i,(inputs,labels) in enumerate(loader):
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            if i%5==0:
                msg = {
                    "type": "progress",
                    "epoch": epoch,
                    "progress": int(100 * i / len(loader)),
                    "loss": loss.item()
                }
                send_msg(s,msg)   # now using send_msg for consistency

    msg = {"type": "weights", "client_id": client_id, "weights": model.state_dict()}
    send_msg(s,msg)
    print(f"[CLIENT {client_id}] Training done, sent weights")
    s.close()