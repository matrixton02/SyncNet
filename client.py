import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import FlexibleNN
from utils import deserialize_model

def connect(SERVER_IP,PORT=5000):
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect((SERVER_IP,PORT))
    print(f"[CLIENT] Connected to server {SERVER_IP}:{PORT}")

    payload=pickle.loads(s.recv(10**8))
    client_id=payload["client_id"]
    arch=payload["arch"]
    model_bytes=payload["model"]
    data_list=payload["data"]

    model=FlexibleNN(arch["input_dim"],arch["hidden_dims"],arch["output_dim"])
    model=deserialize_model(model,model_bytes)

    x,y=zip(*data_list)
    x,y=torch.stack(x),torch.tensor(y)
    dataset=TensorDataset(x,y)
    loader=DataLoader(dataset,batch_size=32,shuffle=True)

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=0.01)

    print(f"[CLIENT {client_id}] Training on {len(dataset)} samples")

    for epoch in range(100):
        for i,(inputs,labels) in enumerate(loader):
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            if i%5==0:
                msg={"type":"progress","client_id":client_id,"epoch":epoch,"progress":int(100*i/len(loader)),"loss":loss.item()}
                s.sendall(pickle.dumps(msg))

    msg = {"type": "weights", "client_id": client_id, "weights": model.state_dict()}
    s.sendall(pickle.dumps(msg))
    print(f"[CLIENT {client_id}] Training done, sent weights")
    s.close()