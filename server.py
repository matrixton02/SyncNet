import socket
import threading
import pickle
import torch
from torch.utils.data import random_split,TensorDataset,Subset
import pandas as pd
from torchvision import datasets,transforms
from utils import serialize_model,average_weights, send_msg, recv_msg
from model import FlexibleNN
from flask import Flask,jsonify,render_template
import time

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/progress")
def progress():
    return jsonify(client_update)
def run_dashboard():
    print("[SERVER] Dashboard started at http://localhost:8000")
    app.run(port=8000, debug=False, use_reloader=False)

client_update={}
final_weight=[]

def load_dataset_user():
    choice=input("Use custom CSV dataset (y/n): ").strip().lower()
    if(choice=="y"):
        csv_path=input("Enter path to CSV file: ").strip()
        df=pd.read_csv(csv_path)
        x=torch.tensor(df.iloc[::-1].values,dtype=torch.float32)
        y=torch.tensor(df.iloc[::-1].value,dtype=torch.long)
        dataset=TensorDataset(x,y)
        input_dim=x.shape[1]
        output_dim=len(set(y.tolist()))
        return dataset,input_dim,output_dim
    else:
        transform=transforms.Compose([transforms.ToTensor()])
        dataset=datasets.MNIST(root="./data",train=True,download=True,transform=transform)
        input_dim,output_dim=28*28,10
        return dataset,input_dim,output_dim

def build_model_user(input_dim,output_dim):
    layers=int(input("Enter number of hidden layers: "))
    hidden_dims=[]
    for i in range(layers):
        neurons=int(input(f"Enter number of neurons in hidden layer {i+1}: "))
        hidden_dims.append(neurons)
    print(f"Building model: {input_dim}->{hidden_dims}->{output_dim}")
    return FlexibleNN(input_dim,hidden_dims,output_dim),hidden_dims


def handle_client(conn,client_id,data_chunk,model_bytes,arch,s):
    print(f"[SERVER] Preparing to send {len(data_chunk)} samples to client {client_id}")
    data_list=[(x,y) for x,y, in data_chunk]
    # payload={
    #     "model":model_bytes,
    #     "arch":arch,
    #     "data":data_list,
    #     "client_id":client_id
    # }
    print(f"[SERVER] Sending data to client {client_id}")
    send_msg(conn, {"client_id":client_id})
    send_msg(conn, {"model": model_bytes})
    print(f"[SERVER] Model data to client {client_id}")
    send_msg(conn, {"arch": arch})
    print(f"[SERVER] arch data to client {client_id}")
    send_msg(conn, {"data": data_list})
    print(f"[SERVER] training data to client {client_id}")
    print(f"[SERVER] Sent data to client {client_id}")

    try:
        while True:
                data=recv_msg(conn)
                if not data:
                    continue
                msg=data
                if msg["type"]=="progress":
                    client_update.setdefault(client_id,[]).append(msg)
                elif msg["type"]=="weights":
                    final_weight.append(msg["weights"])
                    print(f"[SERVER] Received final weights from client {client_id}")
                    break
    except(ConnectionResetError,EOFError) as e: 
        print(f"[SERVER] Connection error with client {client_id}: {e}")
    finally:
        conn.close()

def run_server(HOST="0.0.0.0",PORT=5050,num_clients=2):
    dataset,input_dim,output_dim=load_dataset_user()
    model,hidden_dims=build_model_user(input_dim,output_dim)

    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.bind((HOST,PORT))
    s.listen(num_clients)
    print(f"[SERVER] Listening on {HOST}:{PORT}")

    model_bytes=serialize_model(model)
    chunks = []
    data_chunk_size = len(dataset) // num_clients
    remainder = len(dataset) % num_clients
    start = 0

    connections=[]
    for i in range(num_clients):
        end = start + data_chunk_size + (1 if i < remainder else 0)
        indices = list(range(start, end))
        chunks.append(Subset(dataset, indices))
        start = end

    for client_id in range(num_clients):
        conn,addr=s.accept()
        print(f"[SERVER] Client {client_id} connected from {addr}")
        connections.append([conn,client_id])

    for i in connections:
        client_con=i[0]
        client_id=i[1]
        arch={"input_dim":input_dim,"hidden_dims":hidden_dims,"output_dim":output_dim}
        threading.Thread(target=handle_client,args=(client_con,client_id,chunks[client_id],model_bytes,arch,s)).start()

    # dash_thread = threading.Thread(target=run_dashboard, daemon=True)
    # dash_thread.start()
    
    while len(final_weight)<num_clients:
        time.sleep(0.1)

    average=average_weights(final_weight)
    model.load_state_dict(average)
    torch.save(model.state_dict(),"final_model.pth")
    print("[SERVER] Training complete. Final model saved as final_model.pth")