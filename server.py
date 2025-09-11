import socket
import threading
import pickle
import torch
from torch.utils.data import random_split,TensorDataset,Subset
import pandas as pd
from torchvision import datasets,transforms
from utils import serialize_model,average_weights, send_msg
from model import FlexibleNN
from flask import Flask,jsonify,render_template


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


def handle_client(conn,client_id,data_chunk,model_bytes,arch):
    print(f"[SERVER] Sending model and data to client {client_id}")
    data_list=[(x,y) for x,y, in data_chunk]
    # payload={
    #     "model":model_bytes,
    #     "arch":arch,
    #     "data":data_list,
    #     "client_id":client_id
    # }
    send_msg(conn, {"client_id":client_id})
    send_msg(conn, {"model": model_bytes})
    send_msg(conn, {"arch": arch})
    send_msg(conn, {"data": data_list})

    while True:
        try:
            data=conn.recv(10**6)
            if not data:
                break
            msg=pickle.loads(data)
            if msg["type"]=="progress":
                client_update.setdefault(client_id,[]).append(msg)
            elif msg["type"]=="weights":
                final_weight.append(msg["weights"])
                print(f"[SERVER] Received final weights from client {client_id}")
                break
        except:
            break

    conn.close()

def run_server(HOST="0.0.0.0",PORT=5000,num_clients=2):
    dataset,input_dim,output_dim=load_dataset_user()
    model,hidden_dims=build_model_user(input_dim,output_dim)

    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.bind((HOST,PORT))
    s.listen(num_clients)
    print(f"[SERVER] Listening on {HOST}:{PORT}")

    model_bytes=serialize_model(model)
    data_chunk_size=len(dataset)//num_clients
    data_chunk=[]
    for i in range(num_clients):
        start = i * data_chunk_size
        end = (i + 1) * data_chunk_size
        indices = list(range(start, end))
        data_chunk.append(Subset(dataset, indices))
    
    client_id=0

    while client_id<num_clients:
        conn,addr=s.accept()
        print(f"[SERVER] Client {client_id} connected from {addr}")
        arch={"input_dim":input_dim,"hidden_dims":hidden_dims,"output_dim":output_dim}
        threading.Thread(target=handle_client,args=(conn,client_id,data_chunk[client_id],model_bytes,arch)).start()
        client_id+=1

    # dash_thread = threading.Thread(target=run_dashboard, daemon=True)
    # dash_thread.start()
    
    while len(final_weight)<num_clients:
        pass

    average=average_weights(final_weight)
    model.load_state_dict(average)
    torch.save(model.state_dict(),"final_model.pth")
    print("[SERVER] Training complete. Final model saved as final_model.pth")