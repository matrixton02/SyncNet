import socket
import threading
import torch
from torch.utils.data import Subset, TensorDataset
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from utils import serialize_model, average_weights, send_msg, recv_msg
from model import FlexibleNN
import time

client_update = {}
final_weights = []

def load_dataset_user():
    choice = input("Use custom CSV dataset (y/n): ").strip().lower()
    if choice == "y":
        csv_path = input("Enter path to CSV file: ").strip()
        df = pd.read_csv(csv_path)
        X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
        dataset = TensorDataset(X, y)
        input_dim = X.shape[1]
        output_dim = len(set(y.tolist()))
        return dataset, input_dim, output_dim
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        input_dim, output_dim = 28 * 28, 10
        return dataset, input_dim, output_dim

def create_non_iid_split(dataset, num_clients, num_classes_per_client):
    num_classes = len(dataset.classes)
    labels = np.array(dataset.targets)
    class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}
    client_classes = [
        np.random.choice(num_classes, num_classes_per_client, replace=False)
        for _ in range(num_clients)
    ]
    class_to_clients = {i: [] for i in range(num_classes)}
    for client_id, classes in enumerate(client_classes):
        for class_id in classes:
            class_to_clients[class_id].append(client_id)

    client_data_indices = [[] for _ in range(num_clients)]
    for class_id, clients in class_to_clients.items():
        if not clients:
            continue

        indices_for_class = class_indices[class_id]
        np.random.shuffle(indices_for_class)
        split_indices = np.array_split(indices_for_class, len(clients))
        
        for i, client_id in enumerate(clients):
            client_data_indices[client_id].extend(split_indices[i])

    client_subsets = [Subset(dataset, indices) for indices in client_data_indices]
    return client_subsets

def build_model_user(input_dim, output_dim):
    layers = int(input("Enter number of hidden layers: "))
    hidden_dims = []
    for i in range(layers):
        neurons = int(input(f"Enter number of neurons in hidden layer {i+1}: "))
        hidden_dims.append(neurons)
    print(f"Building model: {input_dim} -> {hidden_dims} -> {output_dim}")
    return FlexibleNN(input_dim, hidden_dims, output_dim), hidden_dims

def handle_client(conn, client_id, data_chunk, model_bytes, arch, local_epochs, training_type):
    print(f"[SERVER] Preparing to send {len(data_chunk)} samples to client {client_id}")
    data_list = [(x, y) for x, y in data_chunk]

    send_msg(conn, {"client_id": client_id})
    send_msg(conn, {"model": model_bytes})
    send_msg(conn, {"arch": arch})
    send_msg(conn, {"data": data_list})
    send_msg(conn, {"epochs": local_epochs})
    send_msg(conn, {"training_type": training_type})
    print(f"[SERVER] Sent config to client {client_id}")

    try:
        while True:
            data=recv_msg(conn)
            if not data:
                    continue
            msg=data
            if msg["type"]=="progress":
                client_update.setdefault(client_id,[]).append(msg)
            elif msg["type"]=="weights":
                final_weights.append(msg["weights"])
                print(f"[SERVER] Received final weights from client {client_id}")
                break
    except(ConnectionResetError,EOFError) as e: 
        print(f"[SERVER] Connection error with client {client_id}: {e}")
    finally:
        conn.close()

def run_server(HOST="0.0.0.0", PORT=5050, num_clients=2):
    dataset, input_dim, output_dim = load_dataset_user()
    model, hidden_dims = build_model_user(input_dim, output_dim)
    dataset_choice=input("what dataset you want to use (iid/noniid): ").strip().lower()
    chunks=[]
    if(dataset_choice=="noniid"):
        num_classes_per_client = int(input(f"Enter number of unique classes per client (e.g., 2 for MNIST): "))
        chunks = create_non_iid_split(dataset, num_clients, num_classes_per_client)
    else:
        data_chunk_size = len(dataset) // num_clients
        remainder = len(dataset) % num_clients
        start = 0
        for i in range(num_clients):
            end = start + data_chunk_size + (1 if i < remainder else 0)
            indices = list(range(start, end))
            chunks.append(Subset(dataset, indices))
            start = end
    local_epochs = int(input("Enter number of local epochs: "))
    training_type = input("Choose training type (fedavg/fedprox): ").strip().lower()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(num_clients)
    print(f"[SERVER] Listening on {HOST}:{PORT}")

    model_bytes = serialize_model(model)

    connections = []
    for client_id in range(num_clients):
        conn, addr = s.accept()
        print(f"[SERVER] Client {client_id} connected from {addr}")
        connections.append([conn, client_id])

    for i in connections:
        client_id=i[1]
        conn=i[0]
        arch = {"input_dim": input_dim, "hidden_dims": hidden_dims, "output_dim": output_dim}
        threading.Thread(target=handle_client,args=(conn, client_id, chunks[client_id], model_bytes, arch, local_epochs, training_type)).start()

    while len(final_weights) < num_clients:
        time.sleep(0.1)

    average = average_weights(final_weights,chunks)
    model.load_state_dict(average)
    torch.save(model.state_dict(), "final_model.pth")
    print("[SERVER] Training complete. Final model saved as final_model.pth")