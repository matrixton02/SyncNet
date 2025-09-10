import socket
import threading
import pickle
import torch
from utils import serialize_model,average_weights

client_update={}
final_weight=[]

def handle_client(conn,addr,client_id,data_chunk,model_bytes):
    print(f"[SERVER] Sending model and data to client {client_id}")

    payload={
        "model":model_bytes,
        "data":data_chunk,
        "client_id":client_id
    }

    conn.sendall(pickle.dumps(payload))

    while True:
        try:
            data=conn.recv(10**6)
            if not data:
                break
            msg=pickle.loads(data)

            if msg["type"]=="progress":
                client_update[client_id]=msg
            elif msg["type"]=="weights":
                print(f"[SERVER] Received final weights from client {client_id}")
                break
        except:
            break

    conn.close()

def run_server(HOST="0.0.0.0",PORT=5000,dataset=None,model=None,num_clients=2):
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.bind((HOST,PORT))
    s.listen(num_clients)
    print(f"[SERVER] Listening on {HOST}:{PORT}")

    model_bytes=serialize_model(model)

    data_chunk_size=len(dataset)//num_clients
    data_chunk=[]
    for i in range(num_clients):
        chunk=dataset[i*data_chunk_size:(i+1)*data_chunk_size]
        data_chunk.append(chunk)
    
    client_id=0

    while client_id<num_clients:
        conn,addr=s.accept()
        print(f"[SERVER] Client {client_id} connected from {addr}")
        threading.Thread(target=handle_client,args=(conn,addr,client_id,data_chunk[client_id],model_bytes)).start()
        client_id+=1
    while len(final_weight)<num_clients:
        pass

    average=average_weights(final_weight)
    model.load_state_dict(average)
    torch.save(model.state_dict(),"final_model.pth")
    print("[SERVER] Training complete. Final model saved as final_model.pth")
