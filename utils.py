import io
import pickle
import torch
import struct

def serialize_model(model):
    buffer=io.BytesIO()
    torch.save(model.state_dict(),buffer)
    return buffer.getvalue()

def deserialize_model(model,model_bytes):
    buffer=io.BytesIO(model_bytes)
    state_dict=torch.load(buffer)
    model.load_state_dict(state_dict)
    return model

def average_weights(weights):
    avg={}
    for key in weights[0].keys():
        avg[key]=sum([w[key] for w in weights])/len(weights)
    return avg

def send_msg(conn,obj):
    data=pickle.dumps(obj)
    length=struct.pack(">I",len(data))
    conn.sendall(length)
    conn.sendall(data)

def recvall(conn,n):
    data=b""
    while len(data)<n:
        packet=conn.recv(n-len(data))
        if not packet:
            return None
        data+=packet
    return data

def recv_msg(conn):
    raw_len=recvall(conn,4)
    if not raw_len:
        return None
    msg_len=struct.unpack(">I",raw_len)[0]
    data=recvall(conn,msg_len)
    return pickle.loads(data)