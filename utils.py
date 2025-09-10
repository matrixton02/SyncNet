import io
import pickle
import torch

def serialize_model(model):
    buffer=io.BytesIO()
    torch.save(model.state_dict(),buffer)
    return buffer.getvalue()

def deserialize_model(model,model_bytes):
    buffer=io.BytesOP(model_bytes)
    state_dict=torch.load(buffer)
    model.load_state_dict(state_dict)
    return model

def average_weights(weights):
    avg={}
    for key in weights[0].keys:
        avg[key]=sum([w[key] for w in weights])/len(weights)
    return avg