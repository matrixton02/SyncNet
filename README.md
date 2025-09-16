# 🚀 SyncNet: Distributed ML Training Across Laptops  

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)  
![PyTorch](https://img.shields.io/badge/PyTorch-ML-orange.svg)  
![Hackathon](https://img.shields.io/badge/Hackathon-Top%2015%20Project-success.svg)  

## 📌 Overview  
**SyncNet** is a lightweight framework for **distributed machine learning training** that allows multiple laptops on the same network to collaboratively train a single ML model — without requiring expensive GPUs or cloud services.  

- One device runs as the **server** (coordinator).  
- Other devices connect as **clients**.  
- The dataset and model are **split across clients** → each client trains locally → weights are aggregated (FedAvg / FedProx).  
- A **real-time dashboard** shows training progress for each client.  

This project was built during a hackathon, where it placed **Top 15 out of 296 teams** 🏆.  

---

## 🎯 Problem Statement  
Training ML models in hackathons or student projects is often bottlenecked by **limited compute**.  
- Teams cannot finish training large models within time.  
- Cloud GPUs are expensive and difficult to set up quickly.  
- Existing federated learning frameworks are heavy and impractical for small setups.  

**SyncNet solves this** by enabling fast, collaborative training across multiple laptops connected via LAN/WiFi.  

---

## ⚡ Key Features  
✅ **Custom Models** – Supports flexible MLPs (user-defined layers) and CNNs.  
✅ **Training Algorithms** – Implements both **FedAvg** and **FedProx** (adaptive μ).  
✅ **IID & Non-IID Splits** – Can simulate real-world skewed data distributions.  
✅ **Parallel Training** – Clients train simultaneously, reducing wall-clock time.  
✅ **Progress Dashboard** – Real-time monitoring of client training (loss, accuracy, progress).  
✅ **Custom Dataset Support** – Use MNIST or upload your own CSV dataset.  

---

## 🛠️ Technical Approach  
- **Model Distribution:** Model + dataset chunks sent via socket protocol (`send_msg`, `recv_msg`).  
- **Dynamic Architectures:**  
  - `FlexibleNN` → arbitrary hidden layers for MLPs.  
  - `SimpleCNN` → lightweight CNN for image datasets.  
- **Federated Training:**  
  - FedAvg for IID data.  
  - FedProx (with adaptive μ and gradient clipping) for Non-IID data to reduce client drift.  
- **Aggregation:** Server collects weights, averages them, and produces the final global model.  

---

## 📂 Project Structure  
```
SyncNet/
│── server.py # Server (coordinator) code
│── client.py # Client (worker) code
│── main.py # Entry point (choose server/client mode)
│── model.py # Model definitions (MLP, CNN)
│── utils.py # Helper functions (send/recv messages, serialization, etc.)
│── dashboard.py # Flask-based real-time dashboard
│── test_model.py # Test bench (loads saved model and evaluates)
│── README.md # Project documentation
```
---

## 🚀 Getting Started  

### 1️⃣ Clone Repository
```bash
git clone https://github.com/<your-username>/SyncNet.git
cd SyncNet
```
## 2️⃣ Install Requirements
```bash
pip install torch torchvision flask pandas numpy
```
## 3️⃣ Run Server
```bash
python main.py
# Choose option 1: Run as Server
```
## 4️⃣ Run Clients
```bash
python main.py
# Choose option 2: Run as Client
# get the ip address of the server and enter the port number
# make sure all clients trying to access the server are connected on the same network
#for testing select 2 layered mlp 128 and 64 neurons then chhose iid and minst data set
```
## 5️⃣ Testing the Model
```bash
python test_model.py
```
## 📊 Results

Centralized (single machine, 20 epochs, 2-layer MLP): ~97% accuracy (60s).

Distributed (3 clients, FedAvg, 25 epochs): ~95% accuracy (12s).

Non-IID (3 clients, FedProx, 5 epochs, 128–64 MLP): ~85–90% accuracy.

CNN (5 epochs, distributed): ~93–95% accuracy.

## 💡 Future Plans

 Support more algorithms (FedNova, Scaffold, FedOpt).

 Work on the efficiency of fedprox.

 Add CIFAR-10 and larger datasets.

 Implement P2P communication (no single server).

 Provide Docker setup for easy deployment.

 Integrate with cloud + edge devices (hybrid training).

## 👥 Team

Built during a hackathon by Team SyncNet 🚀.
