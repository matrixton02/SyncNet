from server import run_server
from client import connect

if __name__=="__main__":
    print("===SyncNet===")
    print("1.Run as Server")
    print("2.Run as Client")
    choice=input("Enter choice: ").strip()

    if choice=="1":
        HOST="0.0.0.0"
        PORT=5050
        num_clients=int(input("Enter number of clients: "))
        run_server(HOST,PORT,num_clients)
    elif choice=="2":
        SERVER_IP=input("Enter the server ip: ")
        PORT=int(input("Enter port number: "))
        connect(SERVER_IP,PORT)