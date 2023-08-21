from .src.node import node_start
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--ip", type=str, default = "localhost:8214", help="IP address of the server")
parser.add_argument("--device", type=str, default = "cpu", help="Device to run the client on")
parser.add_argument('--wait_time', type = int, default= 5, help= 'time to wait before sending the next request')
parser.add_argument('--clients', type=int, required=False, default=10, help='Number of clients to run')
parser.add_argument('--niid', type=int, default=1, help='niid or iid')
parser.add_argument('--algorithm', type=str, default='fedavg', help='algorithm to run')
parser.add_argument('--epochs', type = int, default = 5, help= 'number of epochs')
parser.add_argument('--mu', type = int, default = 0.1, help= 'mu hyperparameter for fedprox')
parser.add_argument('--rounds', type = int, default = 20, help= 'number of communication rounds')
parser.add_argument('--carbon', type = int, default = 0, help= 'if 1 track carbon emission of the node')
args = parser.parse_args()

configs = {
    # "ip_address": args.ip,
    "wait_time": args.wait_time,
    "device": args.device,
    "num_of_clients": args.clients,
    "niid": args.niid,
    "algorithm": args.algorithm,
    "epochs": args.epochs,
    "mu": args.mu,
    "rounds": args.rounds,
    "carbon": args.carbon
}

if __name__ == '__main__':
    node_start(configs)
