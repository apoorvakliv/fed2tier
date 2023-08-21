
import argparse
from .src.server import server_start
from .src.server_lib import save_intial_model

#the parameters that can be passed while starting the server 
parser = argparse.ArgumentParser()
parser.add_argument("--nodes", type=int, default = 2, help="Number of nodes to select from server's perspective")
parser.add_argument("--fraction", 
    type = float,
    default = None, 
    help = "Fraction of nodes to select out of the number provided or those available. Float between 0 to 1 inclusive")
parser.add_argument("--s_rounds", type=int, default = 20, help = "Total number of communication rounds to perform")
parser.add_argument("--n_rounds", type=int, default = 5, help = "Max no. of communication rounds to perform on each node")
parser.add_argument("--model_path", 
    default = "initial_model.pt",
    help = "The path of the initial server model's state dict")
parser.add_argument("--accept_conn",
    type = int,
    default = 1,
    help = "If set to 1, connections will be accpeted even after FL has begun, else if set to 0, they will be rejected.")
parser.add_argument('--algorithm', type= str, default = 'fedavg', help= 'Aggregation algorithm')
parser.add_argument('--dataset', type = str, default= 'MNIST',
                     help= 'datsset.Use CUSTOME for local dataset')
parser.add_argument('--net', type = str, default = 'LeNet', help= 'client network')
parser.add_argument('--batch_size', type = int, default = 8, help= 'batch size')
parser.add_argument('--resize_size', type = int, default = 32, help= 'resize dimension')
parser.add_argument('--threshold', type = float, default = 0.8, help= 'node side accuracy threshold for early stopping')
args = parser.parse_args()

configurations = {
    "num_of_nodes": args.nodes,
    "fraction_of_clients": args.fraction,
    "num_of_rounds": args.s_rounds,
    "max_rounds_per_node": args.n_rounds,
    "initial_model_path": args.model_path,
    "accept_conn_after_FL_begin": args.accept_conn,
    "algorithm": args.algorithm,
    "dataset": args.dataset,
    "net": args.net,
    "batch_size": args.batch_size,
    "resize_size": args.resize_size,
    "threshold": args.threshold,
}

#start the server with the given parameters
if __name__ == '__main__':
    
    save_intial_model(configurations)
    server_start(configurations) 