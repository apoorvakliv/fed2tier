import os
import sys
import time
import json

from torch.multiprocessing import Process
from torch import multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fed2tier.server.src.server import server_start
from fed2tier.node.src.node import node_start

def get_config(action, action2, config_path=""):
    """
    Get the configuration file as json from it 
    """
    
    root_path = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))
    config_path = os.path.join(root_path, 'configs')
    action = action + '.json'
    with open(os.path.join(config_path, action), encoding='UTF-8') as f1:
        config = json.load(f1)
        config = config[action2]

    return config


def execute(process):
    os.system(f'{process}')

    
def tester(configs , no_of_nodes):
    """
    Return the tester to each test algorithm.
    Late is introduced for intermediate connection
    """
    
    multiprocessing.set_start_method('spawn', force=True)

    server = Process(target=server_start, args=(configs['server'],))
    nodes = []
    server.start()
    time.sleep(5)
    for i in range(no_of_nodes):
        client = Process(target=node_start, args=(configs['client'],))
        nodes.append(client)
        client.start()
        time.sleep(2)

    clients_list = list(range(len(nodes)))
    for i in clients_list:
        nodes[i].join()
    server.join()

    
def get_result(dataset, algorithm):
    """
    Return the result to each test algorithm.
    Dataset and algorithm defines as for which dataset the result is required
    """
    
    dir_path = './server_results/'+dataset+'/'+algorithm
    lst = os.listdir(dir_path)
    lst.sort()
    lst = lst[-1]
    dir_path = dir_path+'/'+lst
    lst = os.listdir(dir_path)
    lst.sort()
    lst = lst[-1]
    print(lst)
    with open (f'{dir_path}/{lst}/FL_results.txt', 'r', encoding='UTF-8') as file:
        for line in file:
            pass
        result_dict = eval(line)
    return result_dict