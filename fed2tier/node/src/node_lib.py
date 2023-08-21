import torch
from io import BytesIO
import json
import time
from datetime import datetime
import os
from .net import get_net
from .net_lib import train_model, test_model, load_data, train_fedavg, train_scaffold, fedadam, train_fedprox, train_feddyn
from .create_datasets import make_client_datasets
from .data_utils import distributionDataloader
from .ClientConnection_pb2 import  TrainResponse
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pandas as pd
from datetime import datetime
from codecarbon import  OfflineEmissionsTracker

def prepare_dataset_models(config, device, niid=1, num_of_clients=1, control_variate=None):
    #create datasets for the clients
    config['num_of_clients'] = num_of_clients
    config['niid'] = niid
    data_path, trainset, testset = make_client_datasets(config)


    client_dicts = []
    for i in range(config['num_of_clients']):
        train_dataset = distributionDataloader(trainset=trainset,data_path=data_path,clientID=i)
        client_dict = {}
        client_dict["trainloader"], client_dict["testloader"], client_dict["num_examples"] = load_data(train_dataset, testset)
        model = get_net(config)
        client_dict["model"] = model.to(device)
        if control_variate:
            client_dict['control_variate'] = None
        client_dict['prev_grads']=None
        client_dicts.append(client_dict)


    print("data was loaded")
    return client_dicts

#creates a save folder with the current timestamp and stores the client_dicts current state
fl_timestamp = f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"
save_dir_path = f"node_results/{fl_timestamp}"
os.makedirs(save_dir_path)


##make an eval list which will log all the eval results
eval_list = []


# torch.save(client_dicts, f"{save_dir_path}/initial_models.pt")

# #the different functions that can be performed by the client

# def evaluate(eval_order_message, device):
#     model_parameters_bytes = eval_order_message.modelParameters
#     model_parameters = torch.load( BytesIO(model_parameters_bytes), map_location="cpu" )

#     config_dict_bytes = eval_order_message.configDict
#     config_dict = json.loads( config_dict_bytes.decode("utf-8") )
#     print(config_dict["message"])
    
#     state_dict = model_parameters
#     eval_results = []
#     for client_dict in client_dicts:
#         client_dict["model"].load_state_dict(state_dict)
#         eval_loss, eval_accuracy = test_model(client_dict["model"], client_dict["testloader"])
#         eval_results.append( {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy} )

#     response_dict = average(eval_results)
#     response_dict_bytes = json.dumps(response_dict).encode("utf-8")
#     eval_response_message = EvalResponse(responseDict = response_dict_bytes)
#     return eval_response_message
def create_path(path, i=1):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        #replace last char of path with i
        path = path[:-1] + str(i)
        path = create_path(path, i+1)
    return path


def train(train_order_message, device, args):

    data_bytes = train_order_message.modelParameters
    data = torch.load( BytesIO(data_bytes), map_location="cpu" )
    model_parameters, control_variate_server  = data['model_parameters'], data['control_variate']

    config_dict_bytes = train_order_message.configDict
    config_dict = json.loads( config_dict_bytes.decode("utf-8") )
    print(config_dict["message"])

    ##get accuracy threshold for early stopping
    accuracy_threshold = config_dict['threshold']

    algorithm = args['algorithm']####algorithm for aggregation at node
    ## initialize control variate if scaffold is used
    if algorithm == "scaffold":
        dummy_modeldict = get_net(config_dict)
        control_variate = [torch.zeros_like(param).to(device) for param in dummy_modeldict.parameters()]
    else:
        control_variate = None

    client_dicts = prepare_dataset_models(config_dict, device, niid=args['niid'], num_of_clients=args['num_of_clients'], control_variate=control_variate)
  
    exec(f"from .algorithms.{algorithm} import {algorithm}") # nosec
    aggregator = eval(algorithm)() # nosec

    state_dict = deepcopy(model_parameters)
    rounds = args['rounds']
    if rounds > config_dict['n_rounds']:
        rounds = config_dict['n_rounds']
    path = create_path(f"{save_dir_path}/server_round_0")
    ##Create 2 dataframes for storing time and carbon emission of each client for eachround. column for clients and rows for rounds
    time_df = pd.DataFrame(columns=[f"client_{i}" for i in range(len(client_dicts))])
    carbon_df = pd.DataFrame(columns=[f"client_{i}" for i in range(len(client_dicts))])

    ##set carbon to true if args['carbon'] is 1 else false
    carbon = args['carbon']

    ###run communication rounds for the node in a for loop
    for round in range(rounds):
        ###make path for the round(for saving models and results)
        round_path = f"{path}/round_{round}"
        os.makedirs(round_path)
        #create new file inside model_checkpoints to store training results
        with open(f"{round_path}/FL_results.txt", "w") as file:
            pass
        for i, client_dict in enumerate(client_dicts):
            ##start calculating time and carbon emission for each client
            start_time = time.time()
            if carbon:
                carbon_tracker = OfflineEmissionsTracker(country_iso_code="IND", output_dir = round_path)
                carbon_tracker.start()
            client_dict["model"].load_state_dict(state_dict)
            epochs = args["epochs"]
            if args["algorithm"] == "scaffold":
                client_dict['model'], client_dict['delta_c'], client_dict['control_variate']= train_scaffold(client_dict["model"], control_variate, client_dict['control_variate'], client_dict["trainloader"], epochs, device)
            elif args["algorithm"] == "fedavg":
                client_dict['model']=train_fedavg(client_dict["model"], client_dict["trainloader"], epochs, device)
            elif args["algorithm"] == "fedprox":
                client_dict['model']=train_fedprox(client_dict["model"], client_dict["trainloader"], epochs, device, args['mu'])
            elif args["algorithm"] == "feddyn":
                client_dict['model'], client_dict['prev_grads']=train_feddyn(client_dict["model"], client_dict["trainloader"], epochs, device, client_dict['prev_grads'])
            else:
                client_dict['model']=train_model(client_dict["model"], client_dict["trainloader"], epochs, device)
            ##calculate time and log in time_df
            end_time = time.time()
            time_df.loc[round, f"client_{i}"] = end_time - start_time
            if carbon:
                ##calculate carbon emission and log in carbon_df
                emission = carbon_tracker.stop()
                carbon_df.loc[round, f"client_{i}"] = emission

        # save_model_states(client_dicts, round_path)
        ###aggregate the client models
        trained_models_state_dicts = [client_dict["model"].state_dict() for client_dict in client_dicts]

        if control_variate:
            updated_control_variates = [client_dict['delta_c'] for client_dict in client_dicts]
            trained_model_parameters, control_variate = aggregator.aggregate(state_dict,
                                            control_variate, trained_models_state_dicts, updated_control_variates)
    
        else:
            trained_model_parameters = aggregator.aggregate(state_dict,trained_models_state_dicts)

        state_dict = trained_model_parameters

        ###save the results for the round
        ###eval results can be calculated on any one client as all clients share the same model architecture and testset
        client_dicts[0]["model"].load_state_dict(state_dict)
        eval_loss, eval_accuracy = test_model(client_dicts[0]["model"], client_dicts[0]["testloader"], device)
        eval_result = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}
        print("Eval results: ", eval_result)
        eval_list.append(eval_result)
        #store the results
        with open(f"{round_path}/FL_results.txt", "a") as file:
            file.write( str(eval_result) + "\n" )

        ##check if the accuracy threshold is reached
        if eval_accuracy > accuracy_threshold:
            print("Accuracy threshold reached. Stopping training")
            break

    print("train eval")
    # eval_results = []
    # for client_dict in client_dicts:
    #     client_dict["model"].load_state_dict(trained_model_parameters)
    #     eval_loss, eval_accuracy = test_model(client_dict["model"], client_dict["testloader"])
    #     eval_results.append( {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy} )
    # response_dict = average(eval_results)

    ##eval results can be calculated on any one client as all clients share the same model and testset
    client_dicts[0]["model"].load_state_dict(state_dict)
    eval_loss, eval_accuracy = test_model(client_dicts[0]["model"], client_dicts[0]["testloader"], device)
    response_dict = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}
    response_dict_bytes = json.dumps(response_dict).encode("utf-8")

    ###apply algorithms for server level aggregation
    if config_dict['algorithm'] == "fedavg":
        state_dict = state_dict
    else:
        state_dict = fedadam(model_parameters, state_dict)

    data_to_send = {'model_parameters': state_dict, 'control_variate': control_variate_server}
    buffer = BytesIO()
    torch.save(data_to_send, buffer)
    buffer.seek(0)
    data_to_send_bytes = buffer.read()

    train_response_message = TrainResponse(
        modelParameters = data_to_send_bytes, 
        responseDict = response_dict_bytes)
    ##convert both the dataframes to csv and save them
    time_df.to_csv(f"{save_dir_path}/time.csv")
    carbon_df.to_csv(f"{save_dir_path}/carbon.csv")

    return train_response_message, client_dicts


## make a function to take values eval_list and plot it and save it
def plot_eval():

    accuracy = [eval_list[i]['eval_accuracy'] for i in range(len(eval_list))]
    loss = [eval_list[i]['eval_loss'] for i in range(len(eval_list))]
    x = np.arange(0, len(eval_list), 1)
    ##make to line plots for both accuracy and loss and save them
    plt.plot(x, accuracy)
    plt.xlabel('rounds')
    plt.ylabel('accuracy')
    plt.savefig(f"{save_dir_path}/accuracy.png")
    plt.clf()
    plt.plot(x, loss)
    plt.xlabel('rounds')
    plt.ylabel('loss')
    plt.savefig(f"{save_dir_path}/loss.png")
    plt.clf()


# #replace current model with the model provided
def set_parameters(set_parameters_order_message, client_dicts):
    plot_eval()
    model_parameters_bytes = set_parameters_order_message.modelParameters
    model_parameters = torch.load( BytesIO(model_parameters_bytes), map_location="cpu" )
    for client_dict in client_dicts:
        client_dict["model"].load_state_dict(model_parameters)
    save_model_states(client_dicts, save_dir_path)

#save the current state of the client_dicts
def save_model_states(client_dicts, path):
    save_num = len(os.listdir(path))
    save_path = f"{path}/model_states_{save_num}.pt"
    torch.save(client_dicts, save_path)

