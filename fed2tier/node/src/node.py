
import json
from queue import Queue
import torch
from io import BytesIO
import time

import grpc
from . import ClientConnection_pb2_grpc
from .ClientConnection_pb2 import ClientMessage

from .node_lib import train, set_parameters

#start the client and connect to server
def node_start(config):
    keep_going = True
    wait_time = config["wait_time"]
    device = torch.device(config["device"])
    #device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    while keep_going:
        #wait for specified time before reconnecting
        time.sleep(wait_time)
        
        #create new gRPC channel to the server
        with grpc.insecure_channel('localhost:8213', options=[
                ('grpc.max_send_message_length', -1),
                ('grpc.max_receive_message_length', -1)
                ]) as channel:
            stub = ClientConnection_pb2_grpc.ClientConnectionStub(channel)
            client_buffer = Queue(maxsize = 1)
            client_dicts=None
            #wait for incoming messages from the server in client_buffer
            #then according to fields present in them call the appropraite function
            for server_message in stub.Connect( iter(client_buffer.get, None) ):
                
                if server_message.HasField("trainOrder"):
                    train_order_message = server_message.trainOrder
                    train_response_message, client_dicts = train(train_order_message, device, config)
                    message_to_server = ClientMessage(trainResponse = train_response_message)
                    client_buffer.put(message_to_server)

                if server_message.HasField("setParamsOrder"):
                    set_parameters_order_message = server_message.setParamsOrder
                    set_parameters_response_message = set_parameters(set_parameters_order_message, client_dicts)
                    message_to_server = ClientMessage(setParamsResponse = set_parameters_response_message)
                    client_buffer.put(message_to_server)
                    print("parameters successfuly set")

                if server_message.HasField("disconnectOrder"):
                    print("recieve disconnect order")
                    disconnect_order_message = server_message.disconnectOrder
                    message = disconnect_order_message.message
                    print(message)
                    reconnect_time = disconnect_order_message.reconnectTime
                    if reconnect_time == 0:
                        keep_going = False
                        break
                    wait_time = reconnect_time
