import os
import sys
import unittest
# add main directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fed2tier.server.src.server_lib import save_intial_model
from misc import get_config, tester

def create_train_test_for_fedavg():
    """ Verify the node level fedavg algorithm using one node
    by implementing the following function """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_node_algorithms', 'fedavg')
            save_intial_model(config['server'])

        def test_LeNet(self):
            print("\n==Fed Avg==")
            config = get_config('test_node_algorithms', 'fedavg')
            tester(config, 1)
    return TrainerTest

def create_train_test_for_fedprox():
    """ Verify the node level fedprox algorithm using one node
    by implementing the following function """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_node_algorithms', 'fedprox')
            save_intial_model(config['server'])

        def test_LeNet(self):
            print("\n==Fed Prox==")
            config = get_config('test_node_algorithms', 'fedprox')
            tester(config, 1)
    return TrainerTest

def create_train_test_for_fedadagrad():
    """ Verify the node level fedadagrad algorithm using one node
    by implementing the following function """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_node_algorithms', 'fedadagrad')
            save_intial_model(config['server'])

        def test_LeNet(self):
            print("\n==Fed Adagrad==")
            config = get_config('test_node_algorithms', 'fedadagrad')
            tester(config, 1)
    return TrainerTest

def create_train_test_for_fedadam():
    """ Verify the node level fedadam algorithm using one node
    by implementing the following function """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_node_algorithms', 'fedadam')
            save_intial_model(config['server'])

        def test_LeNet(self):
            print("\n==Fed Adam==")
            config = get_config('test_node_algorithms', 'fedadam')
            tester(config, 1)
    return TrainerTest

def create_train_test_for_feddyn():
    """ Verify the node level feddyn algorithm using one node
    by implementing the following function """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_node_algorithms', 'feddyn')
            save_intial_model(config['server'])

        def test_LeNet(self):
            print("\n==Fed Dyn==")
            config = get_config('test_node_algorithms', 'feddyn')
            tester(config, 1)
    return TrainerTest

def create_train_test_for_fedyogi():
    """ Verify the node level fedyogi algorithm using one node
    by implementing the following function """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_node_algorithms', 'fedyogi')
            save_intial_model(config['server'])

        def test_LeNet(self):
            print("\n==Fed Yogi==")
            config = get_config('test_node_algorithms', 'fedyogi')
            tester(config, 1)
    return TrainerTest

def create_train_test_for_scaffold():
    """ Verify the node level scaffold algorithm using one node
    by implementing the following function """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_node_algorithms', 'scaffold')
            save_intial_model(config['server'])

        def test_LeNet(self):
            print("\n==Scaffold==")
            config = get_config('test_node_algorithms', 'scaffold')
            tester(config, 1)
    return TrainerTest


class TestTrainer_fedavg(create_train_test_for_fedavg()):
    'Test case for fedavg algorithm at node level'

class TestTrainer_fedprox(create_train_test_for_fedprox()):
    'Test case for fedprox algorithm at node level'

class TestTrainer_fedadagrad(create_train_test_for_fedadagrad()):
    'Test case for fedadagrad algorithm at node level'

class TestTrainer_fedadam(create_train_test_for_fedadam()):
    'Test case for fedadam algorithm at node level'

class TestTrainer_feddyn(create_train_test_for_feddyn()):
    'Test case for feddyn algorithm at node level'

class TestTrainer_fedyogi(create_train_test_for_fedyogi()):
    'Test case for fedyogi algorithm at node level'

class TestTrainer_scaffold(create_train_test_for_scaffold()):
    'Test case for scaffold algorithm at node level'

if __name__ == '__main__':
    unittest.main()
