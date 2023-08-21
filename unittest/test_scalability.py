import os
import sys
import unittest
# add main directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fed2tier.server.src.server_lib import save_intial_model
from misc import get_config, tester

def create_train_test_for_10_nodes():
    """ Verify the node level fedavg algorithm using one node
    by implementing the following function """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_scalability', '10_nodes')
            save_intial_model(config['server'])

        def test_LeNet(self):
            print("\n==Fed Avg==")
            config = get_config('test_scalability', '10_nodes')
            tester(config, 10)
    return TrainerTest

class TestTrainer_10_nodes(create_train_test_for_10_nodes()):
    'Test case for 10 nodes'

if __name__ == '__main__':
    unittest.main()
    