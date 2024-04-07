import numpy as np
from functools import reduce

class FedSketchAgg:
      
    def __init__(self):
      pass
    
    def aggregate(self, all_trainer_samples, weights):
        """Compute weighted average."""
        num_clients = len(all_trainer_samples)
        print(num_clients)
        # Calculate the total number of examples used during training
        #num_examples_total = 0
        #for i in results:
        #    num_examples_total += i[1]

        # Create a list of weights, each multiplied by the related number of examples
        #weighted_weights = [
        #    [layer * num_examples for layer in weights] for weights, num_examples in results
        #]

        # Compute average weights of each layer
        weights_prime = [
            reduce(np.add, layer_updates) / num_clients
            for layer_updates in zip(*weights)
        ]
        return weights_prime