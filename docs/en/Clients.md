# Default Client

## Switching Between Trainers

In the client provided with MiniNetFED, one of the features is the ability to choose between example-provided trainers or implement your own trainer for experiments.

To switch between trainers, access the following file:

```
./client/trainer/__init__.py
```

Edit the name of the file from which you want to import the Trainer and the name of the implemented class.

Example:

```python
from .trainerhar import TrainerHar as Trainer
```

**Important Notes:** Do not change the **as Trainer**. This ensures that, for any chosen trainer, other client components will recognize it correctly.

Also note that, as shown, new trainers should be contained in the /client/trainer directory.

## Implementing New Trainers

To create a custom trainer, it is recommended to use one of the example-provided trainers as a base and modify its model, dataset, and data manipulations as desired.

For MiniNetFED to recognize the trainer as valid, at least the following methods should be implemented in the created class:

```python
def __init__(self, ext_id, mode) -> None:
    """
    Initializes the Trainer object with an external ID and operation mode.
    """

def set_args(self, args):
    """
    Defines arguments for the Trainer object when passed through the config.yaml configuration file.
    """

def get_num_samples(self):
    """
    Returns the number of training data samples.
    """

def split_data(self):
    """
    Loads the data and divides it into training and test sets.
    Returns the training and test data in the following format:

    return x_train, y_train, x_test, y_test
    """

def train_model(self):
    """
    Trains the model on the training data.
    """

def eval_model(self):
    """
    Evaluates the model on the test data.
    Returns the model's accuracy as a value between 0 and 1.
    """

def all_metrics(self):
    """
    Evaluates the model on the test data.
    Returns a dictionary of all metrics used by the model.
    """

def get_weights(self):
    """
    Returns the model weights. Can be in any format, provided it is consistent with the chosen aggregation function and update_weights implementation.
    """

def update_weights(self, weights):
    """
    Updates the model weights with the given weights. Can be in any format, provided it is consistent with the chosen aggregation function and get_weights implementation.
    """

def set_stop_true(self):
    """
    Sets the TrainerHar stop flag to True.
    """
    self.stop_flag = True

def get_stop_flag(self):
    """
    Returns the TrainerHar stop flag.
    """
    return self.stop_flag
```