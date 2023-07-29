from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
import lerconfig as cfg

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average)


fl.server.start_server(
    server_address="0.0.0.0:8083",

    config=fl.server.ServerConfig(num_rounds=200),
    strategy=strategy,
)