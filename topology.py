import sys


from pathlib import Path
from time import sleep

from mininet.log import info, setLogLevel

from federated.net import MininetFed
from federated.node import Server, Client


volume = "/flw"
# volumes = [f"{Path.cwd()}:" + volume, "/tmp/.X11-unix:/tmp/.X11-unix:rw"]

volumes = [
    f"{Path.cwd()}:" + volume,
    "/tmp/.X11-unix:/tmp/.X11-unix:rw",
    "/home/user/INESC_TEC/MininetFed/temp/ckksfed_fhe/pasta:/home/user/INESC_TEC/MininetFed/temp/ckksfed_fhe/pasta"
]

experiment_config = {
    "ipBase": "10.0.0.0/24",
    "experiments_folder": "experiments",
    "experiment_name": "test_07",
    "date_prefix": False
}

# server_args = {"min_trainers": 4, "num_rounds": 3,
#                "stop_acc": 0.999, 'client_selector': 'All', 'aggregator': "FedAvg"}
# client_args = {"mode": 'random same_samples',
#                'num_samples': 15000, "trainer_class": "TrainerMNIST"}


server_args = {
    "min_trainers": 4,
    "num_rounds": 3,
    "stop_acc": 0.999,
    "client_selector": "All",
    "aggregator": "Ckksfed"
}

client_args = {
    "mode": 'random same_samples',
    "num_samples": 15000,
    "trainer_class": "TrainerCkksfed",
    "encrypted": True,
    "n_clusters": 2,  
}

def topology():
    net = MininetFed(
        **experiment_config,
        controller=[],
        default_volumes=volumes,
        topology_file=sys.argv[0]
    )

    info('*** Adding Nodes...\n')
    s1 = net.addSwitch("s1", failMode='standalone')

    net.configureMininetFedInternalDevices()

    srv1 = net.addHost('srv1', cls=Server, script="server/server.py",
                       args=server_args, volumes=volumes,
                       dimage='mininetfed:server',
                       )
    
    clients = []
    for i in range(server_args["min_trainers"]):
        clients.append(net.addHost(f'sta{i}', cls=Client, script="client/client.py",
                                   args=client_args, volumes=volumes,
                                   dimage='mininetfed:client',
                                   numeric_id=i
                                   )
                       )

    info('*** Creating links...\n')
    net.connectMininetFedInternalDevices()
    net.addLink(srv1, s1)
    for client in clients:
        net.addLink(client, s1)

    info('*** Starting network...\n')
    net.build()
    net.addNAT(name='nat0', linkTo='s1', ip='192.168.210.254').configDefault()
    s1.start([])

    info('*** Running devices...\n')

    info('*** Running FL internal devices...\n')
    net.runFlDevices()

    srv1.run(broker_addr=net.broker_addr,
             experiment_controller=net.experiment_controller)

    sleep(3)
    for client in clients:
        client.run(broker_addr=net.broker_addr,
                   experiment_controller=net.experiment_controller)

    info('*** Running Autostop...\n')
    net.wait_experiment(start_cli=True)

    # os.system('pkill -9 -f xterm')

    info('*** Stopping network...\n')
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    topology()
