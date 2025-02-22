import sys
from pathlib import Path
from time import sleep

from mininet.log import info, setLogLevel

from federated.net import MininetFed
from federated.node import Server, Client


volume = "/flw"
volumes = [f"{Path.cwd()}:" + volume, "/tmp/.X11-unix:/tmp/.X11-unix:rw"]

experiment_config = {
    "ipBase": "10.0.0.0/24",
    "experiments_folder": "experiments",
    "experiment_name": "ipv4_test",
    "date_prefix": False
}

server_args = {"min_trainers": 8, "num_rounds": 1,
               "stop_acc": 0.999, 'client_selector': 'All'}
client_args = {"mode": 'random same_samples',
               'num_samples': 15000}


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
                       env="../env"
                       )

    clients = []
    for i in range(8):
        clients.append(net.addHost(f'sta{i}', cls=Client, script="client/client.py",
                                   args=client_args, volumes=volumes,
                                   dimage='mininetfed:client',
                                   env="../env",
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

    # net.wait_experiment()
    sleep(3)
    for client in clients:
        client.run(broker_addr=net.broker_addr,
                   experiment_controller=net.experiment_controller)

    info('*** Running Autostop...\n')
    net.wait_experiment(start_cli=False)

    # os.system('pkill -9 -f xterm')

    info('*** Stopping network...\n')
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    topology()
