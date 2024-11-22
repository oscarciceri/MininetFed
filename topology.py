#!/usr/bin/python
"""
This topology is used to test the compatibility of different Docker images.
The images to be tested can be found in 'examples/example-containers'.
They are build with './build.sh'
"""
from mininet.node import Controller
from mininet.log import info, setLogLevel
from pathlib import Path

from federated.node import Server, Client
from federated.net import MininetFed

from containernet.energy import Energy

volume = "/flw"
volumes = [f"{Path.cwd()}:" + volume]


setLogLevel('info')
info('*** Configuring MininetFed basic topology\n')

net = MininetFed(controller=Controller, experiment_name="teste_topology",
                 experiments_folder="experiments", date_prefix=False, default_volumes=volumes)
# info('*** Adding controller\n')
# net.addController('c0')

info('*** Adding docker containers\n')


args = {"min_trainers": 4, "num_rounds": 40, "stop_acc": 0.95}
client_args = None

srv1 = net.addPriorityHost('srv1', cls=Server, script="server/server.py", env="../env", args=args, volumes=volumes,
                           dimage="mininetfed:server")


sta1 = net.addHost('sta1', cls=Client, script="client/client.py", env="../env", numeric_id=0, args=client_args, volumes=volumes,
                   dimage="mininetfed:client")
sta2 = net.addHost('sta2', cls=Client, script="client/client.py", env="../env", numeric_id=1, args=client_args, volumes=volumes,
                   dimage="mininetfed:client")
sta3 = net.addHost('sta3', cls=Client, script="client/client.py", env="../env", numeric_id=2, args=client_args, volumes=volumes,
                   dimage="mininetfed:client")
sta4 = net.addHost('sta4', cls=Client, script="client/client.py", env="../env", numeric_id=3, args=client_args, volumes=volumes,
                   dimage="mininetfed:client")


info('*** Adding switches\n')
s1 = net.addSwitch('s1')

info('*** Creating links\n')
net.addLink(srv1, s1)
net.addLink(net.brk, s1)
net.addLink(net.auto_stop, s1)
net.addLink(net.mnt, s1)

net.addLink(sta1, s1)
net.addLink(sta2, s1)
net.addLink(sta3, s1)
net.addLink(sta4, s1)


info('*** Starting network\n')
net.start(start_cli=False)


info('*** Stopping network')
net.stop()
