#!/usr/bin/python
"""
This topology is used to test the compatibility of different Docker images.
The images to be tested can be found in 'examples/example-containers'.
They are build with './build.sh'
"""
from mininet.node import Controller
from mininet.log import info, setLogLevel
from containernet.cli import CLI
from containernet.net import Containernet
from pathlib import Path

from federated.node import Broker, Server, AutoStop, Monitor, Client
from federated.experiment import Experiment


volume = "/flw"
volumes = [f"{Path.cwd()}:" + volume]


setLogLevel('info')

net = Containernet(controller=Controller)
info('*** Adding controller\n')
net.addController('c0')
experiment = Experiment(experiment_name="teste_topology",
                        experiments_folder="experiments", create_new=False)

info('*** Adding docker containers\n')


brk = net.addHost('brk', cls=Broker, mode="internal", ext_broker_ip='127.0.0.1', volumes=volumes,
                  dimage="mininetfed:broker")

mnt = net.addHost('mnt', cls=Monitor, env='../env', script="network_monitor.py", broker_addr="10.0.0.1",
                  experiment_controller=experiment, volumes=volumes)

stop = net.addHost('stop', cls=AutoStop, env='../env',
                   ip="10.255.255.200", broker_addr="10.0.0.1", volumes=volumes)

args = {"min_trainers": 4, "num_rounds": 40, "stop_acc": 0.95}

# print(brk.IP)
srv1 = net.addHost('srv1', cls=Server, script="server/server.py", broker_addr="10.0.0.1", env="../env", experiment_controller=experiment, args=args, volumes=volumes,
                   dimage="mininetfed:server")


sta1 = net.addHost('sta1', cls=Client, script="client/client.py", broker_addr="10.0.0.1", env="../env", numeric_id=0, experiment_controller=experiment, args=args, volumes=volumes,
                   dimage="mininetfed:client")
sta2 = net.addHost('sta2', cls=Client, script="client/client.py", broker_addr="10.0.0.1", env="../env", numeric_id=1, experiment_controller=experiment, args=args, volumes=volumes,
                   dimage="mininetfed:client")
sta3 = net.addHost('sta3', cls=Client, script="client/client.py", broker_addr="10.0.0.1", env="../env", numeric_id=2, experiment_controller=experiment, args=args, volumes=volumes,
                   dimage="mininetfed:client")
sta4 = net.addHost('sta4', cls=Client, script="client/client.py", broker_addr="10.0.0.1", env="../env", numeric_id=3, experiment_controller=experiment, args=args, volumes=volumes,
                   dimage="mininetfed:client")


info('*** Adding switches\n')
s1 = net.addSwitch('s1')

info('*** Creating links\n')
net.addLink(srv1, s1)
net.addLink(brk, s1)
net.addLink(stop, s1)
net.addLink(mnt, s1)

net.addLink(sta1, s1)
net.addLink(sta2, s1)
net.addLink(sta3, s1)
net.addLink(sta4, s1)


info('*** Starting network\n')
net.start()
stop.start()
brk.start()
mnt.start()
srv1.start()
stop.auto_stop()

sta1.start()
sta2.start()
sta3.start()
sta4.start()

# info('*** Testing connectivity\n')
# net.ping([brk, srv1])


info('*** Running CLI\n')
# CLI(net)
stop.auto_stop()

info('*** Stopping network')
net.stop()
