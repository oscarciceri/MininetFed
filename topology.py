import os
import sys
from pathlib import Path
from time import sleep

from containernet.node import DockerP4Sensor, DockerSensor
from containernet.cli import CLI
from mininet.log import info, setLogLevel
from mn_wifi.sixLoWPAN.link import LoWPAN
from mininet.term import makeTerm
from containernet.energy import Energy
from mn_wifi.energy import BitZigBeeEnergy
# from mn_wifi.bitEnergy import BitEnergy

from federated.net import MininetFed
from federated.node import Server, Client


volume = "/flw"
volumes = [f"{Path.cwd()}:" + volume, "/tmp/.X11-unix:/tmp/.X11-unix:rw"]

server_args = {"min_trainers": 8, "num_rounds": 20,
               "stop_acc": 0.99, 'client_selector': 'All'}
client_args = {"mode": 'random_same', 'num_samples': 15000}


def topology():
    net = MininetFed(ipBase='10.0.0.0/24',
                     #  iot_module='mac802154_hwsim',
                     controller=[], experiment_name="ipv4_test",
                     experiments_folder="experiments", date_prefix=False, default_volumes=volumes, topology_file=sys.argv[0])

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
        clients.append(net.addHost(f'sta{i}', cls=Client, script="server/server.py",
                                   args=server_args, volumes=volumes,
                                   dimage='mininetfed:server',
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
    # ap1.cmd("nohup mosquitto -c /etc/mosquitto/mosquitto.conf &")
    # makeTerm(
    #     ap1, cmd="bash -c 'tail -f /var/log/mosquitto/mosquitto.log'")

    # sleep(2)
    # broker_addr = 'fd3c:be8a:173f:8e80::1'

    # info('*** Server...\n')
    # srv1.run(broker_addr=broker_addr,
    #          experiment_controller=net.experiment_controller, args=server_args)
    # # sleep(1)
    # # srv1.auto_stop()
    # # net.wait_experiment(broker_addr=broker_addr)
    # sleep(5)

    # info('*** Clients...\n')
    # for client in clients:
    #     client.run(broker_addr=broker_addr,
    #                experiment_controller=net.experiment_controller, args=client_args)
    # client2.run(broker_addr=broker_addr,
    #             experiment_controller=net.experiment_controller, args=client_args)
    # client3.run(broker_addr=broker_addr,
    #             experiment_controller=net.experiment_controller, args=client_args)
    # client4.run(broker_addr=broker_addr,
    #             experiment_controller=net.experiment_controller, args=client_args)
    # net.runFlDevices()
    # makeTerm(h1, title='grafana-server', cmd="bash -c 'grafana-server;'")
    # if '-s' in sys.argv:
    #     makeTerm(h1, title='h1',
    #              cmd="bash -c 'httpd && python /root/packet-processing-storing.py;'")
    # else:
    #     makeTerm(h1, title='h1',
    #              cmd="bash -c 'httpd && python /root/packet-processing-non-storing.py;'")

    # h1.cmd("ifconfig h1-eth1 down")

    # info('*** Running CLI...\n')
    # CLI(net)
    info('*** Running Autostop...\n')
    # srv1.auto_stop()
    net.wait_experiment(start_cli=True)

    # os.system('pkill -9 -f xterm')

    info('*** Stopping network...\n')
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    topology()
