
from containernet.net import Containernet
from containernet.cli import CLI

from ..node import Broker, Monitor, AutoStop
from ..experiment import Experiment


class MininetFed(Containernet):

    def __init__(self, experiment_name,
                 experiments_folder, default_volumes, date_prefix=False, broker_mode="internal", ext_broker_ip='127.0.0.1', **kwargs):

        self.nodes = []
        self.priority_nodes = []
        self.default_volumes = default_volumes
        self.experiment_controller = Experiment(experiment_name=experiment_name,
                                                experiments_folder=experiments_folder, create_new=date_prefix)
        Containernet.__init__(self, **kwargs)

        self.brk = super().addHost('brk', cls=Broker, mode=broker_mode, ext_broker_ip=ext_broker_ip, volumes=default_volumes,
                                   dimage="mininetfed:broker")

        self.mnt = super().addHost('mnt', cls=Monitor, env='../env', script="network_monitor.py",
                                   experiment_controller=self.experiment_controller, volumes=default_volumes)

        self.auto_stop = super().addHost('stop', cls=AutoStop, env='../env',
                                         volumes=default_volumes)

    def addHost(self, name, cls=None, **params):
        self.nodes.append(super().addHost(name, cls, **params))
        return self.nodes[-1]

    def addPriorityHost(self, name, cls=None, **params):
        self.priority_nodes.append(super().addHost(name, cls, **params))
        return self.priority_nodes[-1]

    def start(self, start_cli=False):
        super().start()
        self.brk.start()

        self.broker_addr = self.brk.IP(intf="brk-eth0")

        self.auto_stop.start(broker_addr=self.broker_addr)
        self.mnt.start(broker_addr=self.broker_addr)

        for node in self.priority_nodes:
            node.start(broker_addr=self.broker_addr,
                       experiment_controller=self.experiment_controller)

        self.auto_stop.auto_stop()

        for node in self.nodes:
            node.start(broker_addr=self.broker_addr,
                       experiment_controller=self.experiment_controller)

        if start_cli:
            CLI(self)
        else:
            self.auto_stop.auto_stop()
