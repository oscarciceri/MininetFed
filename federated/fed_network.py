try:
    from containernet.net import Containernet
    from containernet.term import makeTerm
    from containernet.cli import CLI
    from containernet.link import TCLink
except:
    from mininet.net import Containernet
    from mininet.term import makeTerm
    from mininet.cli import CLI
    from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.node import Controller

from .ipgen import IPGenerator

try:
    from containernet.node import DockerP4Sensor, DockerSensor
except:
    pass


import json
import os
from pathlib import Path
import time
from datetime import datetime

from .config import Config
from .config import device_definition, link_definition
from .experiment import Experiment
from .external_broker import ExtBroker

AUTO_WAIT_IMAGE = "mininetfed:client"

NETWORT_IPV4 = '10.0.0.0/8'
NETWORT_IPV6 = 'fe80::/10'


class FedNetwork:
    def __init__(self, filename):

        self.ipv4_gen = IPGenerator(NETWORT_IPV4)
        self.ipv6_gen = IPGenerator(NETWORT_IPV6)

        self.devices = {}
        self.ext = None

        self.volume = "/flw"
        self.docker_volume = [f"{Path.cwd()}:" + self.volume]

        setLogLevel('info')
        info('*** Importing configurations\n')

        self.n_cpu = os.cpu_count()
        self.config = Config(filename)

        self.exp_conf = self.config.get("experiment")
        self.general = self.config.get("general")
        self.net_conf = self.config.get("network")

        # Obtendo configurações do experimento
        self.envs_folder = self.general.get("envs_folder")
        if self.envs_folder is None:
            self.envs_folder = "envs"
        self.stop_acc = self.exp_conf["control_metrics"]["stop_accuracy"]
        self.max_n_rounds = self.exp_conf["control_metrics"]["max_n_rounds"]
        self.min_trainers = self.exp_conf["control_metrics"]["min_trainers"]

        # Instanciando o containernet
        try:
            self.net = Containernet(
                iot_module='mac802154_hwsim', controller=Controller, ipBase='10.0.0.0/8')
        except:
            self.net = Containernet(controller=Controller, ipBase='10.0.0.0/8')

        info('*** Adding controller\n')
        self.net.addController('c0')

        self.insert_devices(self.config.get("devices"))
        self.link_devices(self.config.get("network_topology"))

        # Iniciando gerenciador de experimento
        self.experiment = Experiment(
            self.general["experiments_folder"], self.exp_conf["experiment_name"], create_new=self.exp_conf["auto_date_on_experiment_name"])
        self.experiment.copyFileToExperimentFolder(filename)

    def insert_devices(self, devices):

        info('*** Adding devices\n')
        for device in devices:
            device_def = device_definition(device)

            info(f'*** Adicionando {device_def["name"]}\n')
            instantiated_devices = {}
            for x in range(1, device_def["qtd"]+1):

                match device_def["type"]:
                    case "client_docker_host":
                        d = self.insert_docker_host(
                            device_def, device, f'{device_def["name"]}{x}')
                    case "server_docker_host":
                        d = self.insert_docker_host(
                            device_def, device, f'{device_def["name"]}{x}')
                    case "broker":
                        d = self.insert_broker(
                            device_def, device, f'{device_def["name"]}{x}')
                    case "network_monitor":
                        d = self.insert_monitor(
                            device_def, device, f'{device_def["name"]}{x}')
                    case "switch":
                        d = self.insert_switch(
                            device_def, device, f'{device_def["name"]}{x}')
                    case "DockerSensor":
                        d = self.insert_sensor(
                            device_def, device, f'{device_def["name"]}{x}')
                    case "DockerAPSensor":
                        d = self.insert_apsensor(
                            device_def, device, f'{device_def["name"]}{x}')
                    case _:
                        raise Exception(
                            f"""'{device_def["type"]}' is not a valid device type""")

                self.devices[f'{device_def["name"]}{x}'] = d
                instantiated_devices[f'{device_def["name"]}{x}'] = d

            self.devices[device_def["name"]] = instantiated_devices

    def insert_docker_host(self, device_def, device, name):
        info(f'\t*** Adicionando {name}\n')
        volumes = self.docker_volume
        client_quota = device["vCPU_percent"] * self.n_cpu*1000
        d = self.net.addDocker(name, ip=self.ipv4_gen.next_host_ip(name), cpu_quota=client_quota,
                               dimage=device["image"], volumes=volumes,  mem_limit=device["memory"])
        d.cmd("ifconfig eth0 down")
        return d

    def insert_sensor(self, device_def, device, name):
        info(f'\t*** Adicionando {name}\n')
        volumes = self.docker_volume
        client_quota = device["vCPU_percent"] * self.n_cpu*1000
        # cpu_shares=20,
        d = self.net.addSensor(name, ip6=self.ipv6_gen.next_host_ip(name), panid='0xbeef',
                               cls=DockerSensor, dimage=device["image"], cpu_quota=client_quota,
                               volumes=volumes,
                               environment={"DISPLAY": ":0"}, privileged=True, mem_limit=device["memory"])

        d.cmd("ifconfig eth0 down")
        return d

    def insert_apsensor(self, device_def, device, name):
        info(f'\t*** Adicionando {name}\n')
        volumes = self.docker_volume
        client_quota = device["vCPU_percent"] * self.n_cpu*1000
        args = device['args']
        mode = device['mode']
        dimage = device['image']

        d = self.net.addAPSensor(name, cls=DockerP4Sensor, ip6=self.ipv6_gen.next_host_ip(name), panid='0xbeef',
                                 dodag_root=True, storing_mode=mode, privileged=True,
                                 volumes=volumes,
                                 dimage=dimage, cpu_quota=client_quota, netcfg=True,
                                 environment={"DISPLAY": ":0"}, loglevel="info",
                                 thriftport=50001,  mem_limit=device["memory"], IPBASE="172.17.0.0/24", **args)  # IPBASE:##################### Decobobrir o que é esse IPBASE

        d.cmd("ifconfig eth0 down")
        return d

    def insert_broker(self, device_def, device, name):
        info(f'\t*** Adicionando {name}\n')
        volumes = self.docker_volume

        broker = self.net.addDocker(
            name, ip=self.ipv4_gen.next_host_ip(name), dimage=device["image"], volumes=volumes)
        broker.cmd(
            "iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE")

        if ((device.get("mode") is None) or device.get("mode") == 'internal'):
            self.broker_addr = self.ipv4_gen.get_host_ip(name)
        elif (device.get("external_broker_address") is not None):
            self.broker_addr = self.general.get("external_broker_address")
        else:
            raise Exception("external_broker_address needed!")

        return broker

    def insert_switch(self, device_def, device, name):
        info(f'\t*** Adicionando {name}\n')
        volumes = self.docker_volume

        d = self.net.addSwitch(name)
        return d

    def insert_monitor(self, device_def, device, name):
        info(f'\t*** Adicionando {name}\n')
        volumes = self.docker_volume
        d = self.net.addDocker(
            name, ip=self.ipv4_gen.next_host_ip(name), dimage=device["image"], volumes=volumes)
        d.cmd("ifconfig eth0 down")
        return d

    def link_devices(self, network_top):
        info(f'*** Adicionando Links\n')
        for link in network_top:
            link_def = link_definition(link)

            d1 = self.devices.get(link_def["d1"])
            d2 = self.devices.get(link_def["d2"])

            if d1 is None:
                raise Exception(
                    f'{link_def["d1"]} não é um device ou um tipo de device')
            if d2 is None:
                raise Exception(
                    f'{link_def["d2"]} não é um device ou um tipo de device')

            # Tratando tanto tipos de clientes quanto o cliente específico
            if isinstance(d1, dict) is True:
                d1_type = d1
            else:
                d1_type = {link_def["d1"]: d1}

            if isinstance(d2, dict) is True:
                d2_type = d2
            else:
                d2_type = {link_def["d2"]: d2}

            for k1 in d1_type:
                for k2 in d2_type:
                    d1 = d1_type[k1]
                    d2 = d2_type[k2]
                    info(f'\t*** Link {link_def["type"]} entre {k1} e {k2}: ')
                    match link_def["type"]:
                        case "TCLink":
                            self.link_TCLink(link_def, link, d1, d2)
                        case "Default":
                            self.std_link(link_def, link, d1, d2)
                    info('\n')

    def std_link(self, link_def, link, d1, d2):
        self.net.addLink(d1, d2)

    def link_TCLink(self, link_def, link, d1, d2):

        self.net.addLink(d1, d2,
                         cls=TCLink, delay=link.get("delay"), loss=link.get("loss"), bw=link.get("bw"))

    def auto_wait(self, verbose=False):
        self.stop.cmd(
            f'bash -c "cd {self.volume} && . {self.envs_folder}/{self.general.get("auto_wait_env")}/bin/activate && python3 stop.py {self.broker_addr}"', verbose=verbose)

    def insert_stop(self):
        self.stop = self.net.addDocker(
            'stop', ip="10.254.255.255", dimage=AUTO_WAIT_IMAGE, volumes=self.docker_volume)
        self.net.addLink(
            self.stop, self.devices["h1"])  # --------------------------------------- MUDAR ISSO
        self.stop.cmd("ifconfig eth0 down")

    def start(self):
        info('*** Iniciando execução da rede\n')
        self.insert_stop()
        self.net.start()
        self.stop.cmd("route add default gw %s" % self.broker_addr)

        info('*** Iniciando dispositivos\n')
        self.start_devices(self.config.get("devices"))

        if ((self.general.get("stop") is not None) and self.general.get("stop") == 'cli'):
            info('*** Rodando CLI\n')
            CLI(self.net)
        else:
            info('*** Esperando encerramento do experimento\n')
            self.auto_wait(verbose=True)

        self.interrupt_execution()

    def start_devices(self, devices):
        for device in devices:
            device_def = device_definition(device)
            info(f'*** Iniciando {device_def["name"]}\n')

            for d_key in self.devices[device_def["name"]]:
                d = self.devices[d_key]
                info(f'\t*** Iniciando {d_key}\n')
                match device_def["type"]:
                    case "client_docker_host":
                        self.start_clientes(device_def, device, d, d_key)
                    case "server_docker_host":
                        self.start_server(device_def, device, d)
                    case "broker":
                        self.start_broker(device_def, device, d)
                    case "network_monitor":
                        self.start_monitor(device_def, device, d)
                    case "switch":
                        pass
                    case "DockerSensor":
                        pass
                    case "DockerAPSensor":
                        pass
                    case _:
                        raise Exception(
                            f"""'{device_def["type"]}' is not a valid device type""")

    def start_broker(self, device_def, device, broker):
        if ((device.get("mode") is None) or device.get("mode") == 'internal'):
            makeTerm(
                broker, cmd=f'bash -c "mosquitto -c {self.volume}/mosquitto.conf"')
        elif (device.get("mode") == 'external'):
            self.ext = ExtBroker()
            self.ext.run_ext_brk()
        else:
            raise Exception(
                f'Invalid broker mode: {device.get("mode")}')
        time.sleep(2)

    def start_monitor(self, device_def, device, monitor):
        cmd2 = f"bash -c 'cd {self.volume} && . {self.envs_folder}/{device['''env''']}/bin/activate && python3 {device['''script''']} {self.broker_addr} {self.experiment.getFileName(extension='''''')}.net'"
        monitor.cmd("route add default gw %s" % self.broker_addr)
        makeTerm(monitor, cmd=cmd2)

    def start_server(self, device_def, device, server):
        print(self.broker_addr)
        script = device["script"]
        vol = self.volume
        cmd = f"""bash -c "cd {vol} && . {self.envs_folder}/{device['''env''']}/bin/activate && python3 {script} {self.broker_addr} {self.min_trainers} {self.max_n_rounds} {self.stop_acc} {self.experiment.getFileName()} 2> {self.experiment.getFileName(extension='''''')}_err.txt """
        args = device.get("server_client_args")
        if args is not None:
            json_str = json.dumps(args).replace('"', '\\"')
            cmd += f"'{json_str}'"
        cmd += '" ;'
        server.cmd("route add default gw %s" % self.broker_addr, verbose=True)
        makeTerm(server, cmd=cmd)

        self.auto_wait(verbose=True)

    def start_clientes(self, device_def, device, cliente, client_name):

        if hasattr(self, 'count') is not True:
            self.count = 0

        args = device.get('args')
        json_str = None
        if args is not None:
            json_str = json.dumps(args).replace('"', '\\"')

        vol = self.volume
        if device['''env'''] is None:
            raise Exception(
                f"Env para Cliente {str(self.count+1).zfill(2)} não determinado")

        cmd = f"""bash -c "cd {vol} && . {self.envs_folder}/{device['''env''']}/bin/activate && python3 {device['''script''']} {self.broker_addr} {client_name} {self.count} {device['''trainer_mode''']} 2> client_log/{client_name}.txt """
        # print(cmd)
        if json_str is not None:
            cmd += f"'{json_str}'"
        cmd += '" ;'
        cliente.cmd(
            "route add default gw %s" % self.broker_addr)
        makeTerm(cliente, cmd=cmd)

        self.count += 1

    def interrupt_execution(self):
        info('*** Parando MININET')
        if self.ext is not None:
            self.ext.stop_ext_brk()
        self.net.stop()
