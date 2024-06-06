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

import json
from pathlib import Path
import time
from datetime import datetime

from .config import Config
from .experiment import Experiment



BROKER_ADDR = "10.0.0.1"
# MIN_TRAINERS = 3
# NUM_ROUNDS = 10
# STOP_ACC = 1.0
# CSV_LOG="logs/novo.log"


class IpGen:
    def __init__(self,start_ip4) -> None:
        self.ip1 = 10
        self.ip2 = 0
        self.ip3 = 0
        self.ip4 = start_ip4

    def next_ip(self) -> str:
        ip = f"{self.ip1}.{self.ip2}.{self.ip3}.{self.ip4}"
        
        self.ip4 += 1
        if self.ip4 > 254:
            self.ip4 = 1
            self.ip3 += 1
            if self.ip3 > 255:
                self.ip3 = 0
                self.ip2+=1
                if self.ip2 > 255:
                    self.ip2 = 0
                    raise Exception("Error: Ip limit reached!")
    
    
        return ip
    
          

class FedNetwork:
    def __init__(self, filename):
        
        self.ip_gen = IpGen(3)
        self.switchs = list()
        self.clientes = list()
        
        self.volume = "/flw"
        self.docker_volume = [f"{Path.cwd()}:" + self.volume]
        
        setLogLevel('info')
        info('*** Importing configurations\n')
        
        self.config = Config(filename)
        
        
        self.exp_conf = self.config.get("experiment")
        self.general = self.config.get("general")
        self.net_conf = self.config.get("network")
        # self.absolute = self.general["absolute_path"]
        self.n_cpu = self.general["n_available_cpu"]
        self.broker_image = self.general["broker_image"]
        
        self.network_monitor_image = self.net_conf["network_monitor_image"]
        
        self.stop_acc = self.exp_conf["stop_accuracy"]
        self.max_n_rounds = self.exp_conf["max_n_rounds"]
        self.min_trainers = self.exp_conf["min_trainers"]
        
        self.server = self.config.get("server")
        self.server_quota = self.server["vCPU_percent"] * self.n_cpu * 1000
        
        self.server_images = self.server["image"]
        
        
        self.net = Containernet(controller=Controller)
        info('*** Adding controller\n')
        self.net.addController('c0')
        
        self.insert_switch(self.net_conf["network_components"])
        self.insert_broker_container()
        self.insert_monitor_container()
        self.insert_server_container()
        self.insert_client_containers()

        

        self.experiment = Experiment(self.general["experiments_folder"],self.exp_conf["experiment_name"],create_new=self.exp_conf["new_experiment"])
        self.experiment.copyFileToExperimentFolder(filename)

    def interrupt_execution(self):
        self.net.stop()

    def insert_switch(self, qtd):
        info('*** Adicionando SWITCHS\n')
        self.insert_switch
        for i in range(1, qtd + 1):
          self.switchs.append(self.net.addSwitch(f"s{i}"))
    
    
    
    def insert_broker_container(self):
        info('*** Adicionando Container do Broker\n')
        # broker container
        self.broker = self.net.addDocker('brk1', ip='10.0.0.1',dimage=self.broker_image, volumes=self.docker_volume)
        self.net.addLink(self.broker, self.switchs[self.server["connection"] - 1])
    
    
    def insert_monitor_container(self):
        info('*** Adicionando Container do monitor\n')
        self.mnt1 = self.net.addDocker('mnt1', ip='10.0.0.2',dimage=self.network_monitor_image, volumes=self.docker_volume)
        self.net.addLink(self.mnt1, self.switchs[self.server["connection"] - 1])
    
    def insert_server_container(self):
        info('*** Adicionando Container do Server\n')
        self.srv1 = self.net.addDocker('srv1', ip=self.ip_gen.next_ip(),dimage=self.server_images, volumes=self.docker_volume,
                     mem_limit=self.server["memory"], cpu_quota=self.server_quota)
        self.net.addLink(self.srv1, self.switchs[self.server["connection"] - 1])
        
    def insert_client_containers(self):
      info('*** Adicionando Container do Server\n')
     
      qtdDevice = 0
      for client_type in self.config.get("client_types"):
          for x in range(1, client_type["amount"]+1):    
            #   volumes = [f"{Path.cwd()}:" + client_type["volume"]]
                volumes = self.docker_volume
                qtdDevice += 1
                client_quota = client_type["vCPU_percent"] * self.n_cpu*1000
                d = self.net.addDocker(f'sta{client_type["name"]}{x}', ip=self.ip_gen.next_ip(),cpu_quota=client_quota,
                                    dimage=client_type["image"], volumes=volumes,  mem_limit=client_type["memory"])
                self.net.addLink(d, self.switchs[client_type['connection'] - 1],
                                cls=TCLink, delay=client_type.get("delay"), loss=client_type.get("loss"), bw=client_type.get("bw"))
                self.clientes.append(d)
              
    def start(self):
        info('*** Configurando Links\n')
        
        stop = self.net.addDocker('stop', dimage=self.network_monitor_image, volumes=self.docker_volume)
        self.net.addLink(stop, self.switchs[self.server["connection"] - 1]) 
        
        self.net.start()
      
        self.start_broker() 
        time.sleep(2)
        self.start_monitor()
        self.start_server() 
        time.sleep(3)
        self.start_clientes()
        
        info('*** Rodando CLI\n')
        stop.cmd(f'bash -c "cd {self.volume} && . env/bin/activate && python3 stop.py {BROKER_ADDR}"', verbose=True)
        # CLI(self.net)
        info('*** Parando MININET')
        self.net.stop()
        
          
    def start_broker(self):
        info('*** Inicializando broker\n')
        makeTerm(self.broker, cmd=f'bash -c "mosquitto -c {self.volume}/mosquitto.conf"')
        
    def start_monitor(self):
        info('*** Inicializando monitor\n')
        cmd2 = f"bash -c 'cd {self.volume} && . env/bin/activate && python3 {self.net_conf['''network_monitor_script''']} {BROKER_ADDR} {self.experiment.getFileName(extension='''''')}.net'"
        makeTerm(self.mnt1, cmd=cmd2)
        
        
    def start_server(self):
        info('*** Inicializando servidor\n')
        script = self.server["script"]
        vol = self.volume  
        cmd = f"""bash -c "cd {vol} && . env/bin/activate && python3 {script} {BROKER_ADDR} {self.min_trainers} {self.max_n_rounds} {self.stop_acc} {self.experiment.getFileName()} 2> {self.experiment.getFileName(extension='''''')}_err.txt """
        args = self.exp_conf.get("client_args")
        if args is not None:
            json_str = json.dumps(args).replace('"', '\\"')
            cmd += f"'{json_str}'"
        cmd += '" ;'
        # print(cmd)
        makeTerm(self.srv1, cmd=cmd)
        
        
    def start_clientes(self):
        info('*** Inicializando clientes\n')
        count = 0
        for client_type in self.config.get("client_types"):
            for x in range(1, client_type["amount"]+1):
                info(f"*** Subindo cliente {str(count+1).zfill(2)}\n")
                # vol = client_type["volume"]
                vol = self.volume
                cmd = f"bash -c 'cd {vol} && . env/bin/activate && python3 {client_type['script']} {BROKER_ADDR} {self.clientes[count].name} {count} {self.exp_conf['trainer_mode']}' ;"
                # print(cmd)
                makeTerm(self.clientes[count], cmd=cmd)
                count += 1