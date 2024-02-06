from containernet.cli import CLI
from containernet.net import Containernet
from containernet.term import makeTerm
from mininet.log import info, setLogLevel
from mininet.node import Controller

from pathlib import Path
import time
from datetime import datetime

from .config import Config
from .experiment import Experiment



BROKER_ADDR = "172.17.0.2"
# MIN_TRAINERS = 3
# NUM_ROUNDS = 10
# STOP_ACC = 1.0
# CSV_LOG="logs/novo.log"
      

class FedNetwork:
    def __init__(self, filename):
        self.switchs = list()
        self.clientes = list()
        
        self.broker_mem = "128m"
        
        setLogLevel('info')
        info('*** Importing configurations\n')
        
        self.config = Config(filename)
        
        
        self.general = self.config.get("general")
        self.absolute = self.general["absolute_path"]
        self.n_cpu = self.general["n_available_cpu"]
        self.broker_image = self.general["broker_image"]
        
       
        
        self.stop_acc = self.general["stop_accuracy"]
        self.max_n_rounds = self.general["max_n_rounds"]
        self.min_trainers = self.general["min_trainers"]
        
        self.server = self.config.get("server")
        self.server_quota = self.server["vCPU_percent"] * self.n_cpu * 1000
        self.server_docker_volume = [f"{Path.cwd()}:" + self.server["volume"]]
        self.server_images = self.server["image"]
        
        
        self.net = Containernet(controller=Controller)
        info('*** Adding controller\n')
        self.net.addController('c0')
        
        self.insert_switch(self.config.get("network_components"))
        self.insert_broker_container()
        self.insert_monitor_container()
        self.insert_server_container()
        self.insert_client_containers()

        

        self.experiment = Experiment("experiments",self.general["experiment_name"],create_new=self.general["new_experiment"],reopen_name=self.general["reopen_name"])
        self.experiment.copyFileToExperimentFolder(filename)

    def insert_switch(self, qtd):
        info('*** Adicionando SWITCHS\n')
        self.insert_switch
        for i in range(1, qtd + 1):
          self.switchs.append(self.net.addSwitch(f"s{i}"))
    
    
    
    def insert_broker_container(self):
        info('*** Adicionando Container do Broker\n')
        # broker container
        self.broker = self.net.addDocker('brk1', dimage=self.broker_image,
                              volumes=self.server_docker_volume,  mem_limit=self.broker_mem)
        self.net.addLink(self.broker, self.switchs[self.server["conection"] - 1])
    
    
    def insert_monitor_container(self):
        info('*** Adicionando Container do monitor\n')
        self.mnt1 = self.net.addDocker('mnt1', dimage=self.server_images, volumes=self.server_docker_volume,
                     mem_limit=self.server["memory"], cpu_quota=self.server_quota)
        self.net.addLink(self.mnt1, self.switchs[self.server["conection"] - 1])
    
    def insert_server_container(self):
        info('*** Adicionando Container do Server\n')
        self.srv1 = self.net.addDocker('srv1', dimage=self.server_images, volumes=self.server_docker_volume,
                     mem_limit=self.server["memory"], cpu_quota=self.server_quota)
        self.net.addLink(self.srv1, self.switchs[self.server["conection"] - 1])
        
    def insert_client_containers(self):
      info('*** Adicionando Container do Server\n')
     
      qtdDevice = 0
      for client_type in self.config.get("client_types"):
          for x in range(1, client_type["amount"]+1):    
              volumes = [f"{Path.cwd()}:" + client_type["volume"]]
              qtdDevice += 1
              client_quota = client_type["vCPU_percent"] * self.n_cpu*1000
              d = self.net.addDocker(f'sta{client_type["name"]}{x}', cpu_quota=client_quota,
                                dimage=client_type["image"], volumes=volumes,  mem_limit=client_type["memory"])
              self.net.addLink(d, self.switchs[client_type['conection'] - 1],
                          loss=client_type["loss"], bw=client_type["bw"])
              self.clientes.append(d)
              
    def start(self):
        info('*** Configurando Links\n')
        
        self.net.start()
      
        self.start_broker() 
        time.sleep(2)
        self.start_monitor()
        self.start_server() 
        time.sleep(3)
        self.start_clientes()
        
        info('*** Rodando CLI\n')
        CLI(self.net)
        info('*** Parando MININET')
        self.net.stop()
        
          
    def start_broker(self):
        info('*** Inicializando broker\n')
        makeTerm(self.broker, cmd=f'bash -c "mosquitto -c {self.general["broker_volume"]}/mosquitto.conf"')
        
    def start_monitor(self):
        info('*** Inicializando monitor\n')
        cmd2 = f"bash -c 'cd {self.server['''volume''']} && . env/bin/activate && python3 network_monitor.py {BROKER_ADDR} {self.experiment.getFileName(extension='''''')}.net'"
        makeTerm(self.mnt1, cmd=cmd2)
        
        
    def start_server(self):
        info('*** Inicializando servidor\n')
        script = self.server["script"]
        vol = self.server["volume"]
        cmd = f"bash -c 'cd {vol} && . env/bin/activate && python3 {script} {BROKER_ADDR} {self.min_trainers} {self.max_n_rounds} {self.stop_acc} {self.experiment.getFileName()} 2> {self.experiment.getFileName(extension='''''')}_err.txt' ;"
        # print(cmd)
        makeTerm(self.srv1, cmd=cmd)
        
        
    def start_clientes(self):
        info('*** Inicializando clientes\n')
        count = 0
        for client_type in self.config.get("client_types"):
            for x in range(1, client_type["amount"]+1):
                info(f"*** Subindo cliente {str(count+1).zfill(2)}\n")
                vol = client_type["volume"]
                cmd = f"bash -c 'cd {vol} && . env/bin/activate && python3 {client_type['script']} {BROKER_ADDR} {self.clientes[count].name} {count} {self.general['trainer_mode']}' ;"
                # print(cmd)
                makeTerm(self.clientes[count], cmd=cmd)
                count += 1