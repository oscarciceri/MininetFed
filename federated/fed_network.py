from containernet.cli import CLI
from containernet.net import Containernet
from containernet.term import makeTerm
from mininet.log import info, setLogLevel
from mininet.node import Controller

from pathlib import Path
import time

from .config import Config



BROKER_ADDR = "172.17.0.2"
MIN_TRAINERS = 2
TRAINERS_PER_ROUND = 3
NUM_ROUNDS = 10
STOP_ACC = 1.0
      

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
        
        self.server = self.config.get("server")
        self.server_volumes = ""
        self.server_script = ""
        self.server_quota = self.server["vCPU_percent"] * self.n_cpu * 1000
        self.server_volumes = [f"{Path.cwd()}:" + self.server["volume"]]

        # if absolute:
        #     server_script = [f"{Path.cwd()}" + server["script"]]
        # else:
        #     server_script = server['script']

        self.server_images = self.server["image"]
        
        
        self.net = Containernet(controller=Controller)
        info('*** Adding controller\n')
        self.net.addController('c0')
        
        self.insert_switch(self.config.get("network_components"))
        self.insert_broker_container()
        self.insert_server_container()
        self.insert_client_containers()



    def insert_switch(self, qtd):
        info('*** Adicionando SWITCHS\n')
        self.insert_switch
        for i in range(1, qtd + 1):
          self.switchs.append(self.net.addSwitch(f"s{i}"))
    
    
    
    def insert_broker_container(self):
        info('*** Adicionando Container do Broker\n')
        # broker container
        self.broker = self.net.addDocker('brk1', dimage=self.broker_image,
                              volumes=self.server_volumes,  mem_limit=self.broker_mem)
        self.net.addLink(self.broker, self.switchs[self.server["conection"] - 1])
    
    
    
    def insert_server_container(self):
        info('*** Adicionando Container do Server\n')
        self.srv1 = self.net.addDocker('srv1', dimage=self.server_images, volumes=self.server_volumes,
                     mem_limit=self.server["memory"], cpu_quota=self.server_quota)
        self.net.addLink(self.srv1, self.switchs[self.server["conection"] - 1])
      
      
      
    def insert_client_containers(self):
      info('*** Adicionando Container do Server\n')
     
      cont = 0
      qtdDevice = 0
      for client_type in self.config.get("client_types"):
          for x in range(1, client_type["amount"]+1):
              volumes = ""
              if self.absolute:
                  volumes = client_type["volume"]
              else:
                  volumes = [f"{Path.cwd()}:" + client_type["volume"]]
              qtdDevice += 1
              client_quota = client_type["vCPU_percent"] * self.n_cpu*1000
              d = self.net.addDocker(f'sta{client_type["name"]}{x}', cpu_quota=client_quota,
                                dimage=client_type["image"], volumes=volumes,  mem_limit=client_type["memory"])
              self.net.addLink(d, self.switchs[client_type['conection'] - 1],
                          loss=client_type["loss"], bw=client_type["bw"])
              self.clientes.append(d)
              cont = (cont+1) % 16
              
              
    def start(self):
        info('*** Configurando Links\n')
        
        self.net.start()
        time.sleep(2)
        self.start_broker() 
        time.sleep(3)
        self.start_clientes
        
        info('*** Rodando CLI\n')
        CLI(self.net)
        info('*** Parando MININET')
        self.net.stop()
        
          
    def start_broker(self):
        info('*** Inicializando broker\n')
        makeTerm(self.broker, cmd="bash -c 'mosquitto -c /flw/mosquitto.conf'")
        
    def start_server(self):
        info('*** Inicializando servidor\n')
        tScrip = self.server["script"]
        cmd = f"bash -c '. flw/env/bin/activate && python3 flw{tScrip} {BROKER_ADDR} {MIN_TRAINERS} {TRAINERS_PER_ROUND} {NUM_ROUNDS} {STOP_ACC} flw/meu_arquivo.log' ;"
        print(cmd)
        makeTerm(self.srv1, cmd=cmd)
        
        
    def start_clientes(self):
        info('*** Inicializando clientes\n')
        cont = 0
        for client_type in self.config.get("client_types"):
            for x in range(1, client_type["amount"]+1):
                info(f"*** Subindo cliente {str(cont+1).zfill(2)}\n")
                cmd = f"bash -c '. flw/env/bin/activate && python3 flw{client_type['script']} {BROKER_ADDR} {self.clientes[cont].name} ' ;"
                print(cmd)
                makeTerm(self.clientes[cont], cmd=cmd)
                cont += 1