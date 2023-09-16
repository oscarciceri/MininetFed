from containernet.cli import CLI
from containernet.link import TCLink
from containernet.net import Containernet
from mininet.node import Controller
from mininet.log import info, setLogLevel
from containernet.term import makeTerm
from pathlib import Path
import time
from Config import Config
import sys

# total args
n = len(sys.argv)
 
# check args
if (n != 2):
    print("correct use: sudo python3 run.py <config.yaml>")
    exit()
 
CONFIGYAML = sys.argv[1]

setLogLevel('info')
info('*** Importing configurations\n')

#endereço do arquivo de configurações
config = Config(CONFIGYAML) 

general = config.get("general")
absolute = general["absolute_path"]

server = config.get("server")
server_volumes = ""
server_script = ""

if absolute: 
    server_volumes = server["volume"] 
    server_script = server['script']
else: 
    server_volumes = [f"{Path.cwd()}:" + server["volume"]]
    server_script = [f"{Path.cwd()}:" + server["script"]]
server_images = server["image"]





net = Containernet(controller=Controller)
info('*** Adding controller\n')
net.addController('c0')

info('*** Adicionando SWITCHS\n')
s = list()
for i in range(1,config.get("network_components") + 1):
    s.append(net.addSwitch(f"s{i}"))


info('*** Adicionando Containers\n')
# server container
srv1 = net.addDocker('srv1',dimage=server_images, volumes=server_volumes, mem_limit=server["memory"],cpuset_cpus=f"{0}")
net.addLink(srv1,s[server["conection"] - 1])


# client containers
clientes = list()
cont = 0
qtdDevice = 0
for client_type in config.get("client_types"):
    for x in range(1,client_type["amount"]+1):
        volumes = ""
        if absolute: volumes = client_type["volume"]
        else: volumes = [f"{Path.cwd()}:" + client_type["volume"]]
        qtdDevice += 1
        d = net.addDocker(f'sta{client_type["name"]}{x}', cpuset_cpus=f"{cont}",dimage=client_type["image"], volumes=volumes,  mem_limit=client_type["memory"])
        net.addLink(d,s[client_type['conection'] - 1],loss=client_type["loss"],bw=client_type["bw"])
        clientes.append(d)
        cont=(cont+1)%16
        

info('*** Configurando Links\n')

net.start()

BROKER_ADDR = srv1.IP()
MIN_TRAINERS = 10
TRAINERS_PER_ROUND = 10
NUM_ROUNDS = 100
STOP_ACC = 80

print(srv1.IP())

srv1.cmd("bash -c 'cd run && mkdir mosquitto && sudo service mosquitto start'")

info('*** Subindo servidor\n')
makeTerm(srv1,cmd=f"bash -c '. flw/env/bin/activate && python3 flw{server_script} {BROKER_ADDR} {MIN_TRAINERS} {TRAINERS_PER_ROUND} {NUM_ROUNDS} {STOP_ACC}' ;")
time.sleep(2)

cont=0
for client_type in config.get("client_types"):
    for x in range(1,client_type["amount"]+1):
        info(f"*** Subindo cliente {str(cont+1).zfill(2)}\n")
        cmd = f"bash -c '. flw/env/bin/activate && python3 flw/{client_type['script']} {BROKER_ADDR} ' ;"
        makeTerm(clientes[cont],cmd=cmd)
        cont+=1

info('*** Rodando CLI\n')
CLI(net)
info('*** Parando MININET')
net.stop()
