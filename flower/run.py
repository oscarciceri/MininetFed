from containernet.cli import CLI
from containernet.link import TCLink
from containernet.net import Containernet
from mininet.node import Controller
from mininet.log import info, setLogLevel
from containernet.term import makeTerm
from pathlib import Path
import time
from Config import Config

setLogLevel('info')
info('*** Importing configurations\n')
config = Config('flower/config.yaml') #endereço do arquivo de configurações

general = config.get("general")
absolute = general["absolute_path"]

server = config.get("server")
server_volumes = ""
if absolute: server_volumes = server["volume"] 
else: server_volumes = [f"{Path.cwd()}:" + server["volume"]]
server_images = server["image"]

images = server["image"]
volumes = [f"{Path.cwd()}:" + server["volume"]]



net = Containernet(controller=Controller)
info('*** Adding controller\n')
net.addController('c0')



clientes = list()



info('*** Adicionando SWITCHS\n')
s = list()
for i in range(1,config.get("network_components") + 1):
    s.append(net.addSwitch(f"s{i}"))



info('*** Adicionando Containers\n')
# server container
srv1 = net.addDocker('srv1',dimage=server_images, volumes=server_volumes, mem_limit=server["memory"],cpuset_cpus=f"{0}")
net.addLink(srv1,s[server["conection"] - 1])


# client containers
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


info('*** Subindo servidor\n')
makeTerm(srv1,cmd=f"bash -c '. flw/env/bin/activate && python3 flw{server['script']} -nc {qtdDevice}' ;")
time.sleep(2)

cont=0
for client_type in config.get("client_types"):
    for x in range(1,client_type["amount"]+1):
        info(f"*** Subindo cliente {str(cont+1).zfill(2)}\n")
        cmd = f"bash -c '. flw/env/bin/activate && python3 flw{client_type['model']} ' ;"
        makeTerm(clientes[cont],cmd=cmd)
        cont+=1

info('*** Rodando CLI\n')
CLI(net)
info('*** Parando MININET')
net.stop()
