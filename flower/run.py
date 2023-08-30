

from containernet.cli import CLI
from containernet.link import TCLink
from containernet.net import Containernet
from mininet.node import Controller
from mininet.log import info, setLogLevel
from containernet.term import makeTerm
from pathlib import Path
import time
import Config

setLogLevel('info')
net = Containernet(controller=Controller)
info('*** Adding controller\n')
net.addController('c0')

##  1-> CONTAINERNET 
## 2 -> Script ENV ( Pytorch, Contaiener Python.)
## 3 -> Config
## 4 -> Lerdados.
## 5 -> 

volumes = [f"{Path.cwd()}:/flw"]
images = "johann:ubuntu"
clientes = list()
qtdDevice = 2


info('*** Adicionando SWITCH\n')
s1 = net.addSwitch('s1')

info('*** Adicionando Containers\n')
srv1 = net.addDocker('srv1',dimage=images, volumes=volumes, mem_limit="256m",cpuset_cpus=f"{0}")
net.addLink(srv1,s1)

cont = 0
for x in range(1,qtdDevice+1):
    d = net.addDocker(f'sta{x}', cpuset_cpus=f"{cont}",dimage=images, volumes=volumes,  mem_limit="256m")
    net.addLink(d,s1,loss=0,bw=10)
    clientes.append(d)
    cont=(cont+1)%16
    

info('*** Configurando Links\n')

net.start()


info('*** Subindo servidor\n')
makeTerm(srv1,cmd=f"bash -c '. flw/env/bin/activate && python3 flw/flower/servidorFlower.py -nc {qtdDevice}' ;")
time.sleep(2)

cont=0
for b in clientes:
    info(f"*** Subindo cliente {str(cont+1).zfill(2)}\n")
    cmd = f"bash -c '. flw/env/bin/activate && python3 flw/flower/clienteFlower.py ' ;"
    makeTerm(b,cmd=cmd)
    cont+=1

info('*** Rodando CLI\n')
CLI(net)
info('*** Parando MININET')
net.stop()
