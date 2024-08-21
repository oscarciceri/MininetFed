try:
    from containernet.net import Containernet
    from containernet.term import makeTerm
    from containernet.cli import CLI
except:
    from mininet.net import Containernet
    from mininet.term import makeTerm
    from mininet.cli import CLI
from mininet.node import Controller
from mininet.log import info, setLogLevel
from pathlib import Path
import sys

# total args
n = len(sys.argv)

images = "mininetfed:container"
# check args
if (n != 2):
    print("Default image: mininetfed:container")
    print("use suggestion: sudo python3 standalone_machine.py <image>")

else:
    images = sys.argv[1]


setLogLevel('info')
net = Containernet(controller=Controller)
info('*** Adding controller\n')
net.addController('c0')


volumes = [f"{Path.cwd()}:/flw"]


s1 = net.addSwitch('s1')

info('*** Adicionando Containers\n')
srv1 = net.addDocker('srv1', dimage=images, volumes=volumes, mem_limit="4096m")
net.addLink(srv1, s1)

net.start()

CLI(net)
info('*** Parando MININET')
net.stop()
