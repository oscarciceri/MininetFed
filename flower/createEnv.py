

# from containernet.cli import CLI
# from containernet.link import TCLink
# from containernet.net import Containernet
# from mininet.node import Controller
# from mininet.log import info, setLogLevel
# from containernet.term import makeTerm
# from pathlib import Path
# import time

# setLogLevel('info')
# net = Containernet(controller=Controller)
# info('*** Adding controller\n')
# net.addController('c0')

# volumes = [f"{Path.cwd()}:/flw"]
# images = "johann:ubuntu" # mudar isso


# info('*** Adicionando SWITCH\n')
# s1 = net.addSwitch('s1')

# info('*** Adicionando Containers\n')
# srv1 = net.addDocker('srv1',dimage=images, volumes=volumes, mem_limit="2048m",cpuset_cpus=f"{0}")
# net.addLink(srv1,s1)
   

# info('*** Configurando Links\n')

# net.start()


# info('*** Subindo servidor\n')
# makeTerm(srv1,cmd=f"bash -c 'cd flw && python3 -m venv env && . env/bin/activate && python3 -m pip install -r requirements.txt' ;")
# time.sleep(2)

# info('*** Rodando CLI\n')
# CLI(net)
# info('*** Parando MININET')
# net.stop()
