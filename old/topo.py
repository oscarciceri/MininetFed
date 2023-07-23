

from containernet.cli import CLI
from containernet.link import TCLink
from containernet.net import Containernet
from mininet.node import Controller
from mininet.log import info, setLogLevel
from containernet.term import makeTerm
from pathlib import Path
import time

setLogLevel('info')
net = Containernet(controller=Controller)
info('*** Adding controller\n')
net.addController('c0')

# comando para pega o path local (volume compartilhado reduzir dados do imagem criando um env.)
volumes = [f"{Path.cwd()}:/flw"]
images = "ubutun:jovane"
clientes = list()
qtdDevice = 10


info('*** Adicionando SWITCH\n')
# comando para adicionar um switch
s1 = net.addSwitch('s1')

info('*** Adicionando Containers\n')

# criando um container via container net
## dimage -> qual iamgem criado no docker
## volumes -> caso tenha algum voluem a compartilhar , bom para compartilhar dados
## mem_limit -> ajuste de memoria do docker
## cpus=<value> 	Especifique quanto dos recursos de CPU disponíveis um contêiner pode usar. Por exemplo, se a máquina host tiver duas CPUs e você definir --cpus="1.5", o contêiner terá garantido no máximo uma e meia das CPUs. Isso é o equivalente a definir --cpu-period="100000"e --cpu-quota="150000".
## cpu-period=<value> 	Especifique o período do agendador CFS da CPU, que é usado junto com --cpu-quota. O padrão é 100.000 microssegundos (100 milissegundos). A maioria dos usuários não altera isso do padrão. Para a maioria dos casos de uso, --cpusé uma alternativa mais conveniente.
## cpu-quota=<value> 	Imponha uma cota de CPU CFS no contêiner. O número de microssegundos por --cpu-periodque o contêiner é limitado antes de ser limitado. Como tal atuando como o teto efetivo. Para a maioria dos casos de uso, --cpusé uma alternativa mais conveniente.
## cpuset-cpus 	Limite as CPUs ou núcleos específicos que um contêiner pode usar. Uma lista separada por vírgula ou intervalo separado por hífen de CPUs que um contêiner pode usar, se você tiver mais de uma CPU. A primeira CPU é numerada como 0. Um valor válido pode ser 0-3(para usar a primeira, segunda, terceira e quarta CPU) ou 1,3(para usar a segunda e a quarta CPU).
## cpu-shares 	Defina esse sinalizador com um valor maior ou menor que o padrão de 1024 para aumentar ou reduzir o peso do contêiner e dar a ele acesso a uma proporção maior ou menor dos ciclos de CPU da máquina host. Isso só é aplicado quando os ciclos da CPU são restritos. Quando muitos ciclos de CPU estão disponíveis, todos os contêineres usam a quantidade de CPU necessária. Dessa forma, este é um limite suave. --cpu-sharesnão impede que os contêineres sejam agendados no modo swarm. Ele prioriza os recursos de CPU do contêiner para os ciclos de CPU disponíveis. Ele não garante ou reserva qualquer acesso específico à CPU.
# https://github.com/containernet/containernet/wiki
srv1 = net.addDocker('srv1', dimage=images, volumes=volumes, mem_limit="512m",cpuset_cpus=f"{0}")

#conectando o servido ao switch
net.addLink(srv1,s1)

cont = 0

# loop criar dos dispositivos
for x in range(1,qtdDevice+1):
    d = net.addDocker(f'sta{x}', cpuset_cpus=f"{cont}",dimage=images, volumes=volumes,  mem_limit="1024m")
    net.addLink(d,s1,loss=0,bw=10)
    clientes.append(d)
    cont=(cont+1)%16
    

info('*** Configurando Links\n')

net.start()


info('*** Subindo servidor\n')

# comando makeTerm cria uma chamada no srv1 com o comando bash, executa a aplicaçao flower com os parametros
makeTerm(srv1,cmd=f"bash -c '. flw/env/bin/activate &&  python3.8 flw/Server.py -nc {qtdDevice}' ;")

# dependendo do tempo de subir aplicaçao do servidor e bom ter um sleep se nao cliente sobe antes so serviço subir.
time.sleep(2)

info('*** Rodando CLI\n')
CLI(net)
info('*** Parando MININET')
net.stop()
