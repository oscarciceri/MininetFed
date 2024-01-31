# Instalação MininetFed

## Instalar o docker engine

(Completar)

## instale containernet

O MininetFed necessita do ContainerNet. Antes de instala-lo, instale as suas dependências usando o seguinte comando

```
sudo apt-get install ansible git aptitude
```

### Obtendo o ContainerNet

#### Versão do containernet testada

A versão usada do Containernet está em um arquivo .zip no repositório na pasta containernet.

#### Outras versões (não recomendado)

Caso deseje instalar o ContainerNet de outras fontes, ele pode ser encontrado nos seguintes repositórios

##### Oficial

```
git clone https://github.com/containernet/containernet
```

##### Alternativo

```
git clone https://github.com/ramonfontes/containernet.git
```

### Instalando o ContainerNet

```
cd pastaDoContainerNet(substitua pelo nome da pasta)
```

```
sudo util/install.sh -W

```

## Instale as imagens docker

O MininetFed também depende de algumas imagens docker pré configuradas.

Utilize os comandos a seguir para criar essas imagens.

```
cd mininetFed
```

```
docker build --tag "mininetfed:broker" -f docker/Dockerfile.broker .
```

```
docker build --tag "mininetfed:client" -f docker/Dockerfile.container .

```

"mininetfed:broker" e "mininetfed:client" são os nome das imagens.

### Criando o env com as respectivas dependências

O comando a seguir roda um script que cria um **env python** e instala as dependências padrões e as necessárias para os Trainers fornecidos de exemplo. Esse **env** é usado pelos clientes e pelo servidor

```
sudo python3 scripts/create_env.py mininetfed:client scripts/requirements.txt

```

## Executar o MininetFed

```
sudo python3 main.py config.yaml

```

config.yaml é um arquivo padrão de exemplo. Substitua o config.yaml pelo arquivo de configuração de cada experimento.
