# Primeiros passos com o MininetFed

## Clonando o repositório do MininetFed

```
git clone https://github.com/lprm-ufes/MininetFed.git
```

## Pré requisitos

### Instalar o docker engine

Acesse a documentação oficial e siga os passos para a instalação do docker engine:

https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository


### Instalar ContainerNet

O MininetFed necessita do ContainerNet. Antes de instala-lo, instale as suas dependências usando o seguinte comando

```
sudo apt-get install ansible git aptitude
```

#### Versão do ContainerNet testada (recomendado)

A versão usada do ContainerNet está em um arquivo .zip na pasta **containernet** do repositório do MininetFed. Copie esse arquivo .zip e cole ele no lugar onde deseja instalar o ContainerNet na sua máquina

> #### Outras versões (não recomendado)
> 
> Caso deseje instalar o ContainerNet de outras fontes, ele pode ser encontrado nos seguintes repositórios
> 
> ##### Oficial
> 
> ```
> git clone https://github.com/containernet/containernet
> ```
>
> É importante que o método de instalação seja "Bare-metal installation" para que o MininetFed funcione adequadamente
> Os passos de instalação dessa versão podem ser encontrados no seguinte link: https://containernet.github.io/
> Após a instalação, pule para o passo *Gerando as imagens docker*
>
> ##### Alternativo
> 
> ```
> git clone https://github.com/ramonfontes/containernet.git
> ```

#### Script de instalação (caso você estiver instalando a versão recomendada)

Uma vez selecionado o local de instalação de sua preferência, clone ou decompacte os arquivos do containernet e siga com os seguintes comandos

```
cd containernet
```

```
sudo util/install.sh -W

```


## Gerando as imagens docker

O MininetFed também depende de algumas imagens docker pré configuradas.

Utilize os comandos a seguir para criar essas imagens.

```
cd MininetFed
```

```
sudo docker build --tag "mininetfed:broker" -f docker/Dockerfile.broker .
sudo docker build --tag "mininetfed:client" -f docker/Dockerfile.container .

```

"mininetfed:broker" e "mininetfed:client" são os nome das imagens.

## Criando o env com as respectivas dependências

O comando a seguir roda um script que cria um **env** python e instala as dependências padrões e as necessárias para os Trainers fornecidos de exemplo. Esse **env** é usado pelos clientes e pelo servidor

```
sudo python3 scripts/create_env.py mininetfed:client scripts/requirements.txt

```

O comando a seguir cria o **env_analysis** que será usado posteriormente para analisar os resultados gerados por um experimento

```
./scripts/env_analysis.sh
```

# Executar o MininetFed com um exemplo

Para testar se tudo está funcionando adequadamente, é possível executar um dos arquivos de configuração do diretório **exemplos**. Escolha um dos exemplos da pasta e modifique no arquivo config.yaml a chave n_available_cpu: colocando o número de núcleos lógicos disponíveis na máquina utilizada.

```
general:
  n_available_cpu: <número de núcleos lógicos>
  (...)

```

Após essa modificação, modifique e execute o comando a seguir

```
sudo python3 main.py examples/<nome do exemplo escolhido>/config.yaml

```

> ### Exemplo Trainer Har com fed_per_sec e fed_avg
>
> ```
> sudo python3 main.py examples/har_fed_per_sec/config.yaml
> ```

Se tudo estiver funcionando, o exeperimento deve começar a executar abrindo as seguintes janelas:

* Broker MQTT
* Servidor
* Monitor de rede
* N clientes, onde N é o número de clientes do experimento

Após a execução do experimento, é esperado que haja uma nova pasta dentro de **experiments** contendo os resultados do experiemento.

# Fazendo a análise do primeiro experimento

Dentro da pasta do exemplo, há o arquivo **analysis.yaml**. Para executa-lo, primeiramente ative o ambiente python do script de análise 

```
. env_analysis/bin/activate
```
modifique o comando a seguir e execute-o:

```
python3 analysis.py examples/<nome do experimento>/analysis.yaml
```

> ### Exemplo Trainer Har com fed_per_sec e fed_avg
>
> ```
> python3 analysis.py examples/har_fed_per_sec/analysis.yaml
> ```

