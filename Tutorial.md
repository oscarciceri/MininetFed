##Crie a vm do linux

##instale containernet

```
sudo apt-get install ansible git aptitude
git clone https://github.com/ramonfontes/containernet.git
sudo util/install.sh -W

```

##instale a imagem docker
#aquivo dockerfile

```
docker build --tag "johann:ubuntu" .

```

"johann:ubuntu" é o nome da imagem.

```
sudo python3 createEnv.py

```

Cria o env para o uso na versão com o flower. Instala automaticamene as bibliotecas em requirements.txt

```
sudo python3 flower/run.py

```

Executa o flower puxado as configurações de config.yaml
