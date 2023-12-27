# Instalação MininetFed

## Crie a vm do linux (opicional)

Caso tenha interesse em emcapsular a instalação do containernet. Esse passo está sujeito a mudanças futuras

## Instalar o docker engine

(Completar)

## instale containernet

```
sudo apt-get install ansible git aptitude
```

```
git clone https://github.com/ramonfontes/containernet.git
```

```
sudo util/install.sh -W

```

## Instale as imagens docker

```
docker build --tag "mininetfed:broker" -f docker/Dockerfile.broker .
docker build --tag "mininetfed:client" -f docker/Dockerfile.container .

```

"mininetfed:broker" e "mininetfed:client" são os nome das imagens.

```
sudo python3 scripts/create_env.py mininetfed:client scripts/requirements.txt

```

Cria o env para o uso na versão com o flower. Instala automaticamene as bibliotecas em requirements.txt

## Executar o flower

```
sudo python3 flower/run.py

```

Executa o flower puxado as configurações de config.yaml
