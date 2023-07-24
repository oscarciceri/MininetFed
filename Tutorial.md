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
docker build --tag "mosquito" .

```
"mosquito" e nome da imagem lembra de usar ela quando chamar docker pode da qualquer nome. 
1- arquivo topo exemplo com comentarios

