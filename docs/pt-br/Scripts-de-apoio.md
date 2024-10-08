# Scripts de apoio

O MiniNetFED é distribuido com algum scripts de apoio. A seguir a função de cada um será explicada

# clean.sh

```bash
scripts/clean.sh
```

Caso a execução do MiniNetFED seja interrompida indevidamente, o Containernet ou o docker poderão deixar instâncias de elementos de rede ou containers dockers para trás respectivamente. Se outra outra execução do MiniNetFED for iniciada, pode haver erros ou avisos. Nesse caso, o script clean.sh deleta todos os containers docker ativos e executa a limpeza do Containernet.

**Atenção**: O script irá deletar todos os containers docker instanciados, portanto se houver outra aplicação na máquina que utilize containers dockers, ela possivelmente será afetada. Nesse caso é recomendado a deleção manual dos containers instanciados pelo MiniNetFED, e após usar o comando a seguir para apenas limpar o Containernet.

```bash
sudo mn -c
```

# create_env.py

```bash
sudo python3 scripts/create_env.py <imagem docker usada para o cliente> <requirements.txt>
```

O script que cria o _env_ instancia um container docker usando a imagem do cliente do MiniNetFED para que os clientes sejam compatíveis com o _env_ criado. Como o _env_ é criado partindo de um container, não é necessário ter instalado na máquina o _venv_ para python. O script também recebe o arquivo _requirements.txt_, o qual também está na pasta /scripts.

# env_analysis.sh

Assim como o script anterior, esse script cria um _env_ python, mas há algumas diferenças. Nele o _env_ é criado pela e para a própria máquina em que é executado, e não é necessário fornecer nenhuma informação adicional. Ele já busca automaticamente o arquivo de requirements.

Um ponto importante é que esse script necesita do _venv_ para python, e se o mesmo não estiver disponível, ele será autimaticamente instalado.

```bash
./scripts/env_analysis.sh
```
