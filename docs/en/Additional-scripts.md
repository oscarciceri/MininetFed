# Support Scripts

The MiniNetFED is distributed with some support scripts. The function of each will be explained below.

# clean.sh

```bash
scripts/clean.sh
```

If the execution of MiniNetFED is interrupted improperly, Containernet or docker may leave behind instances of network elements or dockers containers respectively. If another execution of MiniNetFED is initiated, errors or warnings may occur. In this case, the clean.sh script deletes all active dockers containers and performs Containernet cleanup.

**Warning**: The script will delete all dockers containers instantiated, so if there is another application on the machine that uses dockers containers, it may be affected. In this case, it is recommended to manually delete the containers instantiated by MiniNetFED, and then use the following command to only clean Containernet.

```bash
sudo mn -c
```

# create_env.py

```bash
sudo python3 scripts/create_env.py <docker image used for client> <requirements.txt>
```

The script that creates the _env_ instantiates a dockers container using the MiniNetFED client image, so that clients are compatible with the created _env_. As the _env_ is created from a container, it is not necessary to have the _venv_ for python installed on the machine. The script also receives the _requirements.txt_ file, which is also in the /scripts folder.

# env_analysis.sh

Similar to the previous script, this script creates a python _env_, but with some differences. In it, the _env_ is created by and for the same machine where it is executed, and no additional information needs to be provided. It already searches automatically for the requirements file.

An important point is that this script needs the _venv_ for python, and if the latter is not available, it will be installed automatically.

```bash
./scripts/env_analysis.sh
```