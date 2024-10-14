# Support Scripts

The MiniNetFED is distributed with some support scripts. The function of each will be explained below.

# clean.sh

```bash
scripts/clean.sh
```

If the execution of MiniNetFED is interrupted improperly, Containernet or docker may leave behind instances of network elements or dockers respectively. If another instance of MiniNetFED is initiated, there may be errors or warnings. In this case, the clean.sh script deletes all active docker containers and executes a cleanup of Containernet.

**Attention**: The script will delete all docker containers instantiated, so if there is another application on the machine that uses dockers, it may be affected. In this case, it is recommended to manually delete the containers instantiated by MiniNetFED, and then use the following command to only clean up Containernet.

```bash
sudo mn -c
```

# Environment Manager

```bash
sudo python3 scripts/envs_manage/create_container_env.py [-c|-l] req/folder|exemplo.requirements.txt ... -std|image_name
```

The flag `-c` indicates that the environment will be created to run in a container, and `-l` for execution on the local machine. You can then pass either the address of a folder or the address of multiple requirements files. Finally, you can pass the flag `-std` to use the standard container image or the name of the image.

**Important**: The file must end with `.requirements.txt` to be recognized. Example: `meu_env.requirements.txt`.

The purpose of this script is to assist in creating Python environments used by MininetFed.

MininetFed already follows some `requirements.txt` files within the `envs_requirements/local` folder for local machine execution and `envs_requirements/container` for container execution.

The destination folder is set to `envs/` by default. During MininetFed setup, it is interesting to instantiate the environments `clients.requirement`

To instantiate all local environments provided, you can execute the following command

```bash
sudo python3 scripts/envs_manage/create_container_env.py -l envs_requirements/local -std
```

To instantiate your own environments containing additional dependencies for the algorithms you have implemented, you can run the script as follows

```bash
sudo python3 scripts/envs_manage/create_container_env.py -c meu/req/exemplo.requirements.txt -std
```