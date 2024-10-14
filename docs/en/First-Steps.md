# First Steps with MiniNetFED

## Cloning the MiniNetFED Repository

```bash
git clone https://github.com/lprm-ufes/MininetFed.git
```

## Prerequisites

### Installing Docker Engine

Access the official documentation and follow the steps to install Docker engine:

https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository

### Installing ContainerNet

MiniNetFED requires ContainerNet. Before installing it, install its dependencies using the following command:

```bash
sudo apt-get install ansible git aptitude
```

#### Tested Version of ContainerNet (Recommended)

The version used of ContainerNet is in a .zip file in the **containernet** folder of the MiniNetFED repository. Copy that .zip file and paste it where you want to install ContainerNet on your machine.

> #### Other Versions (Not Recommended)
>
> If you want to install ContainerNet from other sources, it can be found in the following repositories:
>
> ##### Official
>
> ```bash
> git clone https://github.com/containernet/containernet
> ```
>
> Make sure the installation method is "Bare-metal installation" so that MiniNetFED works properly. The installation steps for this version can be found at: https://containernet.github.io/
> After installing, skip to the step _Generating Docker Images_
>
> ##### Alternative
>
> ```bash
> git clone https://github.com/ramonfontes/containernet.git
> ```

#### Installation Script (If You Are Installing the Recommended Version)

Once you have selected your preferred installation location, clone or decompress the ContainerNet files and follow these commands:

```bash
cd containernet
```

```bash
sudo util/install.sh -W
```

## Generating Docker Images

MiniNetFED also depends on some pre-configured docker images.

Use the following commands to create those images.

```bash
cd MininetFed
```

<!-- ```bash
sudo docker build --tag "mininetfed:broker" -f docker/Dockerfile.broker .
sudo docker build --tag "mininetfed:client" -f docker/Dockerfile.container .

``` -->

```bash
sudo ./docker/create_images.sh
```

The names of the images are "mininetfed:broker", "mininetfed:container", "mininetfed:client", and "mininetfed:server".

## Creating Environments with Dependencies

To create environments with dependencies to run an example, use the environment management script. The resulting environments will be in the **envs/** folder.

Creating environments for containerized devices:

```bash
sudo python scripts/envs_manage/create_container_env.py -c envs_requirements/container/client_tensorflow.requirements.txt envs_requirements/container/server.requirements.txt -std
```

Creating an environment for the analysis script:

```bash
sudo python scripts/envs_manage/create_container_env.py -l envs_requirements/local/analysis.requirements.txt -std
```

# Running MiniNetFED with an Example

To test whether everything is working properly, you can run one of the configuration files in the **exemplos/** directory. Choose a sample from the folder and execute it.

```bash
sudo python3 main.py examples/<name_of_the_selected_sample>/config.yaml
```

> ### Example Trainer Har with fed_sec_per and fed_avg
>
> ```bash
> sudo python3 main.py examples/har_fed_sec_per/config.yaml
> ```

If everything is working, the experiment should start to run by opening the following windows:

- Broker MQTT
- Server
- Network Monitor
- N clients, where N is the number of clients in the experiment

After running the experiment, a new folder should be created inside **experiments/** containing the results of the experiment.

# Analyzing the First Experiment

Inside the example folder, there is the **analysis.yaml** file. To execute it, first activate the Python environment for the analysis script:

```bash
. env_analysis/bin/activate
```

Modify and run the following command:

```bash
python3 analysis.py examples/<name_of_the_experiment>/analysis.yaml
```

> ### Example Trainer Har with fed_sec_per and fed_avg
>
> ```bash
> python3 analysis.py examples/har_fed_sec_per/analysis.yaml
> ```

Note: This translation aims to preserve the original formatting and details of the text. However, it might not adhere strictly to English documentation best practices or conventions.