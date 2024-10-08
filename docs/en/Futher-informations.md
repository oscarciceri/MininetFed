# Dependency Details

## MiniNetFED Environment

The clients and server provided with MiniNetFED are as follows:

- numpy
- scikit-learn
- keras
- pandas
- paho-mqtt
- tensorflow
- scikit-learn

Note, however, that there are many other dependencies. For more details, see the _requirements.txt_ file in the /scripts directory.

These can be modified according to new Trainers, and selection or aggregation functions implemented. For example, if the installation of pytorch is required for the execution of a new Trainer, you should add the dependency to the requirements.txt file inside the scripts directory, delete (if it exists) the current _env_, and run the script creation _env_ again for MiniNetFED client.

## Analysis Script Environment

The main dependencies are as follows:

- pandas
- scikit-learn
- matplotlib

**Note**, this script also includes the necessary dependencies to instantiate a trainer (e.g., tensorflow, keras) in order to analyze the data distribution between clients. Therefore, if it is necessary to include some new dependencies for the implementation of a new Trainer (e.g., pytorch), they will also need to be included in the analysis tool's dependencies if you want to analyze the class distribution of this trainer. In this case, you must add the new dependency to the _requirements.txt_ file inside the **/analysis** directory, delete the _/env_analysis_ directory and create it again using the script creation env of the analyzer.

---

## MiniNetFED Environment

The clients and server provided with MiniNetFED are as follows:

- numpy
- scikit-learn
- keras
- pandas
- paho-mqtt
- tensorflow
- scikit-learn

Note, however, that there are many other dependencies. For more details, see the _requirements.txt_ file in the /scripts directory.

These can be modified according to new Trainers, and selection or aggregation functions implemented. For example, if the installation of pytorch is required for the execution of a new Trainer, you should add the dependency to the requirements.txt file inside the scripts directory, delete (if it exists) the current _env_, and run the script creation _env_ again for MiniNetFED client.

## Analysis Script Environment

The main dependencies are as follows:

- pandas
- scikit-learn
- matplotlib

**Note**, this script also includes the necessary dependencies to instantiate a trainer (e.g., tensorflow, keras) in order to analyze the data distribution between clients. Therefore, if it is necessary to include some new dependencies for the implementation of a new Trainer (e.g., pytorch), they will also need to be included in the analysis tool's dependencies if you want to analyze the class distribution of this trainer. In this case, you must add the new dependency to the _requirements.txt_ file inside the **/analysis** directory, delete the _/env_analysis_ directory and create it again using the script creation env of the analyzer.

---

## Env Script de Análise

The main dependencies are as follows:

- pandas
- scikit-learn
- matploylib