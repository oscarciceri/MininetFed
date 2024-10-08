Here is the translation of the text:

**Analysis Script**

Along with MininetFed, a Python script is provided to analyze the results.

**Prerequisites**

## Environment creation and activation

Run the script to create the correct Python environment dependencies

```
./scripts/analysis_env.sh
```

Activate the environment using the following command

```
. analysis_env/bin/activate
```

**Execution**

Standard example

```
python3 analysis.py analysis.yaml
```

General example

```
python3 analysis.py (analysis_configuration.yaml)
```

**Analysis Configuration File**

## Example file

```yaml
experiments_folder: experiments

experiments_analysis:
  save_csv: true
  save_graphics: true # not implemented

  from:
    - experiment: 04_02_2024fed_sec_per_har
      alias: fed per sec experiment
      files:
        - 20h29m32sfed_sec_per_har.log

    - experiment: 05_02_2024fed_sec_per_har #implicitly get all .log files from the folder
    - experiment: 06_02_2024fed_sec_per_har
    - experiment: 06_02_2024fed_sec_per_mnist

  graphics:
    - type:
		- type: (...)

datasets_analysis:
  id: 0
  mode: client
  graphics:
    - type: class_distribution
    - type: histogram
    - type: boxplot
    - type: correlation_matrix
```

## Main structure

The analysis script configuration file is divided into the following sections:

```yaml
experiments_folder: (folder containing subfolders for each experiment)
experiments_analysis:
datasets_analysis:
```

## experiments_analysis

This section is dedicated to analyzing the results of an experiment. It is divided into the following:

```yaml
  save_csv: (true or false)
  save_graphics: (true or false)
  from:
    - experiment: (experiment_name)
      alias: (alias for the experiment) (optional)
      files: (optional)
        - (file.log)
				- (...)
		- experiment: (other_experiment_name)
		- (...)

  graphics:
	 - type: (graphic_type)
	 - type: (other_graphic_type_with_params)
		 (param1): (value)
		 (param2): (value)
		 (...)
	 - type: (...)
```

### save_csv

If true, saves the .csv files with the experiment information obtained from the log. This can be useful if you want to use these data in other data analysis tools.

### save_graphics (incomplete)

### from

A list of data sources. Indicates which experiments and files you want to extract information for analysis.

Each **experiment** subitem indicates the name of an experiment. Placing only the name makes all .log files be imported for analysis, but it is still possible to add the key **files** and provide a list of .log files that you want to consider.

### graphics

Receives a list of types, where each item represents a plot of a graph.

Some graph types require additional parameters. These can be included as keys along with the graph type as in the example given above.

## datasets_analysis

This subitem is dedicated to the settings for exploratory data analysis received from a specific client.

### Pre-configuration

The analysis script imports the same trainer used by MininetFed, so it's essential to pre-configure which Trainer will be analyzed. For this, see how to select a trainer in the client.

### Structure

```yaml
datasets_analysis:
  id: (int 0 - N)
  mode: (client_mode (string))
  graphics:
	 - type: (graphic_type)
	 - type: (other_graphic_type_with_params)
		 (param1): (value)
		 (param2): (value)
		 (...)
	 - type: (...)
```

### id

The id is the integer value received by the MininetFed client. This value is used as a base for dividing the dataset when it is performed by each client. It's possible to select which client you want to analyze.

### mode

It's the same mode of config.yaml from MininetFed. It's used to select the operation and data division mode of the client.

### graphics

Receives a list of types, where each item represents a plot of a graph.

Some graph types require additional parameters. These can be included as keys along with the graph type as in the example given above.

**Available graphs by default**

## Delta T per round

## Average accuracy

## Absolute number of clients

## Relative number of clients

## Network consumption
