## Layout of the SageMaker Experiments at Scale Project Template

The template provides a starting point for you ML experimentation setup.

```
.
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── codebuild-buildspec.yml
├── config.json
├── img
│   └── pipeline-full.png
├── pipelines
│   ├── __init__.py
│   ├── __version__.py
│   ├── _utils.py
│   ├── abalone
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── hyperparameters
│   │   │   ├── hyperparameters_rf.json
│   │   │   └── hyperparameters_xgboost.json
│   │   ├── metric_definitions
│   │   │   └── metric_definitions_rf.json
│   │   ├── pipeline.py
│   │   ├── preprocessing
│   │   │   ├── preprocess_rf.py
│   │   │   └── preprocess_xgboost.py
│   │   └── training
│   │       └── training_script_rf.py
│   ├── get_pipeline_definition.py
│   └── run_pipeline.py
├── run_experiments.py
├── sagemaker-pipelines-project.ipynb
├── setup.cfg
├── setup.py
├── tests
│   └── test_pipelines.py
└── tox.ini
```

## Start here
This is a sample code repository that demonstrates how you can organize your code when running your ML experiments at scale.
This code repository is created as part of creating a Project in SageMaker. 

In this example, we are solving the abalone age prediction problem using the abalone dataset (see below for more on the dataset). The following section provides an overview of how the code is organized and what you need to modify. In particular, the `run_experiments.py` contains the code to create the ML steps involved in generating experimentation process of ML model.

Once you understand the code structure described below, you can inspect the code and you can start customizing it for your own scenarios. This is only sample code, and you own this repository for your business use case. Please go ahead, modify the files, commit them and see the changes kick off the SageMaker pipelines in the CICD system.

You can also use the `sagemaker-pipelines-project.ipynb` notebook to experiment from SageMaker Studio before you are ready to checkin your code.

A description of some of the artifacts is provided below:
<br/><br/>
Your codebuild execution instructions. This file contains the instructions needed to kick off an execution of the SageMaker Pipeline in the CICD system (via CodePipeline). You will see that this file has the fields definined for naming the Pipeline, ModelPackageGroup etc, and is the file in charge of running the `run_experiments.py` script. You can customize them as required.

```
|-- codebuild-buildspec.yml
```

<br/><br/>
The experiments configuration is stored in the config.json file, however, experiment specific files are stored in the correponding subfolders inside the abalone folder.
You can update this files and expand them as necessary to run your own set of experiments.

```
├── config.json
├── pipelines
│   ├── abalone
│   │   ├── hyperparameters
│   │   │   ├── hyperparameters_rf.json
│   │   │   └── hyperparameters_xgboost.json
│   │   ├── metric_definitions
│   │   │   └── metric_definitions_rf.json
│   │   ├── preprocessing
│   │   │   ├── preprocess_rf.py
│   │   │   └── preprocess_xgboost.py
│   │   └── training
│   │       └── training_script_rf.py
```


<br/><br/>
Your pipeline artifacts, which includes a pipeline module defining the required `get_pipeline` method that returns an instance of a SageMaker pipeline. This is the core business logic, and if you want to create your own folder, you can do so, and implement the `get_pipeline` interface as illustrated here.

```
|-- pipelines
|   |-- abalone
|   |   |-- evaluate.py
|   |   |-- __init__.py
|   |   |-- pipeline.py

```
<br/><br/>
Utility modules for getting pipeline definition jsons and running pipelines (you do not typically need to modify these):

```
|-- pipelines
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
```
<br/><br/>
Python package artifacts:
```
|-- setup.cfg
|-- setup.py
```
<br/><br/>
A stubbed testing module for testing your pipeline as you develop:
```
|-- tests
|   `-- test_pipelines.py
```
<br/><br/>
The `tox` testing framework configuration:
```
`-- tox.ini
```

## Dataset for the Example Abalone Pipeline

The dataset used is the [UCI Machine Learning Abalone Dataset](https://archive.ics.uci.edu/ml/datasets/abalone) [1]. The aim for this task is to determine the age of an abalone (a kind of shellfish) from its physical measurements. At the core, it's a regression problem. 
    
The dataset contains several features - length (longest shell measurement), diameter (diameter perpendicular to length), height (height with meat in the shell), whole_weight (weight of whole abalone), shucked_weight (weight of meat), viscera_weight (gut weight after bleeding), shell_weight (weight after being dried), sex ('M', 'F', 'I' where 'I' is Infant), as well as rings (integer).

The number of rings turns out to be a good approximation for age (age is rings + 1.5). However, to obtain this number requires cutting the shell through the cone, staining the section, and counting the number of rings through a microscope -- a time-consuming task. However, the other physical measurements are easier to determine. We use the dataset to build a predictive model of the variable rings through these other physical measurements.

We'll upload the data to a bucket we own. But first we gather some constants we can use later throughout the notebook.

[1] Dua, D. and Graff, C. (2019). [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml). Irvine, CA: University of California, School of Information and Computer Science.
