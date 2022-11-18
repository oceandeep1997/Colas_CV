# Damage detection

This repository provides a model to classify airport runways into whether they have certain damage. It also provides the code to scrape all runways in France to use as a training set. The labeling has to be done manually (the dataset that was provided for the challenge is company internal).

This repository provides a model to classify news into the french NAF codes at division level. It also provides an already labeled dataset to train the model on as well as helper function to scrape more data and easily transform it to the required format.

## Installation
Either clone this github repository or download all its files to the folder you want this project to be in.
Just like with any project you should first create a virtual environment to avoid messing up the pre-installed version. There are several different ways of doing this, one of which is using the [venv](https://docs.python.org/3/library/venv.html#module-venv) module that comes pre-shipped with python. It will install the environment by default in your current working directory. Therefore, the first step is to open the terminal and navigate to the directory you want the environment to be in by pasting the following line in your terminal (and replacing the text between the double quotes with the path to your dedicated folder):
```sh
# MacOS and Windows
cd "path/to/the/desired/folder"
```
Now that we are in the desired directory we can create the environment by writing the following line in the terminal (and replacing "name_of_your_environment" with whatever you want to call your environment):
```sh
# MacOS
python3 -m venv name_of_your_environment
# Windows
python -m venv name_of_your_environment
```
Having created the environment, we can activate it with the following line in the terminal:
```sh
# MacOS
source name_of_your_environment/bin/activate
# Windows
name_of_your_environment/Scripts/activate.bat
```
This will prepend the environment name in parenthesis in the terminal, telling us that we are now in the virtual environment.
For more information on the venv module refer to the offical venv [**documentation**](https://docs.python.org/3/library/venv.html#module-venv).

To create virtual environments using conda, refer to their [**documentation**](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

Finally, we can install the necessary packages from the requirements.txt file that comes with this repository. The following line does just that:
```sh
# MacOS and Windows
pip install -r /path/to/requirements.txt
```

# Main Features
- A tensorflow classification model built on top of the pretrained CNN Xception with an additional dense layer
- Functions to train and evaluate the model
- Functions to scrape airport and runway coordinates
- Functions to collect ORTHO photos
- Function to perform data augmentation via image rotation

# Repo organization:
- Data folder:  geospatial vector files of airports and runways
- model_building folder : python scripts in which the python classes used to create the model are defined
- notebooks : folder containing the different notebooks used for data collection, data augmentation and data analysis

# Examples
A Jupyter Notebook that showcases the use of the different classes and functions has been created. It can be found in the "notebooks"  folder

# Background
This project was created for a data science challenge posed and managed by [**Colas**](https://www.colas.com/fr) in collaboration with [**HEC**](https://www.hec.edu/en/master-s-programs/ecole-polytechnique-hec-programs/master-science-data-sciencefor-business-ecole-polytechnique-hec). It aimed at detecting five different types of datamage in the tarmac of airport runways. A labeled dataset was provided by Colas, computing resources including GPUs were made available by HEC.