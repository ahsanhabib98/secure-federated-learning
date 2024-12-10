# Robust and Secure Federated Learning Framework
## Run the following commands to get started:
You have to download and set up anaconda or venv to follow this command.
Please download and set up [Anaconda](https://www.anaconda.com/products/individual#Downloads) or you can set up python [Venv](https://docs.python.org/3/library/venv.html) library.
### Initial set up
````
git clone git@github.com:ahsanhabib98/secure-federated-learning.git
cd secure-federated-learning
````
### Environment set up
#### Anaconda
````
conda create -n env_name python=3.9.7
conda activate env_name
````
#### or Python Venv
````
python -m venv env_name
source env_name/bin/activate
````
### Installing dependency
````
pip install -r requirements.txt
````
### Run federate learning without DP and SMC
````
python federate_learning.py
````
### Run federate learning with DP and SMC
````
python federate_learning_dp_smc.py
````
### Run Flask App
````
python federate_learning.py
flask db init
flask db migrate
flask db upgrade
flask run
````
## Git branching model
We will follow Git Flow as our branching model.
Please read this article to know about [Git Flow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)