# deep-q-fight
A simple 2D fighting game where a player or agents can fight against agents. The agents can be trained with Deep-Q-Learning.

# getting started
To be able to run the project, you first need to install the required python packages in an anaconda environment.

First, create a new anaconda environment from the requirements.txt, e.g.:

conda create --name <environment_name> --file requirements.txt

If you do not have CUDA available, or want to run PyTorch on CPU, or want to run the project on Mac/Linux you will have to manually install all required packages.
To do this, first create an empty anaconda environment and activate it. Then install the following packages (or any other compatible versions):

numpy==1.24.1

numba==0.57.1

pygame==2.5.0

Then you need to install PyTorch. To do this, visit [PyTorch start locally](https://pytorch.org/get-started/locally/) and configure the settings for your
specifications and OS, then install the required torch version with the generated pip instruction from this site.

If you have problems installing pip inside the anaconda environment, due to an OpenSSL error, try the following solution:

[Conda SSL Error: OpenSSL appears to be unavailable on this machine.](https://github.com/conda/conda/issues/11982#issuecomment-1285538983)

When you have set up the environment, select the generated python executable from that environment as your interpreter and you should be able to run the main.py file.

# running the project
To start the program, just run the main.py file.

You can control many of the different options of the game and ai by (un)commenting the respective lines as described by the line comments.

To train agents on a new set of training games, uncomment "ai_controller.train()" in main.py. This will train agents and dump the trained neural networks to the folder ressources/nets.

You can then set the path "_PATH_TO_EVAL" to any trained network file you wish to evaluate (usually you want to evaluate the target networks) and uncomment "ai_controller.test(_PATH_TO_EVAL)"
to then evaluate and observe the agents on a given set of evaluation games.

The game will create a csv named "ressources/stats/stats.csv" with all the relevant data gathered during evaluation for each frame of each evaluation game which can then be analyzed for
objective evaluation and review of the agents. Examples for that can be found in the Jupyter-Notebook "jupyter/auswertung.ipynb".
