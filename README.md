# deep-q-fight
A simple 2D fighting game where a player or agents can fight against agents. The agents can be trained with Deep-Q-Learning.

# getting started
To be able to run the project, you first need to install the required python packages.

If you want to run the project as is, just use the given virtual environment in ".venv".

Or you can create your own environment and install all packages via the "requirements.txt" file. To do this, run the following command in your activated environment:
pip install -r requirements.txt

If you do not have CUDA available, or want to run PyTorch on CPU, or want to run the project on Mac/Linux you will have to manually install all required packages.
To do this, first create an empty virtual environment and activate it. Then install the following packages (or any other compatible versions):

numpy==1.24.1
numba==0.57.1
pygame==2.5.0

Then you need to install PyTorch. To do this, visit [PyTorch start locally](https://pytorch.org/get-started/locally/) and configure the settings for your
specifications and OS, then install the required torch version with the generated pip instruction from this site.

# running the project
