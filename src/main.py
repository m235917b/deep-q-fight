"""
Main file of the project. This initializes the training and test procedures.

For evaluating the trained model after each training game, set _EVAL_ROUNDS in "src/ai/model.py".

For testing the agents without the ability to see team members use the respective lines in
ray_tracing_kernel() in "src/ai/game_adapter.py".

For testing the agents without the ability to communicate, comment out the respective for-loop in
__control_actors() in "src/ai/game_adapter.py".

For testing the agents without the obstacle in the world, use the respective line in
__init__() in "src/game/model.py".

For disable the ability of the agents to turn right after 50 frames, uncomment the respective line in
__control_actors() in "src/ai/game_adapter.py".
"""

from ai import controller

"""Switch if human player should control an actor.
Should only be used for test game."""
PLAYER_MODE = False

# the path to the model used during inference for testing
_PATH_TO_EVAL = '../resources/results/base_trained_agents.pt'

if __name__ == '__main__':
    """
    Main function of the program.
    Here the training and testing of the models is started.
    """

    """The controller for the agents with the path to the model, that the
    controller uses to test the model in each evaluation round after each
    training game."""
    ai_controller = controller.Controller('../resources/results/base_trained_agents.pt')

    # use this line to train a model
    # ai_controller.train()

    # use this line to test the model specified by _PATH_TO_EVAL
    ai_controller.test(_PATH_TO_EVAL)

    # use this line to test random agents
    # ai_controller.test(_PATH_TO_EVAL, random_agents=True)
