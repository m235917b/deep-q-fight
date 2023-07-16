"""
The controller for the agents and training algorithm.

Training and evaluation can be terminated using the ESC-key.
"""

from ai.game_adapter import GameAdapterImpl
from ai.model import Trainer, Tester

# the number of games, the agents should be tested on
_TEST_GAMES = 100
# the number of games, the agents should be trained on
_TRAINING_GAMES = 250


class Controller:
    def __init__(self, eval_path):
        self.__game = GameAdapterImpl()
        self.__trainer = Trainer(self.__game, eval_path)

    def test(self, path=None, random_agents=False):
        tester = Tester(self.__game, path, random_agents=random_agents)

        for i in range(1, _TEST_GAMES + 1):
            tester.evaluate()

            print(f'Testing game {i} completed!')

            self.__game.game_ctr += 1

            if self.__game.quit():
                break

    def train(self):
        for i in range(1, _TRAINING_GAMES + 1):
            self.__trainer.train_episode()

            if i % 10 == 0:
                self.__trainer.save_policy(f'../resources/nets/policy_{i}.pt')
                self.__trainer.save_target(f'../resources/nets/target_{i}.pt')
            print(f'Training game {i} completed!')

            self.__game.game_ctr += 1

            if self.__game.quit():
                break
