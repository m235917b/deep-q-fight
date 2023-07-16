"""
This adapter interfaces the game for the ai controller.
"""

import abc
import csv

import numpy as np
from numba import cuda, from_dtype

from game.model import Direction
from game.controller import Controller
from game.constants import *
import utils.math as math

# the max number of frames, one game should run
_TRUNCATION = 1000
# dump the stats for the csv every _DUMP_EVERY_FRAME frames
_DUMP_EVERY_FRAME = 1000
# the visual resolution for the ray-tracing algorithm with which the agents can see their environment
_GRIDS = VISUAL_RESOLUTION // 2
# the number of threads that should be used fo the cuda algorithms
_THREADS = 16

# the angle for an actor that he can turn per frame
_ACTOR_TURN_SPEED = 0.2
# the number of communication channels for the agents
_COMM_CHANNELS = 1
# The number of non-visual inputs (information vector) for the agents. Has to be updated if inputs get changed.
N_NV_INPUTS = 3 + N_AGENTS + _COMM_CHANNELS
# The number of the actions an agent can choose from. Has to be updated, if actions get changed
N_ACTIONS = 8 + 2 * _COMM_CHANNELS

# a default value for the inputs of the agent
_EMPTY_INPUT = [0.0] * (VISUAL_RESOLUTION * 2 + N_NV_INPUTS)

# defines the datatype used for lines in the ray-tracing algorithm
Line = np.dtype([('x1', 'f4'), ('y1', 'f4'), ('x2', 'f4'), ('y2', 'f4'), ('c', 'f4'),
                 ('id', 'i4'), ('team', 'i4')], align=True)
Line_t = from_dtype(Line)


@cuda.jit
def ray_tracing_kernel(actor_id, actor_team, pos_x, pos_y, dir_x, dir_y, lines, bitmap):
    index = cuda.grid(1)

    image_x = -dir_y
    image_y = dir_x
    pixel_x = dir_x + image_x * (1 - index * 2 / VISUAL_RESOLUTION)
    pixel_y = dir_y + image_y * (1 - index * 2 / VISUAL_RESOLUTION)

    i = 0
    pixel_value = (-1, -1)
    while i < len(lines):
        # use this line if only actors of team 0 should see teammates
        # if lines[i].id != actor_id and (lines[i].team != actor_team or actor_team == 0):
        # use this line if actors of the same team should not be able to see teammates
        # if lines[i].id != actor_id and (lines[i].team != actor_team):
        # use this line if actors should see teammates
        if lines[i].id != actor_id:
            line_x = lines[i].x1 - lines[i].x2
            line_y = lines[i].y1 - lines[i].y2
            det = pixel_x * line_y - pixel_y * line_x

            if abs(det) > ROUNDING_TOLERANCE:
                inv_a = line_y / det
                inv_b = -line_x / det
                inv_c = -pixel_y / det
                inv_d = pixel_x / det
                b_x = lines[i].x1 - (pos_x - dir_x)
                b_y = lines[i].y1 - (pos_y - dir_y)
                param_0 = inv_a * b_x + inv_b * b_y
                param_1 = inv_c * b_x + inv_d * b_y

                if 0 <= param_1 <= 1 and 1 + ROUNDING_TOLERANCE < param_0 \
                        and (param_0 < pixel_value[1] or pixel_value[1] == -1):
                    pixel_value = (lines[i].c, param_0)

        i += 1

    bitmap[index, 0] = pixel_value[0]
    bitmap[index, 1] = pixel_value[1]


class GameAdapter(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'n_agents') and
                callable(subclass.n_agents) and
                hasattr(subclass, 'reset') and
                callable(subclass.reset) and
                hasattr(subclass, 'step') and
                callable(subclass.step) and
                hasattr(subclass, 'observations') and
                callable(subclass.observations) and
                hasattr(subclass, 'actions') and
                callable(subclass.actions) and
                NotImplemented)

    @abc.abstractmethod
    def n_agents(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> list[list[float]]:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, actions: list[int]) -> tuple[list[list[float]], list[float], bool]:
        raise NotImplementedError

    @abc.abstractmethod
    def observations(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def actions(self) -> int:
        raise NotImplementedError


class GameAdapterImpl(GameAdapter):
    def __init__(self):
        super().__init__()

        self.game_ctr = 0
        self.__frame_ctr = 0
        self.__inputs = N_NV_INPUTS
        self.__v_inputs = VISUAL_RESOLUTION * 2
        self.__observations = self.__v_inputs + self.__inputs
        self.__actions = N_ACTIONS
        self.__controller = Controller()
        self.__n_agents = self.__controller.n_actors
        self.__comm_channels = [[False] * _COMM_CHANNELS] * self.__n_agents

        bitmap = np.zeros([VISUAL_RESOLUTION, 2], dtype=np.float64)
        self.__d_bitmap = cuda.to_device(bitmap)
        self.__d_lines = None

        self.__frame_stats = []
        self.__stats_file = '../resources/stats/stats.csv'

        with open(self.__stats_file, 'w+', newline='') as stats_file:
            csv_writer = csv.writer(stats_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(STATS_HEADER)

    def n_agents(self):
        return self.__n_agents

    def __visual_input_cuda(self, actor):
        ray_tracing_kernel[_GRIDS, _THREADS](actor.id, actor.team,
                                             actor.position[0][0], actor.position[1][0],
                                             actor.direction[0][0], actor.direction[1][0],
                                             self.__d_lines, self.__d_bitmap)

        bitmap = self.__d_bitmap.copy_to_host()

        return [(p[0], p[1]) for p in bitmap]

    def __set_scene(self):
        lines = []

        for obstacle in self.__controller.world.obstacles:
            points = obstacle.shape.get_points()
            for i in range(len(points)):
                line = (points[i][0][0],
                        points[i][1][0],
                        points[(i + 1) % 4][0][0],
                        points[(i + 1) % 4][1][0],
                        0, -1, -1)
                lines.append(line)

        for item in self.__controller.world.items:
            points = item.shape.get_points()
            for i in range(len(points)):
                line = (points[i][0][0],
                        points[i][1][0],
                        points[(i + 1) % 4][0][0],
                        points[(i + 1) % 4][1][0],
                        1, -1, -1)
                lines.append(line)

        for actor in self.__controller.world.actors:
            line = (actor.position[0][0] + actor.direction[0][0],
                    actor.position[1][0] + actor.direction[1][0],
                    actor.position[0][0] - actor.direction[0][0],
                    actor.position[1][0] - actor.direction[1][0],
                    2 + actor.team, actor.id, actor.team)
            lines.append(line)

            ortho_x = -actor.direction[1][0]
            ortho_y = actor.direction[0][0]
            line = (actor.position[0][0] + ortho_x,
                    actor.position[1][0] + ortho_y,
                    actor.position[0][0] - ortho_x,
                    actor.position[1][0] - ortho_y,
                    2 + actor.team, actor.id, actor.team)
            lines.append(line)

            front = actor.weapon.front

            shield = actor.position + math.rotate(actor.direction, actor.weapon.block_area)
            line = (shield[0][0], shield[1][0], front[0][0], front[1][0],
                    4 + actor.team if actor.weapon.blocking else 2 + actor.team, actor.id, actor.team)
            lines.append(line)

            shield = actor.position + math.rotate(actor.direction, -actor.weapon.block_area)
            line = (shield[0][0], shield[1][0], front[0][0], front[1][0],
                    4 + actor.team if actor.weapon.blocking else 2 + actor.team, actor.id, actor.team)
            lines.append(line)

        self.__d_lines = cuda.to_device(np.array(lines, dtype=Line_t))

    def __observation(self):
        inputs = [None] * self.__n_agents

        for actor in self.__controller.world.actors:
            visual_inputs = self.__visual_input_cuda(actor)
            actor_inputs = [v_i[0] for v_i in visual_inputs] + [v_i[1] for v_i in visual_inputs]

            # uncomment if the visual input of an actor should be drawn
            """if actor.id == 0:
                self.__controller.draw_eye(actor_inputs)"""

            # blocking
            actor_inputs.append(float(actor.weapon.blocking))

            # attacking
            actor_inputs.append(float(actor.weapon.attacking))

            # number of healing potions
            actor_inputs.append(actor.num_items('healing_potion'))

            # communication channels
            actor_inputs.extend(self.__comm_channels[actor.id])

            # health points
            hp = [-1] * N_AGENTS

            for other in self.__controller.world.actors:
                hp[other.id] = other.hit_points

            actor_inputs.extend(hp)

            inputs[actor.id] = actor_inputs

        for agent in range(self.__n_agents):
            if inputs[agent] is None:
                inputs[agent] = _EMPTY_INPUT

        return inputs

    def __control_actors(self, actions):
        for actor in self.__controller.world.actors:
            if actions[actor.id] == 0:
                actor.move(Direction.FORWARD)
            elif actions[actor.id] == 1:
                actor.turn(_ACTOR_TURN_SPEED)
                # uncomment the second argument if agents should not be able to turn right after 50 games
            elif actions[actor.id] == 2:  # or self.game_ctr > 50:
                actor.turn(-_ACTOR_TURN_SPEED)
            elif actions[actor.id] == 3:
                actor.weapon.attack()
            elif actions[actor.id] == 4:
                actor.weapon.blocking = True
            elif actions[actor.id] == 5:
                actor.weapon.blocking = False
            elif actions[actor.id] == 6:
                actor.use_item('healing_potion', actor)
            elif actions[actor.id] == 7:
                team_mate = None

                for other in self.__controller.world.actors:
                    if other.team == actor.team and other != actor:
                        team_mate = other

                if team_mate is not None:
                    actor.use_item('healing_potion', team_mate)
            elif 7 < actions[actor.id] <= 7 + _COMM_CHANNELS:
                # set communication channels

                set_channel = actions[actor.id] - 8

                # if agents should not be able to communicate, comment out this for-loop
                for other in self.__controller.world.actors:
                    if other != actor and other.team == actor.team:  # and actor.team == 0:
                        self.__comm_channels[other.id][set_channel] = True
            elif 7 + _COMM_CHANNELS < actions[actor.id] <= 7 + 2 * _COMM_CHANNELS:
                # reset communication channels

                reset_channel = actions[actor.id] - 8 - _COMM_CHANNELS

                for other in self.__controller.world.actors:
                    if other != actor and other.team == actor.team:  # and actor.team == 0:
                        self.__comm_channels[other.id][reset_channel] = False

    def __dump_stats(self):
        with open(self.__stats_file, 'a', newline='') as stats_file:
            csv_writer = csv.writer(stats_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for frame in self.__frame_stats:
                csv_writer.writerow(frame)

            self.__frame_stats = []

    def reset(self, randomize=False):
        self.__frame_ctr = 0

        self.__controller = Controller(randomize)

        self.__set_scene()

        return self.__observation()

    def step(self, actions, record_stats=False):
        self.__frame_ctr += 1

        rewards = [0.] * self.__n_agents
        team_rewards = [0.] * self.__controller.world.teams
        frame_stats = [0] * N_STATS * self.__n_agents
        # set all dead stats to True
        for agent in range(self.__n_agents):
            frame_stats[agent * N_STATS + 8] = 1

        positions = {actor.id: actor.position.copy() for actor in self.__controller.world.actors}
        stats = {actor.id: actor.stats.copy() for actor in self.__controller.world.actors}

        # for death detection
        ids = {actor.id: True for actor in self.__controller.world.actors}

        self.__control_actors(actions)
        self.__controller.update()
        self.__set_scene()

        for actor in self.__controller.world.actors:
            ids[actor.id] = False
            frame_stats[actor.id * N_STATS + 8] = 0

            team_mates = [other.id for other in self.__controller.world.actors if other.team == actor.team]

            if (0 > positions[actor.id][0][0] < actor.position[0][0]
                    or 0 < positions[actor.id][0][0] > actor.position[0][0]):
                rewards[actor.id] += .01
                # frame_stats[actor.id * N_STATS] = 1
            elif positions[actor.id][0][0] != actor.position[0][0]:
                rewards[actor.id] -= .01
            if (0 > positions[actor.id][1][0] < actor.position[1][0]
                    or 0 < positions[actor.id][1][0] > actor.position[1][0]):
                rewards[actor.id] += .01
                # frame_stats[actor.id * N_STATS] = 1
            elif positions[actor.id][1][0] != actor.position[1][0]:
                rewards[actor.id] -= .01

            if rewards[actor.id] > 0.:
                frame_stats[actor.id * N_STATS] = 1

            if stats[actor.id]['hits_taken'] < actor.stats['hits_taken']:
                rewards[actor.id] -= .5
                frame_stats[actor.id * N_STATS + 1] = 1

            if stats[actor.id]['blocked'] < actor.stats['blocked']:
                rewards[actor.id] += 1.
                frame_stats[actor.id * N_STATS + 2] = 1

            if stats[actor.id]['hits'] < actor.stats['hits']:
                # team_rewards[actor.team] += 10.
                rewards[actor.id] += 10.
                for team_mate in team_mates:
                    rewards[team_mate] += 20.
                frame_stats[actor.id * N_STATS + 3] = 1

            if stats[actor.id]['team_hits'] < actor.stats['team_hits']:
                # rewards[actor.id] -= .01
                team_rewards[actor.team] -= .01
                frame_stats[actor.id * N_STATS + 4] = 1

            if stats[actor.id]['kills'] < actor.stats['kills']:
                # team_rewards[actor.team] += 10.
                rewards[actor.id] += 10.
                for team_mate in team_mates:
                    rewards[team_mate] += 20.
                frame_stats[actor.id * N_STATS + 5] = 1

            if stats[actor.id]['team_kills'] < actor.stats['team_kills']:
                # rewards[actor.id] -= .1
                team_rewards[actor.team] -= .1
                frame_stats[actor.id * N_STATS + 6] = 1

            if stats[actor.id]['items_collected'] < actor.stats['items_collected']:
                rewards[actor.id] += .1
                frame_stats[actor.id * N_STATS + 7] = 1

            if stats[actor.id]['healed_self'] < actor.stats['healed_self']:
                rewards[actor.id] += actor.stats['healed_self'] - stats[actor.id]['healed_self']
                frame_stats[actor.id * N_STATS + 9] = \
                    actor.stats['healed_self'] - stats[actor.id]['healed_self']

            if stats[actor.id]['healed_other'] < actor.stats['healed_other']:
                rewards[actor.id] += (actor.stats['healed_other'] - stats[actor.id]['healed_other']) * 10
                frame_stats[actor.id * N_STATS + 10] = \
                    actor.stats['healed_other'] - stats[actor.id]['healed_other']

            frame_stats[actor.id * N_STATS + 11] = actor.hit_points

            if stats[actor.id]['items_used'] < actor.stats['items_used']:
                frame_stats[actor.id * N_STATS + 12] = 1

        for k, v in ids.items():
            if v:
                rewards[k] -= 10.

        for actor in self.__controller.world.actors:
            rewards[actor.id] += team_rewards[actor.team]

        # add stats for current frame
        stats = [self.game_ctr, self.__frame_ctr]
        stats.extend(frame_stats)
        stats.extend(rewards)
        if record_stats:
            self.__frame_stats.append(stats)

        # dump stats to file if game ended or frame limit reached
        if self.__frame_ctr > _TRUNCATION or self.__frame_ctr % _DUMP_EVERY_FRAME == 0:
            self.__dump_stats()

        return self.__observation(), rewards, self.__controller.won or self.__frame_ctr > _TRUNCATION

    def observations(self):
        return self.__observations

    def v_inputs(self):
        return self.__v_inputs

    def inputs(self):
        return self.__inputs

    def actions(self):
        return self.__actions

    def quit(self):
        return not self.__controller.running
