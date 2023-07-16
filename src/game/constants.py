"""
This file contains all the relevant constants for the game.
"""

from numpy import array, pi

# the visual resolution with which the agent can see its environment
VISUAL_RESOLUTION = 128
# a tolerance to specify at which difference 2 values for the ray-tracing algorithm should be seen as equal
ROUNDING_TOLERANCE = 0.001

# the diagonal size of an item in the game
BOUNDS_ITEM = array([[2.0],
                     [2.0]])
# the radius of an actor / agent in the game
RADIUS_ACTOR = 1.0
# the initial direction vector of the actors / agents in the game
DIRECTION_ZERO = array([[1],
                        [0]])
# the position where the obstacle should spawn in the world
POS_OBSTACLE = array([[0.0],
                      [0.0]])
# the diagonal size of an obstacle in the world
BOUNDS_OBSTACLE = array([[5.0],
                         [5.0]])

# the factor with which the weapon length should be multiplied or divided for the attacking animation
WEAPON_SPEED = 1.1
# the weapon damage for a weapon
WEAPON_DMG = 1
# the blocking arc that gets covered if the agents block
WEAPON_BLOCK_AREA = pi
# the cooldown in frames that an actor / agent can not attack after finishing the attacking animation
WEAPON_COOLDOWN = 50
# the maximal length of the weapon during attacking animation
WEAPON_REACH = 3.0
# the amount of health points a healing potion heals
AMOUNT_HEALING_POTION = 3
# the moving speed of an actor / agent per frame
SPEED_NORMAL = 0.1
# the sprinting speed of an actor / agent per frame (not used!)
SPEED_SPRINT = 2.0
# the stamina cost of an actor / agent per frame if sprinting (not used!)
STAMINA_COST_SPRINT = 20
# the number of agents per team
N_AGENTS_TEAM = 2
# the number of opposing teams in the world
N_TEAMS = 2
# the number of agents in the game
N_AGENTS = N_AGENTS_TEAM * N_TEAMS

# the different types of items used in the game
ITEMS = ['healing_potion']

# the stats per agent per frame to be recorded in the CSV for evaluation of the agents for a testing game
STATS = {'hits_taken': 0, 'blocked': 0, 'hits': 0, 'team_hits': 0, 'kills': 0, 'team_kills': 0,
         'items_collected': 0, 'healed_self': 0, 'healed_other': 0, 'items_used': 0}
# number of stats per actor except reward
N_STATS = 13
# the header for the CSV that contains all the stats of all the agents (one line are the stats for one frame)
STATS_HEADER = ['game', 'frame',
                'pos_1', 'hits_taken_1', 'blocked_1', 'hits_1', 'team_hits_1', 'kills_1',
                'team_kills_1', 'items_collected_1', 'dead_1',
                'healed_self_1', 'healed_other_1', 'hit_points_1', 'items_used_1',
                'pos_2', 'hits_taken_2', 'blocked_2', 'hits_2', 'team_hits_2', 'kills_2',
                'team_kills_2', 'items_collected_2', 'dead_2',
                'healed_self_2', 'healed_other_2', 'hit_points_2', 'items_used_2',
                'pos_3', 'hits_taken_3', 'blocked_3', 'hits_3', 'team_hits_3', 'kills_3',
                'team_kills_3', 'items_collected_3', 'dead_3',
                'healed_self_3', 'healed_other_3', 'hit_points_3', 'items_used_3',
                'pos_4', 'hits_taken_4', 'blocked_4', 'hits_4', 'team_hits_4', 'kills_4',
                'team_kills_4', 'items_collected_4', 'dead_4',
                'healed_self_4', 'healed_other_4', 'hit_points_4', 'items_used_4',
                'reward_1', 'reward_2', 'reward_3', 'reward_4']
