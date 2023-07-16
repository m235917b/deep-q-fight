"""
This file implements the model of the game according to the MVC structure.
"""

import abc
import random
from enum import Enum

import numpy as np

from .animation import Animated, Animation
from .constants import *
from utils import math


class Direction(Enum):
    """
    An enum class for specifying the direction in which a player / actor can move
    """
    FORWARD = 0
    FORWARD_LEFT = 1
    LEFT = 2
    BACKWARD_LEFT = 3
    BACKWARD = 4
    BACKWARD_RIGHT = 5
    RIGHT = 6
    FORWARD_RIGHT = 7


class Weapon(Animated):
    def __init__(self, position, direction, reach, damage, cooldown, block_area):
        super().__init__()
        self.__position = position
        self.__front_relative = math.rotate(DIRECTION_ZERO, direction)
        self.__direction = direction
        self.__reach = reach
        self.__length = RADIUS_ACTOR
        self.__damage = damage
        self.__cooldown = cooldown
        self.__block_area = block_area
        self.__block_arc = np.cos(block_area / 2.0)
        self.__blocking = False
        self.__attacking = False
        self.__attacking_animation = None

    @property
    def front(self):
        return self.__position + self.__front_relative

    @property
    def damage(self):
        return self.__damage

    @property
    def direction(self):
        return self.__direction

    @property
    def block_area(self):
        return self.__block_area

    @property
    def attacking(self):
        return self.__attacking_animation is not None

    @property
    def blocking(self):
        return self.__blocking

    @blocking.setter
    def blocking(self, val):
        if not self.__attacking:
            self.__blocking = val

    def move(self, velocity):
        self.__position += velocity

    def turn(self, angle):
        self.__direction += angle
        self.__front_relative = math.rotate(self.__front_relative, angle)

    def attack(self):
        if not self.__attacking:
            self.__attacking_animation = Animation(
                begin=lambda: self.__set_attacking(True),
                update=lambda: self.__move_front(WEAPON_SPEED),
                condition=lambda: np.linalg.norm(self.__front_relative) < self.__reach,
                next_animation=self.__back_animation()
            )

            self.start_animation(self.__attacking_animation)

    def break_attack(self):
        if self.__attacking_animation is not None:
            # cancel attack animation
            self.kill_animation(self.__attacking_animation)
            self.__attacking_animation = None

            # move weapon back
            self.start_animation(self.__back_animation())

    def blocked(self, other):
        """does only detect if weapon is inside blocking arc, not if it is also inside the actor"""
        if self.__blocking:
            diff = other.front - self.__position
            len_diff = np.linalg.norm(diff)
            len_front = np.linalg.norm(self.__front_relative)
            arc = self.__front_relative.T.dot(diff).flatten()[0] / (len_front * len_diff)

            return arc >= self.__block_arc

    def __back_animation(self):
        a_a_cooldown = Animation(
            final=lambda: self.__set_attacking(False),
            duration=self.__cooldown
        )
        return Animation(
            update=lambda: self.__move_front(1.0 / WEAPON_SPEED),
            condition=lambda: np.linalg.norm(self.__front_relative) > self.__length,
            next_animation=a_a_cooldown
        )

    def __set_attacking(self, attacking):
        """used for attack animation"""
        self.__attacking = attacking
        self.__blocking = not attacking

    def __move_front(self, velocity):
        """used for attack animation"""
        self.__front_relative *= velocity


class Shape(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'closest_point') and
                callable(subclass.closest_point) and
                hasattr(subclass, 'is_inside') and
                callable(subclass.is_inside) and
                hasattr(subclass, 'collision_point_normal') and
                callable(subclass.collision_point_normal) and
                hasattr(subclass, 'move') and
                callable(subclass.move) or
                NotImplemented)

    def __init__(self, position):
        self.position = position

    @abc.abstractmethod
    def closest_point(self, other: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def is_inside(self, point: np.ndarray) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def collision_point_normal(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def move(self, velocity: np.ndarray):
        raise NotImplementedError


class Box(Shape):
    def __init__(self, position, dimensions):
        if dimensions[0] <= 0 and dimensions[1] <= 0:
            raise ValueError('dimensions for bounds must be positive')

        super().__init__(position)
        self.__dimensions = dimensions
        self.__bounds = [position + 0.5 * np.array([[-dimensions[0][0]],
                                                    [-dimensions[1][0]]]),
                         position + 0.5 * np.array([[dimensions[0][0]],
                                                    [-dimensions[1][0]]]),
                         position + 0.5 * np.array([[dimensions[0][0]],
                                                    [dimensions[1][0]]]),
                         position + 0.5 * np.array([[-dimensions[0][0]],
                                                    [dimensions[1][0]]])]

    def closest_point(self, other):
        diff = self.position - other

        if abs(diff[0][0]) >= self.__dimensions[0][0] / 2.0 and abs(diff[1][0]) >= self.__dimensions[1][0] / 2.0:
            return min(self.__bounds, key=lambda vertex: np.linalg.norm(vertex - other))

        closest = min(self.__bounds, key=lambda vertex: np.linalg.norm(vertex - other)) - other

        if (abs(diff[0][0]) >= self.__dimensions[0][0] / 2.0 or
                abs(closest[0][0]) > abs(closest[1][0]) and
                not abs(diff[1][0]) >= self.__dimensions[1][0] / 2.0):
            closest[1][0] = 0.0
        else:
            closest[0][0] = 0.0

        return other + closest

    def is_inside(self, point):
        diff = point - self.position
        xy = self.__dimensions / 2.0
        return abs(diff[0][0]) <= xy[0][0] and abs(diff[1][0]) <= xy[1][0]

    def collision_point_normal(self, point):
        return self.closest_point(point) - point

    def get_points(self):
        return self.__bounds

    def move(self, velocity):
        self.position += velocity
        self.__bounds = list(map(lambda vertex: vertex + velocity, self.__bounds))


class Circle(Shape):
    def __init__(self, position, radius):
        if radius <= 0:
            raise ValueError('radius must be positive')

        super().__init__(position)
        self.__radius = radius

    def closest_point(self, other):
        diff = other - self.position
        if np.linalg.norm(diff) == 0.0:
            return self.position + np.array([[self.__radius], [0.0]])
        normed = diff / np.linalg.norm(diff)
        return self.position + self.__radius * normed

    def is_inside(self, point):
        return np.linalg.norm(self.position - point) <= self.__radius

    def collision_point_normal(self, point):
        return self.closest_point(point) - point

    def edges(self, viewing_direction: np.ndarray):
        pass

    def move(self, velocity):
        self.position += velocity


class Object:
    def __init__(self, shape):
        self.__shape = shape

    @property
    def shape(self):
        return self.__shape

    @property
    def position(self):
        return self.__shape.position

    @position.setter
    def position(self, pos):
        self.__shape.position = pos

    def collides_with(self, other):
        if self.__shape.__class__ == Circle:
            return self.__shape.is_inside(other.__shape.closest_point(self.__shape.position))
        else:
            return other.__shape.is_inside(self.__shape.closest_point(other.position))

    def bounce(self, other):
        if self.__shape.__class__ == Circle:
            closest_point = other.__shape.closest_point(self.__shape.position)
            if self.__shape.is_inside(closest_point):
                self.__shape.move(-self.__shape.collision_point_normal(closest_point))
        else:
            closest_point = self.__shape.closest_point(other.position)
            if other.__shape.is_inside(closest_point):
                self.__shape.move(self.__shape.collision_point_normal(closest_point))

    def move(self, velocity):
        self.__shape.move(velocity)


class Actor(Object):
    id_ctr = 0

    def __init__(self, team, position, direction, max_hit_points, max_stamina, max_mana):
        super().__init__(Circle(position, RADIUS_ACTOR))

        self.__id = Actor.id_ctr
        Actor.id_ctr += 1

        self.__team = team

        self.__direction = math.rotate(DIRECTION_ZERO, direction)
        self.__max_hit_points = max_hit_points
        self.__hit_points = max_hit_points
        self.__max_stamina = max_stamina
        self.__stamina = max_stamina
        self.__max_mana = max_mana
        self.__mana = max_mana
        self.sprinting = False
        self.weapon = Weapon(position, direction, RADIUS_ACTOR + WEAPON_REACH,
                             WEAPON_DMG, WEAPON_COOLDOWN, WEAPON_BLOCK_AREA)
        self.stats = STATS.copy()
        self.__inventory = {item: [] for item in ITEMS}

    def move(self, direction):
        left = np.array([[-self.__direction[1][0]],
                         [self.__direction[0][0]]])
        speed = SPEED_SPRINT if self.sprinting and not self.remove_stamina(STAMINA_COST_SPRINT) else SPEED_NORMAL

        direction = (self.__direction * speed
                     if direction == Direction.FORWARD else
                     left * speed
                     if direction == Direction.LEFT else
                     -self.__direction * speed
                     if direction == Direction.BACKWARD else
                     -left * speed)

        self.shape.move(direction)
        self.weapon.move(direction)

    def turn(self, angle):
        self.__direction = math.rotate(self.__direction, angle)
        self.weapon.turn(angle)

    def hit(self, weapon, team):
        if weapon.attacking and self.shape.is_inside(weapon.front):
            if not team:
                weapon.break_attack()
            if not self.weapon.blocked(weapon):
                if not team:
                    self.remove_hit_points(weapon.damage)
                return True
            else:
                self.stats['blocked'] += 1

        return False

    def add_hit_points(self, hit_points):
        if hit_points >= 0:
            if self.__hit_points + hit_points <= self.__max_hit_points:
                self.__hit_points += hit_points
            else:
                self.__hit_points = self.__max_hit_points

    def remove_hit_points(self, hit_points):
        self.__hit_points -= hit_points

    def add_stamina(self, amount):
        if amount >= 0:
            if self.__stamina + amount <= self.__max_stamina:
                self.__stamina += amount
            else:
                self.__stamina = self.__max_stamina

    def remove_stamina(self, amount):
        if amount >= 0:
            if self.__stamina - amount <= 0:
                return True

            if self.__stamina - amount <= 0:
                self.__stamina = 0
            else:
                self.__stamina -= amount

    @property
    def id(self):
        return self.__id

    @property
    def team(self):
        return self.__team

    @property
    def direction(self):
        return self.__direction

    @property
    def angle(self):
        direction = (self.__direction[0][0]
                     if abs(self.__direction[0][0]) <= 1.0 else
                     1.0 * np.sign(self.__direction[0][0]))
        arc = np.arccos(direction)

        return arc if self.__direction[1][0] >= 0 else 2. * np.pi - arc

    def add_item(self, item):
        self.__inventory.get(item.name).append(item)

    def use_item(self, name, target):
        items = self.__inventory.get(name)

        if items:
            hp_before = target.hit_points

            items.pop(0)(target)

            self.stats['items_used'] += 1
            if target == self:
                self.stats['healed_self'] += target.hit_points - hp_before
            else:
                self.stats['healed_other'] += target.hit_points - hp_before

    @property
    def hit_points(self):
        """only for tests"""
        return self.__hit_points

    @property
    def is_dead(self):
        return self.__hit_points <= 0

    @property
    def max_hit_points(self):
        return self.__max_hit_points

    def num_items(self, name):
        return len(self.__inventory.get(name))


class Effect(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, '__call__') and callable(subclass.__call__) or NotImplemented

    @abc.abstractmethod
    def __call__(self, actor: Actor):
        raise NotImplementedError


class Item(Object):
    def __init__(self, position, name):
        super().__init__(Box(position, BOUNDS_ITEM))
        self.__name = name

    @property
    def name(self):
        return self.__name


class HealingPotion(Item, Effect):
    def __init__(self, position):
        super().__init__(position, 'healing_potion')

    def __call__(self, actor: Actor):
        actor.add_hit_points(AMOUNT_HEALING_POTION)


class World:
    def __init__(self, randomize=False):
        Actor.id_ctr = 0

        if randomize:
            coordinate_range = list(range(-15, -4, 2))
            coordinate_range.extend(range(4, 15, 2))
            # coordinates = random.sample(coordinate_range, 18)
            coordinates = set()
            while len(coordinates) < 18:
                coordinates.add(tuple(random.sample(coordinate_range, 2)))
            coordinates = list(coordinates)
            positions = [np.array([[float(c[0])], [float(c[1])]])
                         for c in coordinates]
        else:
            pos_0 = np.array([[15.0],
                              [15.0]])
            pos_1 = np.array([[15.0],
                              [-15.0]])
            pos_2 = np.array([[-15.0],
                              [15.0]])
            pos_3 = np.array([[-15.0],
                              [-15.0]])
            pos_4 = np.array([[10.0],
                              [10.0]])
            pos_5 = np.array([[-10.0],
                              [10.0]])
            pos_6 = np.array([[10.0],
                              [-10.0]])
            pos_7 = np.array([[-10.0],
                              [10.0]])
            pos_8 = np.array([[0.0],
                              [7.0]])
            positions = [pos_0, pos_1, pos_2, pos_3, pos_4, pos_5, pos_6, pos_7, pos_8]

        a1 = Actor(0, positions[0], 0.0, 6, 150, 10)
        a2 = Actor(0, positions[1], 0.0, 6, 150, 10)
        a3 = Actor(1, positions[2], 0.0, 6, 150, 10)
        a4 = Actor(1, positions[3], 0.0, 6, 150, 10)

        i1 = HealingPotion(positions[4])
        i2 = HealingPotion(positions[5])
        i3 = HealingPotion(positions[6])
        i4 = HealingPotion(positions[7])
        i5 = HealingPotion(positions[8])

        self.__actors = [a1, a2, a3, a4]
        self.__items = [i1, i2, i3, i4, i5]

        # use respective lines for setting the obstacle in the world, or not
        self.__obstacles = [Object(Box(POS_OBSTACLE.copy(), BOUNDS_OBSTACLE.copy()))]
        # self.__obstacles = []

        self.__teams = N_TEAMS

    @property
    def actors(self):
        return self.__actors

    @property
    def items(self):
        return self.__items

    @property
    def obstacles(self):
        return self.__obstacles

    @property
    def teams(self):
        return self.__teams

    def update(self):
        # shuffle the lists, so no npc or player gets an unfair advantage in the order of execution
        for actor in random.sample(self.__actors, len(self.__actors)):
            for obstacle in self.__obstacles:
                actor.bounce(obstacle)
                if obstacle.shape.is_inside(actor.weapon.front):
                    actor.weapon.break_attack()
            for other in random.sample(self.__actors, len(self.__actors)):
                if actor != other:
                    actor.bounce(other)
                    # if other.team != actor.team and other.hit(actor.weapon):
                    if other.hit(actor.weapon, other.team == actor.team):
                        other.stats['hits_taken'] += 1
                        if other.team == actor.team:
                            actor.stats['team_hits'] += 1
                        else:
                            actor.stats['hits'] += 1
                        if other.is_dead:
                            if other.team == actor.team:
                                actor.stats['team_kills'] += 1
                            else:
                                actor.stats['kills'] += 1
                            self.actors.remove(other)
            for item in self.__items:
                if actor.collides_with(item):
                    actor.stats['items_collected'] += 1
                    actor.add_item(item)
                    self.__items.remove(item)
