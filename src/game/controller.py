"""
This file implements the controller for the game according do the MVC structure.

The game can be zoomed in and out by the mouse wheel.

The camera can be moved by the W, S, A, D keys.

The camera can be rotated by the left and right keys.
"""

from enum import Enum

import numpy as np
import pygame

from main import PLAYER_MODE
from game.model import World, Direction
from game.view import View, WIDTH, HEIGHT
from game.animation import AnimationController

# the speed with which the mouse for a player turns the actor, if player is used
MOUSE_SPEED = 0.01

# change fps limit to reduce fps
_FPS = 120 if PLAYER_MODE else 0


class GameEvents(Enum):
    MOVE_UP = 0,
    MOVE_DOWN = 1,
    MOVE_LEFT = 2,
    MOVE_RIGHT = 3,
    TURN_LEFT = 4,
    TURN_RIGHT = 5,
    ZOOM_IN = 7,
    ZOOM_OUT = 8


key_map = {
    pygame.K_w: GameEvents.MOVE_UP,
    pygame.K_s: GameEvents.MOVE_DOWN,
    pygame.K_a: GameEvents.MOVE_LEFT,
    pygame.K_d: GameEvents.MOVE_RIGHT,
    pygame.K_UP: GameEvents.ZOOM_IN,
    pygame.K_DOWN: GameEvents.ZOOM_OUT,
    pygame.K_LEFT: GameEvents.TURN_LEFT,
    pygame.K_RIGHT: GameEvents.TURN_RIGHT
}


class Controller:
    def __init__(self, randomize=False):
        self.__world = World(randomize)
        self.__view = View()
        self.__animation_controller = AnimationController()
        self.__running = True
        self.__key_flags = {
            GameEvents.MOVE_UP: False,
            GameEvents.MOVE_DOWN: False,
            GameEvents.MOVE_LEFT: False,
            GameEvents.MOVE_RIGHT: False,
            GameEvents.TURN_LEFT: False,
            GameEvents.TURN_RIGHT: False,
            GameEvents.ZOOM_IN: False,
            GameEvents.ZOOM_OUT: False
        }
        self.__follow_actor = 0 if PLAYER_MODE else -1
        self.__control_actor = 0 if PLAYER_MODE else -1
        self.__clock = pygame.time.Clock()
        self.__ticks = pygame.time.get_ticks()

        pygame.init()

        if PLAYER_MODE:
            pygame.mouse.set_visible(False)

        for actor in self.__world.actors:
            actor.weapon.register(self.__animation_controller)

        self.__view.zoom_camera(1.)

    @property
    def world(self):
        return self.__world

    @property
    def running(self):
        return self.__running

    @property
    def won(self):
        return (all(actor.team == 0 for actor in self.__world.actors)
                or all(actor.team == 1 for actor in self.__world.actors))

    @property
    def n_actors(self):
        return len(self.__world.actors)

    def handle_input_output(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.__running = False
            if event.type == pygame.KEYDOWN:
                mod_ctrl = pygame.key.get_mods() & pygame.KMOD_CTRL

                if event.key == pygame.K_1 and not mod_ctrl:
                    self.__follow_actor = 0
                elif event.key == pygame.K_2 and not mod_ctrl:
                    self.__follow_actor = 1
                elif event.key == pygame.K_0 and not mod_ctrl:
                    self.__follow_actor = -1
                elif event.key == pygame.K_1 and mod_ctrl:
                    self.__control_actor = 0
                elif event.key == pygame.K_2 and mod_ctrl:
                    self.__control_actor = 1
                elif event.key == pygame.K_0 and mod_ctrl:
                    self.__control_actor = -1
                elif event.key in key_map and key_map[event.key] in self.__key_flags:
                    self.__key_flags[key_map[event.key]] = True
                elif event.key == pygame.K_ESCAPE:
                    self.__running = False
            if event.type == pygame.KEYUP:
                if event.key in key_map and key_map[event.key] in self.__key_flags:
                    self.__key_flags[key_map[event.key]] = False
            if event.type == pygame.MOUSEBUTTONDOWN and self.__control_actor >= 0:
                if event.button == pygame.BUTTON_LEFT:
                    self.__world.actors[self.__control_actor].weapon.attack()
                elif event.button == pygame.BUTTON_RIGHT:
                    self.__world.actors[self.__control_actor].weapon.blocking = True
            if event.type == pygame.MOUSEBUTTONUP and self.__control_actor >= 0:
                if event.button == pygame.BUTTON_RIGHT:
                    self.__world.actors[self.__control_actor].weapon.blocking = False
            if event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self.__view.zoom_camera(1.1)
                elif event.y < 0:
                    self.__view.zoom_camera(0.9)

        if self.__control_actor >= 0:
            if self.__key_flags[GameEvents.MOVE_UP]:
                self.__world.actors[self.__control_actor].move(Direction.FORWARD)
            if self.__key_flags[GameEvents.MOVE_LEFT]:
                self.__world.actors[self.__control_actor].move(Direction.LEFT)
            if self.__key_flags[GameEvents.MOVE_DOWN]:
                self.__world.actors[self.__control_actor].move(Direction.BACKWARD)
            if self.__key_flags[GameEvents.MOVE_RIGHT]:
                self.__world.actors[self.__control_actor].move(Direction.RIGHT)
            if self.__key_flags[GameEvents.TURN_LEFT]:
                self.__world.actors[self.__control_actor].turn(0.1)
            if self.__key_flags[GameEvents.TURN_RIGHT]:
                self.__world.actors[self.__control_actor].turn(-0.1)

            self.__world.actors[self.__control_actor] \
                .turn((WIDTH / 2.0 - pygame.mouse.get_pos()[0]) * MOUSE_SPEED)

        if PLAYER_MODE:
            pygame.mouse.set_pos(WIDTH / 2.0, HEIGHT / 2.0)

        if self.__follow_actor < 0:
            if self.__control_actor < 0:
                if self.__key_flags[GameEvents.MOVE_UP]:
                    self.__view.move_camera(np.array([[0.0], [3.0]]))
                if self.__key_flags[GameEvents.MOVE_LEFT]:
                    self.__view.move_camera(np.array([[-3.0], [0.0]]))
                if self.__key_flags[GameEvents.MOVE_DOWN]:
                    self.__view.move_camera(np.array([[0.0], [-3.0]]))
                if self.__key_flags[GameEvents.MOVE_RIGHT]:
                    self.__view.move_camera(np.array([[3.0], [0.0]]))
                if self.__key_flags[GameEvents.TURN_LEFT]:
                    self.__view.rotate_camera(0.1)
                if self.__key_flags[GameEvents.TURN_RIGHT]:
                    self.__view.rotate_camera(-0.1)
        else:
            self.__view.follow_actor(self.__world.actors[self.__follow_actor])

        if self.__key_flags[GameEvents.ZOOM_OUT]:
            self.__view.zoom_camera(0.9)
        if self.__key_flags[GameEvents.ZOOM_IN]:
            self.__view.zoom_camera(1.1)

    def update(self):
        self.__clock.tick(_FPS)

        self.handle_input_output()

        self.__animation_controller.update()

        self.__world.update()

        self.__view.draw(self.__world)

        if (pygame.time.get_ticks() - self.__ticks) >= 1000:
            self.__ticks = pygame.time.get_ticks()
            # print(self.__clock.get_fps())

    def start_game(self):
        self.__running = True

        while self.__running:
            self.update()

    def draw_eye(self, pixels):
        self.__view.draw_eye(pixels)
