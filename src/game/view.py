"""
This file implements the view of the game according to the MVC structure.

The view is implemented using pygame.
"""

import pygame

import numpy as np

from .constants import *
from utils import math

# the width of the window
WIDTH = 1000
# the height of the window
HEIGHT = 1000
"""the coordinates with respect to the window that the view should center on
(defines the origin to be the middle of the screen)"""
_VIEW_POINT = np.array([[WIDTH / 2.0],
                        [HEIGHT / 2.0]])
# the initial angle at which the view should be oriented at
_ANGLE_COORDINATE_SYSTEM = -np.pi / 2.0
# the background color for the game
_BACKGROUND = (100, 100, 100)
# the width of the drawn shapes (lines, circles and arcs)
_SHAPE_WIDTH = 1
# the initial zoom level, the view has
_INIT_ZOOM = 10.0
# the initial position of the camera
_INIT_CAMERA_POSITION = np.array([[0.0],
                                  [0.0]])
# the initial rotation at which the camera is rotated at
_INIT_CAMERA_ANGLE = 0.0


def to_tuple(arr):
    return arr[0][0], arr[1][0]


def draw_line(surface, start, end, camera_zoom, camera_position, camera_angle):
    transformed_start = camera_zoom * math.rotate(math.flip(start, math.Axis.Y) -
                                                  math.flip(camera_position, math.Axis.Y),
                                                  camera_angle) + _VIEW_POINT
    transformed_end = camera_zoom * math.rotate(math.flip(end, math.Axis.Y) -
                                                math.flip(camera_position, math.Axis.Y),
                                                camera_angle) + _VIEW_POINT

    pygame.draw.line(surface, (0, 255, 0), to_tuple(transformed_start), to_tuple(transformed_end))


def draw_polygon(surface, points, camera_zoom, camera_position, camera_angle):
    transformed = map(lambda point:
                      camera_zoom *
                      math.rotate(math.flip(point, math.Axis.Y) -
                                  math.flip(camera_position, math.Axis.Y),
                                  camera_angle) +
                      _VIEW_POINT, points)
    transformed = list(map(lambda point: to_tuple(point), transformed))

    pygame.draw.polygon(surface, (0, 255, 0), transformed, width=_SHAPE_WIDTH)


def draw_circle(hue, surface, position, radius, camera_zoom, camera_position, camera_angle):
    transformed = camera_zoom * math.rotate(math.flip(position, math.Axis.Y) -
                                            math.flip(camera_position, math.Axis.Y),
                                            camera_angle) + _VIEW_POINT

    if hue > 0.9:
        pygame.draw.circle(surface, (255, 0, 0), to_tuple(transformed), camera_zoom * radius, width=_SHAPE_WIDTH)
    else:
        pygame.draw.circle(surface, (0, 0, 255), to_tuple(transformed), camera_zoom * radius, width=_SHAPE_WIDTH)


def draw_arc(surface, position, radius, begin, end, camera_zoom, camera_position, camera_angle):
    transformed = camera_zoom * math.rotate(math.flip(position, math.Axis.Y) -
                                            math.flip(camera_position, math.Axis.Y),
                                            camera_angle) + _VIEW_POINT
    pos = to_tuple(transformed)
    rect = pygame.Rect(pos[0] - camera_zoom * radius, pos[1] - camera_zoom * radius,
                       2.0 * camera_zoom * radius, 2.0 * camera_zoom * radius)

    pygame.draw.arc(surface, (255, 255, 0), rect, begin - camera_angle, end - camera_angle)


class View:
    def __init__(self):
        self.__screen = pygame.display.set_mode([WIDTH, HEIGHT])
        self.__clock = pygame.time.Clock()
        self.__camera_zoom = _INIT_ZOOM
        self.__camera_position = _INIT_CAMERA_POSITION
        self.__camera_angle = _INIT_CAMERA_ANGLE + _ANGLE_COORDINATE_SYSTEM

    def rotate_camera(self, angle):
        self.__camera_angle += angle
        self.__camera_angle %= np.pi * 2.0

    def move_camera(self, velocity):
        self.__camera_position += math.rotate_axis(velocity, self.__camera_angle, np.array([[0.0], [0.0]]))

    def zoom_camera(self, factor):
        self.__camera_zoom *= factor

    def follow_actor(self, actor):
        self.__camera_position = actor.position.copy()
        self.__camera_angle = actor.angle + _ANGLE_COORDINATE_SYSTEM

    def draw(self, world):
        self.__screen.fill(_BACKGROUND)

        for actor in world.actors:
            draw_circle((actor.team + 1) / 2, self.__screen, actor.position, RADIUS_ACTOR,
                        self.__camera_zoom, self.__camera_position, self.__camera_angle)
            draw_line(self.__screen, actor.position, actor.position + actor.direction,
                      self.__camera_zoom, self.__camera_position, self.__camera_angle)
            # weapon
            draw_line(self.__screen, actor.position + actor.direction, actor.weapon.front,
                      self.__camera_zoom, self.__camera_position, self.__camera_angle)
            # shield
            if actor.weapon.blocking:
                draw_arc(self.__screen, actor.position, RADIUS_ACTOR,
                         -actor.weapon.block_area / 2.0 + actor.weapon.direction,
                         actor.weapon.block_area / 2.0 + actor.weapon.direction,
                         self.__camera_zoom, self.__camera_position, self.__camera_angle)

        for item in world.items:
            draw_polygon(self.__screen, item.shape.get_points(),
                         self.__camera_zoom, self.__camera_position, self.__camera_angle)

        for obstacle in world.obstacles:
            draw_polygon(self.__screen, obstacle.shape.get_points(),
                         self.__camera_zoom, self.__camera_position, self.__camera_angle)

        pygame.display.flip()

    def draw_eye(self, pixels):
        for i in range(len(pixels) // 2):
            color = [0, 0, 0]

            if 3 > pixels[i] != -1:
                color[int(pixels[i])] = 255
            elif int(pixels[i]) == 3:
                color[0] = 255
                color[1] = 255

            self.__screen.set_at((i, 10), color)
            self.__screen.set_at((i, int(pixels[len(pixels) // 2 + i] * 10 + 10)), color)

        pygame.display.flip()
