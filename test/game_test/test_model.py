"""
This file contains some tests for the game model.
"""

import unittest

import numpy as np

from game import constants
from game import model
from game.animation import AnimationController


class TestHealingPotion(unittest.TestCase):
    pos1 = np.array([[0.0],
                     [0.0]])
    pos2 = np.array([[0.0],
                     [0.0]])

    def test_healing_max(self):
        x = model.Actor(0, self.pos1, 5.0, 5, 150, 10)
        y = model.HealingPotion(self.pos2)
        x.remove_hit_points(2)
        y(x)
        self.assertGreaterEqual(x.max_hit_points, x.hit_points)

    def test_healing(self):
        x = model.Actor(0, self.pos1, 5.0, 5, 150, 10)
        y = model.HealingPotion(self.pos2)
        x.remove_hit_points(4)
        y(x)
        self.assertEqual(x.hit_points, 4)


class TestActorAttributes(unittest.TestCase):
    pos1 = np.array([[0.0],
                     [0.0]])

    def test_add_health_neg(self):
        x = model.Actor(0, self.pos1, 5.0, 5, 150, 10)
        x.add_hit_points(-3)
        self.assertEqual(x.hit_points, x.max_hit_points)


class TestCollision(unittest.TestCase):
    pos_0 = np.array([[0.0],
                      [0.0]])

    def test_no_collision_circles(self):
        pos = np.array([[2.0 * np.cos(np.pi / 4) + 0.1],
                        [2.0 * np.sin(np.pi / 4)]])
        x = model.Actor(0, self.pos_0, 5.0, 5, 150, 10)
        z = model.Actor(0, pos, 5.0, 5, 150, 10)
        self.assertFalse(x.collides_with(z))
        self.assertEqual(x.collides_with(z), z.collides_with(x))

    def test_collision_circles(self):
        pos = np.array([[2.0 * np.cos(np.pi / 4)],
                        [2.0 * np.sin(np.pi / 4)]])
        x = model.Actor(0, self.pos_0, 5.0, 5, 150, 10)
        z = model.Actor(0, pos, 5.0, 5, 150, 10)
        self.assertTrue(x.collides_with(z))
        self.assertEqual(x.collides_with(z), z.collides_with(x))

    def test_no_collision_circle_box(self):
        pos = np.array([[1.0 + np.cos(np.pi / 4) + 0.1],
                        [1.0 + np.sin(np.pi / 4)]])
        x = model.Actor(0, self.pos_0, 5.0, 5, 150, 10)
        y = model.HealingPotion(pos)
        self.assertFalse(x.collides_with(y))
        self.assertEqual(x.collides_with(y), y.collides_with(x))

    def test_collision_circle_box(self):
        pos = np.array([[1.0 + np.cos(np.pi / 4)],
                        [1.0 + np.sin(np.pi / 4)]])
        x = model.Actor(0, self.pos_0, 5.0, 5, 150, 10)
        y = model.HealingPotion(pos)
        self.assertTrue(x.collides_with(y))
        self.assertEqual(x.collides_with(y), y.collides_with(x))

    def test_circle_arc(self):
        pos = np.array([[0.2],
                        [1.8]])
        x = model.Actor(0, self.pos_0, 5.0, 5, 150, 10)
        y = model.HealingPotion(pos)
        self.assertTrue(x.collides_with(y))
        self.assertTrue(y.collides_with(x))

    def test_circle_box_small_y(self):
        pos = np.array([[10.0],
                        [0.9]])
        x = model.Actor(0, self.pos_0, 5.0, 5, 150, 10)
        y = model.HealingPotion(pos)
        self.assertFalse(x.collides_with(y))
        self.assertFalse(y.collides_with(x))


class TestWeaponBlock(unittest.TestCase):
    pos_weapon = np.array([[1.0],
                           [1.0]])
    pos_actor = np.array([[1.0 + constants.RADIUS_ACTOR * 1.5],
                          [1.0]])
    weapon = model.Weapon(pos_weapon, 0.0, 2.0, 2, 5, np.pi / 4.0)

    def test_non_blocking(self):
        actor = model.Actor(0, self.pos_actor, np.pi / 2.0, 5, 150, 10)
        self.assertFalse(actor.weapon.blocked(self.weapon))
        actor.weapon.blocking = True
        self.assertFalse(actor.weapon.blocked(self.weapon))

    def test_blocking(self):
        actor = model.Actor(0, self.pos_actor, np.pi, 5, 150, 10)
        actor.weapon.blocking = True
        self.assertTrue(actor.weapon.blocked(self.weapon))

    def test_blocking_moved_up(self):
        up = np.array([[0.0], [0.1]])
        actor = model.Actor(0, self.pos_actor + up, np.pi, 5, 150, 10)
        actor.weapon.blocking = True
        self.assertTrue(actor.weapon.blocked(self.weapon))

    def test_blocking_moved_down(self):
        down = np.array([[0.0], [-0.1]])
        actor = model.Actor(0, self.pos_actor + down, np.pi, 5, 150, 10)
        actor.weapon.blocking = True
        self.assertTrue(actor.weapon.blocked(self.weapon))

    def test_non_blocking_moved_up(self):
        up = np.array([[0.0], [0.51]])
        actor = model.Actor(0, self.pos_actor + up, np.pi, 5, 150, 10)
        actor.weapon.blocking = True
        self.assertFalse(actor.weapon.blocked(self.weapon))

    def test_non_blocking_moved_down(self):
        down = np.array([[0.0], [-0.51]])
        actor = model.Actor(0, self.pos_actor + down, np.pi, 5, 150, 10)
        actor.weapon.blocking = True
        self.assertFalse(actor.weapon.blocked(self.weapon))


class TestWeaponHit(unittest.TestCase):
    pos_weapon = np.array([[1.0],
                           [1.0]])
    pos_actor = np.array([[1.0 + constants.RADIUS_ACTOR * 1.5],
                          [1.0]])
    weapon = model.Weapon(pos_weapon, 0.0, 2.0, 6, 5, np.pi / 4.0)
    controller = AnimationController()
    weapon.register(controller)

    def test_non_blocking(self):
        actor = model.Actor(0, self.pos_actor, np.pi / 2.0, 5, 150, 10)
        self.weapon.attack()
        self.assertTrue(actor.hit(self.weapon))

    def test_blocking_hit(self):
        actor = model.Actor(0, self.pos_actor, np.pi / 2.0, 5, 150, 10)
        actor.weapon.blocking = True
        self.weapon.attack()
        self.assertTrue(actor.hit(self.weapon))

    def test_blocking(self):
        actor = model.Actor(0, self.pos_actor, np.pi, 5, 150, 10)
        actor.weapon.blocking = True
        self.weapon.attack()
        self.assertFalse(actor.hit(self.weapon))
        self.assertEqual(5, actor.hit_points)

    def test_kill_animation(self):
        actor = model.Actor(0, np.array([[-4.0], [0.0]]), 0.0, 1, 10, 10)
        obs = model.Object(model.Box(np.array([[0.0], [0.0]]), np.array([[5.0], [5.0]])))
        actor.weapon.register(self.controller)
        actor.weapon.attack()
        actor.move(0.0)
        for i in range(50):
            self.controller.update()
            actor.bounce(obs)
            if obs.shape.is_inside(actor.weapon.front):
                print("Hit")
                actor.weapon.break_attack()
        self.assertFalse(obs.shape.is_inside(actor.weapon.front))


if __name__ == '__main__':
    unittest.main()
