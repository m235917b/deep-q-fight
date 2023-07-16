"""
This file contains some tests for the animation controller.
"""

import unittest

from game import animation


class TestAnimation(unittest.TestCase):
    def test_animation(self):
        actions = []

        a1 = animation.Animation(
            update=lambda: actions.append('anim1'),
            final=lambda: actions.append('end1'),
            duration=2
        )
        a2 = animation.Animation(
            lambda: actions.append('begin2'),
            lambda: actions.append('anim2'),
            lambda: actions.append('end2'), duration=0, next_animation=a1
        )
        a3 = animation.Animation(
            lambda: actions.append('begin3'),
            lambda: actions.append('anim3'),
            lambda: actions.append('end3'), next_animation=a2
        )

        while a3.running:
            a3.update()

        self.assertEqual(['begin3', 'anim3', 'end3', 'begin2', 'end2', 'anim1', 'anim1', 'end1'], actions)
        self.assertFalse(a3.running)
        a3.update()
        self.assertEqual(['begin3', 'anim3', 'end3', 'begin2', 'end2', 'anim1', 'anim1', 'end1'], actions)
        self.assertFalse(a3.running)

    def test_animation_condition(self):
        actions = []
        ctr = 0

        a1 = animation.Animation(
            update=lambda: actions.append('anim1'),
            final=lambda: actions.append('end1'),
            condition=lambda: ctr < 4
        )
        a2 = animation.Animation(
            lambda: actions.append('begin2'),
            lambda: actions.append('anim2'),
            lambda: actions.append('end2'), condition=lambda: ctr < 2, next_animation=a1
        )

        while a2.running:
            a2.update()
            ctr += 1

        self.assertEqual(['begin2', 'anim2', 'anim2', 'end2', 'anim1', 'anim1'], actions)
        self.assertFalse(a2.running)
        a2.update()
        self.assertEqual(['begin2', 'anim2', 'anim2', 'end2', 'anim1', 'anim1', 'end1'], actions)
        self.assertFalse(a2.running)


class TestAnimated(unittest.TestCase):
    class TestAnimated(animation.Animated):
        def __init__(self):
            super().__init__()
            self.final = True
            self.actions = []

        def finalize(self):
            self.final = True

        def animate(self):
            self.final = False
            a = animation.Animation(
                update=lambda: self.actions.append('anim2'),
                final=lambda: self.finalize(),
                duration=2
            )
            self.start_animation(animation.Animation(
                update=lambda: self.actions.append('anim1'),
                next_animation=a
            ))

    def test_animation_observer_pattern(self):
        controller = animation.AnimationController()
        test_animated = TestAnimated.TestAnimated()
        test_animated.register(controller)
        test_animated.animate()

        self.assertFalse(controller.empty())

        while not test_animated.final:
            controller.update()
        controller.update()

        self.assertTrue(controller.empty())

        controller.update()

        self.assertTrue(controller.empty())

        self.assertEqual(['anim1', 'anim2', 'anim2'], test_animated.actions)


if __name__ == '__main__':
    unittest.main()
