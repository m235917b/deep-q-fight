"""
This file implements the animation controller.

The animation controller is used to manage and run all animations
of the game, that last for several frames.
"""


class Animation:
    def __init__(self, begin=lambda: None, update=lambda: None, final=lambda: None, duration=1, condition=None,
                 next_animation=None):
        if duration < 0:
            raise ValueError("duration must be positive")

        self.__begin = begin
        self.__update = update
        self.__final = final
        self.__ctr = 0
        self.__duration = duration
        self.__condition = condition
        self.__next = next_animation

    def __update_ctr(self):
        if self.__ctr == 0:
            self.__begin()

        if self.__ctr < self.__duration:
            self.__update()

        if self.__ctr == self.__duration:
            self.__final()

        self.__ctr += 1

        if self.__ctr > self.__duration:
            if self.__next is not None:
                self.__next.update()

    def __update_condition(self):
        if self.__ctr == 0 and self.__condition():
            self.__ctr += 1
            self.__begin()

        if self.__condition():
            self.__update()

        if self.__ctr == 1 and not self.__condition():
            self.__ctr += 1
            self.__final()

        if self.__ctr > 1 and not self.__condition():
            self.__condition = lambda: False
            if self.__next is not None:
                self.__next.update()

    def update(self):
        if self.__condition is None:
            self.__update_ctr()
        else:
            self.__update_condition()

    @property
    def running(self):
        self_running = self.__ctr <= self.__duration if self.__condition is None else self.__condition()

        if self.__next is None:
            return self_running
        else:
            return self_running or self.__next.running


class AnimationController:
    def __init__(self):
        self.__animations = []

    def start(self, animation):
        self.__animations.append(animation)

    def kill(self, animation):
        if self.__animations.__contains__(animation):
            self.__animations.remove(animation)

    def update(self):
        self.__animations = [animation for animation in self.__animations if animation.running]

        for animation in self.__animations:
            animation.update()

    def empty(self):
        """only for tests"""
        return not self.__animations


class Animated:
    def __init__(self):
        self.__controller = None

    def register(self, controller: AnimationController):
        self.__controller = controller

    def unregister(self):
        self.__controller = None

    def start_animation(self, animation: Animation):
        self.__controller.start(animation)

    def kill_animation(self, animation: Animation):
        self.__controller.kill(animation)
