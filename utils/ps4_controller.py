from __future__ import annotations

import logging
import platform
import pygame
import queue
import time
import threading
import typing as T

import numpy as np

from collections import defaultdict

class PS4Controller:
    # buttons configuration on Windows
    X = 0
    O = 1
    SQUARE = 2 if platform.system() == "Windows" else 3
    TRIANGLE = 3 if platform.system() == "Windows" else 2

    LEFT_UD = 1
    LEFT_LR = 0
    RIGHT_LR = 2 if platform.system() == "Windows" else 3
    RIGHT_UD = 3 if platform.system() == "Windows" else 4

    def __init__(self, joystick_id: int = 0) -> None:
        self.joystick_id = joystick_id
        self.terminate_evt = threading.Event()
        self.axes = defaultdict(float)
        self.buttons = defaultdict(bool)
        self.button_callbacks = {}
        self.step = 0

    def __enter__(self) -> PS4Controller:
        self.terminate_evt.clear()
        self.event_t = threading.Thread(target=self._event_loop)
        self.event_t.start()

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.terminate_evt.set()
        self.event_t.join()

    def register_button_callback(self, button_id: int, func: T.Callable[[], None]) -> None:
        self.button_callbacks[button_id] = func

    def _event_loop(self) -> None:
        pygame.init()
        joy = pygame.joystick.Joystick(self.joystick_id)
        joy.init()

        while not self.terminate_evt.is_set():
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.JOYAXISMOTION:
                    self.axes[event.axis] = event.value
                elif event.type == pygame.JOYBUTTONDOWN:
                    self.buttons[event.button] = True
                    if event.button in self.button_callbacks:
                        self.button_callbacks[event.button]()
                elif event.type == pygame.JOYBUTTONUP:
                    self.buttons[event.button] = False

        joy.quit()
        logging.info("Joystick quitted gracefully")


if __name__ == "__main__":
    # press "X" to stop logging
    ctrl = PS4Controller()
    with ctrl:
        while not ctrl.buttons[0]:
            print(f"Axis -- 0: {ctrl.axes[0]:.3f} "
                  f"1: {ctrl.axes[1]:.3f} "
                  f"2: {ctrl.axes[2]:.3f} "
                  f"3: {ctrl.axes[3]:.3f} "
                  f"4: {ctrl.axes[4]:.4f} "
                  f"5: {ctrl.axes[5]:.5f} ")
            print(f"Buttons -- 0: {ctrl.buttons[0]} "
                  f"1: {ctrl.buttons[1]} "
                  f"2: {ctrl.buttons[2]} "
                  f"3: {ctrl.buttons[3]} ")
