"""
Task 1
In this task bot gets a reward if it finds a pass and opens the door.
"""
# Imports
import pygame
import sys
import gym
import numpy as np
from gym import spaces
from pygame.surfarray import array3d
from causalgym.envs.task_objs import Bot, Box, Door


class Task1Env(gym.Env):
    """
    Gym Environment for Task1
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Task1Env, self).__init__()
        # Initialize pygame
        pygame.init()
        # Constants
        self.DISPX, self.DISPY = 256, 256
        self.FPS = 30
        _NUM_ACTIONS = 5
        self.KEY_MAP ={0: pygame.K_UP,
                       1: pygame.K_DOWN,
                       2: pygame.K_LEFT,
                       3: pygame.K_RIGHT,
                       4: pygame.K_SPACE}
        # Setup Observation and Action Spaces
        self.action_space = spaces.Discrete(_NUM_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=max(self.DISPX, self.DISPY),
                                            shape=(self.DISPX, self.DISPY), dtype=np.uint8)
        # Setup Task
        self.setup_task()

    def setup_task(self, for_play=False):
        # Instance a display
        if for_play:
            self.screen = pygame.display.set_mode((self.DISPX, self.DISPY))
        else:
            self.screen = pygame.Surface((self.DISPX, self.DISPY))
            self.render_scr = pygame.display.set_mode((self.DISPX, self.DISPY))
        # Instance a clock
        self.clock = pygame.time.Clock()
        # Set white as background
        self.screen.fill((255, 255, 255))
        # Instance Bot, Box and Door sprites
        # 1. Bot
        self.bot = Bot(30, 30, self.DISPX, self.DISPY)
        # 2. Boxes
        self.boxes = [Box(100, 50, box_color='red'),
                      Box(100, 120, box_color='blue'),
                      Box(100, 200, box_color='green')]
        # Harcoding Red box to have pass.
        self.boxes[0].has_pass = True
        # 3. Doors
        self.doors = [Door(200, 50, door_color='orange'),
                      Door(200, 200, door_color='green')]
        self.done = False
        self.reward = 0
        # Draw Sprites
        self._draw_sprites()

    def _draw_sprites(self):
        self.bot.draw(self.screen)
        for box in self.boxes:
            box.draw(self.screen)
        for door in self.doors:
            door.draw(self.screen)

    def reset(self):
        self.setup_task()
        obs_img = array3d(self.screen)
        obs_img = np.transpose(obs_img, (1, 0, 2))
        done = False
        reward = 0
        return obs_img, done, reward

    def step(self, action):
        done = False
        reward = 0
        # Set white as background
        self.screen.fill((255, 255, 255))
        # Update State based on action
        # 1. Bot Update
        self.bot.update(self.KEY_MAP[action])
        # 2. Box Update
        for box in self.boxes:
            box.update(self.bot, self.KEY_MAP[action])
        # 3. Door Update
        for door in self.doors:
            door_retval = door.update(self.bot, self.KEY_MAP[action])
            if door_retval is not None:
                done = True
                reward = door_retval
                break
        # Draw Sprites
        self._draw_sprites()
        # Get image of observation
        obs_img = array3d(self.screen)
        obs_img = np.transpose(obs_img, (1, 0, 2))
        # Tick clock
        self.clock.tick(self.FPS)
        return obs_img, done, reward

    def close(self):
        sys.exit(0)

    def render(self, mode='human', close=False):
        self.render_scr.blit(self.screen, (0,0))
        pygame.display.update()

    def play(self):
        self.setup_task(for_play=True)
        # Increment bot speed to make bot move faster
        self.bot.speed = 2
        pygame.display.update()
        while True:
            # Handle Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
            # Fill Screen
            self.screen.fill((255, 255, 255))
            # Find which key is pressed
            pressed_keys = pygame.key.get_pressed()
            key_hit = pygame.K_0
            if pressed_keys[pygame.K_UP]:
                key_hit = pygame.K_UP
            elif pressed_keys[pygame.K_DOWN]:
                key_hit = pygame.K_DOWN
            elif pressed_keys[pygame.K_LEFT]:
                key_hit = pygame.K_LEFT
            elif pressed_keys[pygame.K_RIGHT]:
                key_hit = pygame.K_RIGHT
            elif pressed_keys[pygame.K_SPACE]:
                key_hit = pygame.K_SPACE
            else:
                pass
            # Update Sprite state
            self.bot.update(key_hit)
            for box in self.boxes:
                box.update(self.bot, key_hit)
            for door in self.doors:
                door_retval = door.update(self.bot, key_hit)
                if door_retval is not None:
                    if door_retval:
                        print("Found pass and got reward")
                        sys.exit(0)
                    else:
                        print("Did not get any reward")
                        sys.exit(0)
             # Draw task elements
            self.bot.draw(self.screen)

            for box in self.boxes:
                box.draw(self.screen)

            for door in self.doors:
                door.draw(self.screen)

            pygame.display.update()
            self.clock.tick(self.FPS)
