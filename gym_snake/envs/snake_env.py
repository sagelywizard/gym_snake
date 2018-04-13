import time
import collections
import operator
import random
import unittest
from enum import Enum

import numpy as np

import gym
from gym import spaces
from gym.envs.classic_control import rendering


class Dir(Enum):
    NORTH = (0, 1)
    EAST = (1, 0)
    SOUTH = (0, -1)
    WEST = (-1, 0)


class Event(Enum):
    DEAD = 0
    EAT = 1
    NEW_FOOD = 2
    ADD = 3
    REMOVE = 4
    WIN = 5


class SnakeEnv(gym.Env):
    """An environment for snake

    Arguments:
        game (SnakeGame, optional): If not supplied, a SnakeGame with some
            sane defaults is used.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 40
    }
    def __init__(self, game=None):
        self.game = game
        if self.game is None:
            self.game = SnakeGame()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(self.game.height, self.game.width, 1),
            dtype=np.uint8
        )
        self.viewer = None

    def step(self, action):
        """Steps the snake game.

        Arguments:
            action (int, 0-3 inclusive): whether to move the snake north,
                east, south, or west (respectively).
        """
        if self.game.game_over:
            return (None, 0, True, {})
        assert self.action_space.contains(action), "invalid action"
        if action == 0:
            direction = Dir.NORTH
        elif action == 1:
            direction = Dir.EAST
        elif action == 2:
            direction = Dir.SOUTH
        elif action == 3:
            direction = Dir.WEST
        events = self.game.step(direction)
        reward = 0
        for event in events:
            if self.viewer is not None:
                self.viewer.process_event(event)
            if event == Event.DEAD:
                reward = -1
            elif isinstance(event, tuple) and event[0] == Event.EAT:
                reward = 1
        return (self.game.board, reward, self.game.game_over, {})

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = SnakeViewer(self.game)
        return self.viewer.render(mode)

    def reset(self):
        self.game.reset()
        if self.viewer is not None:
            self.viewer.reset()
        return self.game.board

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class SnakeViewer(object):
    """A renderer for the snake game.
    Arguments:
        game (SnakeGame): the game to be rendered
        screen_width (int, default 600): the width of the display in pixels
        screen_height (int, default 600): the height of the display in pixels
        food_color (3-int tuple, default (0.4, 0.6, 0.8): the color of food
            squares.
        snake_color (3-int tuple, default (0.3, 0.3, 0.3): the color of snake
            squares.
    """
    def __init__(self, game, screen_width=600, screen_height=600,
            food_color=(0.4, 0.6, 0.8), snake_color=(0.3, 0.3, 0.3)):
        self.game = game
        self.food_color = food_color
        self.snake_color = snake_color
        self.square_width = screen_width / self.game.width
        self.square_height = screen_height / self.game.height
        self.viewer = rendering.Viewer(screen_width, screen_height)
        self.food_color = food_color
        self.snake_color = snake_color
        self.reset()

    def reset(self):
        self.viewer.geoms = []
        self.snake_geoms = {}
        _, food_square = self.game.get_state()
        self.food_geom = self.get_square(food_square, self.food_color)
        self.viewer.add_geom(self.food_geom)
        for snake_square in self.game.get_snake_squares():
            snake_geom = self.get_square(snake_square, self.snake_color)
            self.snake_geoms[snake_square] = snake_geom
            self.viewer.add_geom(snake_geom)

    def get_square(self, square, color):
        square_x, square_y = square
        geom = rendering.FilledPolygon([
            (square_x*self.square_width, square_y*self.square_height),
            (square_x*self.square_width, (square_y+1)*self.square_height),
            ((square_x+1)*self.square_width, (square_y+1)*self.square_height),
            ((square_x+1)*self.square_width, square_y*self.square_height)

        ])
        geom.set_color(*color)
        return geom

    def process_event(self, event):
        if isinstance(event, tuple):
            if event[0] == Event.EAT:
                self.viewer.geoms.remove(self.food_geom)
                self.food_geom = None
            elif event[0] == Event.NEW_FOOD:
                self.food_geom = self.get_square(event[1], self.food_color)
                self.viewer.add_geom(self.food_geom)
            elif event[0] == Event.ADD:
                square = self.get_square(event[1], self.snake_color)
                self.snake_geoms[event[1]] = square
                self.viewer.add_geom(square)
            elif event[0] == Event.REMOVE:
                square = self.snake_geoms[event[1]]
                del self.snake_geoms[event[1]]
                self.viewer.geoms.remove(square)

    def close(self):
        self.viewer.close()

    def render(self, mode='human'):
        return self.viewer.render(return_rgb_array = mode=='rgb_array')


class SnakeGame(object):
    """Class to capture state of snake game, as well as perform state
    transitions.

    Arguments:
        width (int, default 20): number of squares wide the board should be
        height (int, default 20) number of squares tall the board should be
    """
    def __init__(self, width=20, height=20):
        assert width > 1 and height > 1
        self.width = width
        self.height = height
        self.reset()

    def reset(self, head=None):
        self.game_over = False
        self.current_dir = Dir.NORTH
        self.board = np.zeros((self.width, self.height), dtype=np.uint8)
        if head is None:
            head = (
                random.randint(0, self.width-1),
                random.randint(0, self.height-1)
            )
        self.board[head] = 1
        self.snake_squares = collections.deque([head])
        self.food_square = tuple(random.choice(np.argwhere(self.board == 0)))
        self.board[self.food_square] = 2

    def get_state(self):
        head = self.snake_squares.pop()
        self.snake_squares.append(head)
        return (head, self.food_square)

    def get_snake_squares(self):
        return map(tuple, np.argwhere(self.board == 1))

    def step(self, direction):
        if self.game_over:
            return []
        events = []
        # Check if going directly backward. If so, just step forward.
        old_head = self.snake_squares.pop()
        second_oldest_head = None
        if len(self.snake_squares) > 0:
            second_oldest_head = self.snake_squares.pop()
            self.snake_squares.append(second_oldest_head)
        self.snake_squares.append(old_head)
        new_head = tuple(map(operator.add, old_head, direction.value))
        if second_oldest_head == new_head:
            direction = self.current_dir
            new_head = tuple(map(operator.add, old_head, direction.value))
        self.current_dir = direction

        if not (0 <= new_head[0] < self.width) or not (0 <= new_head[1] < self.height):
            self.game_over = True
            return [Event.DEAD]
        elif new_head == self.food_square:
            try:
                new_food_square = tuple(random.choice(np.argwhere(self.board == 0)))
            except IndexError:
                self.game_over = True
                return [(Event.EAT, self.food_square), Event.WIN]

            events.append((Event.EAT, self.food_square))
            events.append((Event.NEW_FOOD, new_food_square))
            self.food_square = new_food_square
            self.board[self.food_square] = 2
        else:
            tail = self.snake_squares.popleft()
            self.board[tail] = 0
            events.append((Event.REMOVE, tail))
        # Check if the new head is a valid spot AFTER making changes from
        # eating food.
        if self.board[new_head] == 1:
            self.game_over = True
            return [Event.DEAD]
        self.board[new_head] = 1
        tail = self.snake_squares.append(new_head)
        events.append((Event.ADD, new_head))
        return events


class SnakeGameTest(unittest.TestCase):
    def setUp(self):
        self.game = SnakeGame(10, 10)

    def test_walk_off_edge(self):
        self.game.reset(head=(4, 0))

        # manually make sure food is out of the way
        self.game.board[self.game.food_square] = 0
        self.game.food_square = (8,8)
        self.game.board[self.game.food_square] = 2

        move_west = self.game.step(Dir.WEST)
        self.assertTrue((Event.REMOVE, (4, 0)) in move_west)
        self.assertTrue((Event.ADD, (3, 0)) in move_west)
        move_east = self.game.step(Dir.EAST)
        self.assertTrue((Event.REMOVE, (3, 0)) in move_east)
        self.assertTrue((Event.ADD, (4, 0)) in move_east)

        self.assertEqual(self.game.step(Dir.SOUTH), [Event.DEAD])

    def test_eat_food(self):
        self.game.reset(head=(8, 0))

        self.game.board[self.game.food_square] = 0
        self.game.food_square = (7,0)
        self.game.board[self.game.food_square] = 2

        move_west = self.game.step(Dir.WEST)

        self.assertTrue((Event.EAT, (7, 0)) in move_west)
        self.assertTrue((Event.ADD, (7, 0)) in move_west)
        events = set(list(zip(*move_west))[0])
        self.assertEqual(events, {Event.EAT, Event.ADD, Event.NEW_FOOD})
        self.assertEqual(len(move_west), 3)

    def test_eat_self(self):
        self.game.reset(head=(8, 0))

        self.game.board[self.game.food_square] = 0
        self.game.food_square = (7,0)
        self.game.board[self.game.food_square] = 2

        self.game.step(Dir.WEST)

        self.game.board[self.game.food_square] = 0
        self.game.food_square = (6,0)
        self.game.board[self.game.food_square] = 2

        self.game.step(Dir.WEST)

        self.game.board[self.game.food_square] = 0
        self.game.food_square = (6,1)
        self.game.board[self.game.food_square] = 2

        self.game.step(Dir.NORTH)

        self.game.board[self.game.food_square] = 0
        self.game.food_square = (7,1)
        self.game.board[self.game.food_square] = 2

        self.game.step(Dir.EAST)

        self.assertEqual([Event.DEAD], self.game.step(Dir.SOUTH))
