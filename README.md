# Snake for OpenAI Gym

A version of [snake](https://en.wikipedia.org/wiki/Snake_(video_game_genre)) for [OpenAI Gym](https://github.com/openai/gym).

## Installation

```
git clone git@github.com:sagelywizard/gym_snake.git
cd gym_snake
pip install -e .
```

## Usage

```
import gym_snake
import gym

env = gym.make('snake-v0')
state = env.reset()
done = False
while not done:
    action = policy(state)
    state, reward, done, info = env.step(action)
    env.render()
```
