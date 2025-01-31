

class Agent():
    def __init__(self, logger):
        self.logger = logger

    def episode_start(self):
        pass

    def get_action(self, state, done):
        pass

    def update(self, obs, actions, rewards, dones, next_obs):
        pass
