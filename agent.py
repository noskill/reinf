

class Agent():
    def __init__(self, *args, logger=None, state_extractor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.state_extractor = state_extractor
        self.hparams = dict()

    def log_hparams(self):
        self.logger.add_hparams(self.hparams, {})

    def episode_start(self):
        pass

    def get_action(self, state, done):
        pass

    def update(self, obs, actions, rewards, dones, next_obs):
        pass
