

class _PrefixedLogger:
    def __init__(self, logger, prefix):
        self.logger = logger
        self.prefix = str(prefix).strip("/")

    def log_scalar(self, name, value, step=None):
        name = str(name)
        if self.prefix and not name.startswith(self.prefix + "/"):
            name = f"{self.prefix}/{name}"
        return self.logger.log_scalar(name, value, step=step)

    def __getattr__(self, name):
        return getattr(self.logger, name)


class Agent():
    def __init__(self, *args, logger=None, state_extractor=None, log_prefix=None, **kwargs):
        super().__init__()
        self.log_prefix = None if log_prefix is None else str(log_prefix).strip("/")
        self.logger = _PrefixedLogger(logger, self.log_prefix) if logger is not None and self.log_prefix else logger
        self.state_extractor = state_extractor
        self.hparams = dict()

    def log_hparams(self):
        self.logger.add_hparams(self.hparams, {})

    def episode_start(self):
        pass

    def get_action(self, state, episode_start):
        pass

    def update(self, rewards, dones, info=None, **kwargs):
        pass
