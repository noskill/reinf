try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    class SummaryWriter:
        
        def __init__(self, log_dir="logs"):
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            # create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # add formatter to ch
            ch.setFormatter(formatter)

            # add ch to logger
            self.logger.addHandler(ch)
            ch = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
        def add_scalar(self, *args):
            logging.info(str(args))
        
        def log_scalar(self, name, value, step=None):
            logging.info("iter " + str(step) + " " + str(name) + " " + str(value))
    
        def add_hparams(self, *args, **kwargs):
            logging.info(f"add_hparams({args}, {kwargs})")


class Logger:
    def __init__(self, log_dir="logs"):
        self.writer = SummaryWriter(log_dir)
        self.episode_count = 0

    def log_scalar(self, name, value, step=None):
        if step is None:
            step = self.episode_count
        self.writer.add_scalar(name, value, step)
    
    def add_hparams(self, *args, **kwargs):
        self.writer.add_hparams(*args, **kwargs)

    def increment_episode(self):
        self.episode_count += 1


