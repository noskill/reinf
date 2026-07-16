import torch


class CacheModuleMixin:

    def get_cache_state(self):
        raise NotImplementedError

    def set_cache_state(self, state):
        raise NotImplementedError

    def index_cache_state(self, state, batch_indices: torch.Tensor):
        raise NotImplementedError
                                                 
    def reset_cache(self, reset_mask: torch.Tensor):
        pass

    def clear_cache(self):
        pass


def clear_cache(module):                                                                                                                                                   
    for child in module.children():                                                                                                                                                                             
        if isinstance(child, CacheModuleMixin):                                                                                                                                                                 
            child.clear_cache()


def reset_cache(module, reset_mask: torch.Tensor):                                                                                                                                                   
    for child in module.children():                                                                                                                                                                             
        if isinstance(child, CacheModuleMixin):                                                                                                                                                                 
            child.reset_cache(reset_mask) 
