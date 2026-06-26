import torch


class CacheModuleMixin:
                                                
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
