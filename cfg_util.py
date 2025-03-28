import os
import re
from urllib.parse import urlparse

def replace_remote_with_local(url_path):
    """Replace any remote URL with a local path using ISAAC_ASSET_ROOT."""
    if isinstance(url_path, str) and url_path.startswith(('http://', 'https://', 's3://')):
        parsed_url = urlparse(url_path)
        path_part = parsed_url.path
        if path_part.startswith('/'):
            path_part = path_part[1:]
        
        isaac_asset_root = os.environ.get("ISAAC_ASSET_ROOT", "")
        return os.path.join(isaac_asset_root, path_part)
    return url_path


def replace_urls_in_config(config, visited=None):
    """
    Recursively replace all URL strings in a configuration object with local paths.
    Uses a visited set to prevent infinite recursion.
    """
    if visited is None:
        visited = set()
    
    # Skip if we've already visited this object or it's a primitive type
    obj_id = id(config)
    if obj_id in visited or not hasattr(config, '__dict__'):
        return config
    
    visited.add(obj_id)
    
    # Check specific fields that might contain USD paths
    if hasattr(config, 'usd_path'):
        config.usd_path = replace_remote_with_local(config.usd_path)
    
    if hasattr(config, 'spawn') and hasattr(config.spawn, 'usd_path'):
        config.spawn.usd_path = replace_remote_with_local(config.spawn.usd_path)
    
    # Process other attributes, but only certain types to avoid recursion issues
    for attr_name, attr_value in vars(config).items():
        # Skip special attributes and already processed ones
        if attr_name.startswith('__') or attr_name in ('usd_path', 'spawn'):
            continue
        
        # Process nested configuration objects
        if hasattr(attr_value, '__dict__'):
            replace_urls_in_config(attr_value, visited)
    
    return config
