import os
import yaml
import logging

logger = logging.getLogger(__name__)

def load_local_training_config():
    """Load local training config from ml directory."""
    config_path = os.path.join(
        os.path.dirname(__file__), 
        'local_training_config.yaml'
    )
    
    if os.path.exists(config_path):
        #logger.info(f"Loading local training config from: {config_path}")
        with open(config_path, 'r') as f:
            local_config = yaml.safe_load(f)
        #logger.info(f"Local config loaded: {local_config}")
        return local_config
    else:
        logger.warning(f"Local training config not found at: {config_path}")
        return {}