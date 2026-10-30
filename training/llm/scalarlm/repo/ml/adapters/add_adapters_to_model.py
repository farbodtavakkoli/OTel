from cray_infra.util.get_job_config import get_job_config

from adapters.create_tokenformer_model import create_tokenformer_model
from adapters.create_lora_model import create_lora_model

def add_adapters_to_model(model, device):
    job_config = get_job_config()

    if job_config["adapter_type"] == "tokenformer":
        return create_tokenformer_model(model=model, device=device)
    elif job_config["adapter_type"] == "lora":
        return create_lora_model(model=model, device=device)
    elif job_config["adapter_type"] == "none":
        return model
    else:
        raise ValueError(f"Unsupported adapter type: {job_config['adapter_type']}")
