from cray_megatron.models.model_manager_base import ModelManagerBase

from cray_megatron.models.load_model import load_model


class TokenformerModelManager(ModelManagerBase):
    def load_model(self):
        return load_model()
