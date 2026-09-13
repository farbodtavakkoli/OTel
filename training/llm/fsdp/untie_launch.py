import os
import sys
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from transformers.models.auto.auto_factory import _BaseAutoModelClass
_orig = _BaseAutoModelClass.from_pretrained.__func__
def _patched(cls, pretrained_model_name_or_path, *args, **kwargs):
    kwargs.setdefault("tie_word_embeddings", False)
    return _orig(cls, pretrained_model_name_or_path, *args, **kwargs)
_BaseAutoModelClass.from_pretrained = classmethod(_patched)

# hand argv to the shipped script and run it as __main__
sys.argv[0] = REPO + "/train_llm_fsdp.py"
import runpy
runpy.run_path(REPO + "/train_llm_fsdp.py", run_name="__main__")
