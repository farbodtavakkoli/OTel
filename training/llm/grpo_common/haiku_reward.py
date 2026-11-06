import sys

try:
    from . import grpo_judge_api
except ImportError:
    import grpo_judge_api

sys.modules[__name__] = grpo_judge_api
