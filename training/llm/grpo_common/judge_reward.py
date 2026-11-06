import sys

try:
    from . import grpo_judge_local
except ImportError:
    import grpo_judge_local

if __name__ == "__main__":
    import runpy

    runpy.run_path(grpo_judge_local.__file__, run_name="__main__")
else:
    sys.modules[__name__] = grpo_judge_local
