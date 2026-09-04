"""
Render an exception into a non-empty diagnostic string.

Lives in its own module so it can be unit-tested without paying for
the create_generate_worker import chain (vllm, aiohttp, transformers).

`str(exc)` blanks out for bare-args exceptions: `AssertionError()`,
`KeyError("")`, vLLM's request_output_to_completion_response asserts,
`RuntimeError()` from cancelled tasks, etc. The empty string then
propagated through the worker → finish_work → chat completions
wrapper, surfacing as a 200 OK with empty `content` — looking like
silent success for what was actually a hard failure.
"""


def format_exception(exc: BaseException) -> str:
    """
    Carry the exception class even when `__str__` is empty. Output is
    one of three shapes:

      "<Type>: <repr>"   when repr has args
      "<Type>"           when repr is e.g. `AssertionError()` (empty)
    """
    repr_text = repr(exc)
    type_name = type(exc).__name__
    if repr_text and not repr_text.endswith("()"):
        return f"{type_name}: {repr_text}"
    return type_name
