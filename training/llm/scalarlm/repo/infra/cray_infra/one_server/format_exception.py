def format_exception(exc: BaseException) -> str:
    repr_text = repr(exc)
    type_name = type(exc).__name__
    if repr_text and not repr_text.endswith("()"):
        return f"{type_name}: {repr_text}"
    return type_name
