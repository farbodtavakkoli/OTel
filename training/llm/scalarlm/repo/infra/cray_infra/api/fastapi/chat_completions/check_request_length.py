from typing import Optional


class RequestTooLongError(Exception):
    def __init__(
        self,
        *,
        prompt_tokens: int,
        max_tokens: int,
        max_model_length: int,
    ):
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.max_model_length = max_model_length
        self.total = prompt_tokens + max_tokens
        super().__init__(self._message())

    def _message(self) -> str:
        return (
            f"Request too long: prompt_tokens={self.prompt_tokens} + "
            f"max_tokens={self.max_tokens} = {self.total} > "
            f"max_model_length={self.max_model_length}. "
            f"Reduce the prompt or max_tokens."
        )


def check_request_length(
    *,
    prompt_tokens: int,
    max_tokens: Optional[int],
    max_model_length: int,
) -> None:
    if max_model_length <= 0:
        return
    requested_max = max_tokens or 0
    if prompt_tokens + requested_max > max_model_length:
        raise RequestTooLongError(
            prompt_tokens=prompt_tokens,
            max_tokens=requested_max,
            max_model_length=max_model_length,
        )
