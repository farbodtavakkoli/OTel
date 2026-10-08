from __future__ import annotations

from dataclasses import dataclass

from renderers import Message

from torchtitan.experiments.rl.environment import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)

from titanrl_telecom.data import TelecomSample


class TelecomEnv(MessageEnv):
    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        pass

    def __init__(self, config: Config, *, env_input: TelecomSample) -> None:
        del config
        self._messages = [dict(m) for m in env_input.prompt]

    async def init(self) -> MessageEnvInitOutput:
        return MessageEnvInitOutput(init_prompt_messages=list(self._messages))

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        del completion_message
        return MessageEnvStepOutput(done=True)
