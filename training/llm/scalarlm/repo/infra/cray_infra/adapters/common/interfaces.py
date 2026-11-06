from typing import ClassVar, Literal, Protocol, Type, Union, overload, runtime_checkable
from typing_extensions import TypeIs

import torch


@runtime_checkable
class SupportsTokenformer(Protocol):
    supports_tokenformer: ClassVar[Literal[True]] = True

    def enable_tokenformer(self, config: dict) -> None:
        ...

    def disable_tokenformer(self) -> None:
        ...


@runtime_checkable
class _SupportsTokenformerType(Protocol):
    supports_tokenformer: Literal[True]


@overload
def supports_tokenformer(model: Type[object]) -> TypeIs[Type[SupportsTokenformer]]:
    ...


@overload 
def supports_tokenformer(model: object) -> TypeIs[SupportsTokenformer]:
    ...


def supports_tokenformer(
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[SupportsTokenformer]], TypeIs[SupportsTokenformer]]:
    if isinstance(model, type):
        return isinstance(model(), _SupportsTokenformerType)
    else:
        return isinstance(model, _SupportsTokenformerType)