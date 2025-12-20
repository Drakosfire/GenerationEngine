"""Protocol interfaces for GenerationEngine."""

from typing import Protocol, TypeVar

from generationengine.models.responses import GenerationResponse

TInput = TypeVar("TInput", contravariant=True)
TOutput = TypeVar("TOutput", covariant=True)


class IGenerator(Protocol[TInput, TOutput]):
    """Common interface for all DungeonMind generators."""

    async def generate(self, input: TInput) -> GenerationResponse[TOutput]:
        """Generate content from input. Returns standardized response."""
        ...

    @property
    def generator_type(self) -> str:
        """Return generator type identifier (statblock, card, character, store)."""
        ...

