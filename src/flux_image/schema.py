from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: Annotated[str, Field(min_length=1, max_length=16000)]
    width: Annotated[int, Field(ge=256, le=2048, multiple_of=16)] = 768
    height: Annotated[int, Field(ge=256, le=2048, multiple_of=16)] = 768
    steps: Annotated[int, Field(ge=4, le=16)] = 16
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] | None = None
    reference_images: list[Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]] = Field(default_factory=list, max_length=4)
    client_request_id: Annotated[str, Field(min_length=1, max_length=128)] | None = None

    @model_validator(mode="after")
    def pixel_budget(self):
        if self.width * self.height > 1024 * 1024:
            raise ValueError("Output is limited to 1 megapixel for the 12 GB GPU profile.")
        if not self.prompt.strip():
            raise ValueError("Prompt must contain non-whitespace characters.")
        return self
