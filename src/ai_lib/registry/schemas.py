from typing import Literal

from pydantic import BaseModel, Field

Provider = Literal["gemini", "groq"]
Purpose = Literal["chat", "vision", "embedding"]
Outcome = Literal["ok", "rate_limited", "invalid"]


class AuthHeader(BaseModel):
    name: str
    value: str = Field(repr=False)


class KeyLease(BaseModel):
    provider: Provider
    key_id: str
    api_key: str = Field(repr=False)
    base_url: str
    auth_header: AuthHeader
