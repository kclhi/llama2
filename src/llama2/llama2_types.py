from dataclasses import dataclass


@dataclass
class TokenisedResponse:
    tokens: list[str]


@dataclass
class Llama2ChatExchange:
    human: str
    assistant: str


@dataclass
class Llama2Parameters:
    prompt: str
    temperature: float
    top_k: int
    top_p: float
    n_keep: int
    n_predict: int
    stop: list[str]  # stop completion after generating this
    stream: bool
