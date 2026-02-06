from abc import ABC, abstractmethod

class CommonLLM(ABC):

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def generate_batch(self, prompts: list[str]) -> list[str]:
       pass

    @abstractmethod
    def inference_batch(self, conversations: list[list[dict[str, str]]]) -> list[str]:
        pass