from typing import Callable


class ApprovalGate:
    def __init__(self, prompt: str):
        self.prompt = prompt

    def request_approval(self, confirmation_callback: Callable[[], bool]) -> bool:
        return confirmation_callback()
