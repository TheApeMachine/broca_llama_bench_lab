from typing import Protocol

class Frontend(Protocol):
    def run(self):
        pass