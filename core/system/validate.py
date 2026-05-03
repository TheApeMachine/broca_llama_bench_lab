class Validate:
    def __init__(self, name: str) -> None:
        self.name = name

    def validate(self, predicates: list[bool]) -> None:
        if not all(predicates):
            raise ValueError(f"{self.name} validation failed")