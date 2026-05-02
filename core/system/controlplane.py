from .frontend import Frontend


class ControlPlane:
    def __init__(self, frontend: Frontend):
        self.frontend = frontend

    def run(self) -> None:
        self.frontend.run()
