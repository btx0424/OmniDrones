from .playerpool import Player, PlayerPool

from multielo import MultiElo


class SelfPlay:
    def __init__(self) -> None:
        self.elo = MultiElo()
