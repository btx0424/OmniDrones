from typing import List, Callable

import random

import torch


class Player:
    """Player
    Simple wrapper around a model
    """

    def __init__(self, model) -> None:
        self.n_win = 0
        self.n_lose = 0
        self.n_draw = 0
        self.model = model
        self.env_indices = torch.empty([], dtype=torch.long, requires_grad=False)

    @property
    def n_game(self) -> int:
        return self.n_win + self.n_lose + self.n_draw

    @property
    def win_rate(self) -> float:
        if self.n_game == 0:
            return 0.5
        else:
            return (self.n_win + 0.5 * self.n_draw) / self.n_game

    def update_stats(self, cnt_win: int, cnt_lose: int, cnt_draw: int):
        self.n_win += cnt_win
        self.n_lose += cnt_lose
        self.n_draw += cnt_draw

    def reset_stats(self):
        self.n_win = 0
        self.n_lose = 0
        self.n_draw = 0


class PlayerPool:
    def __init__(
        self,
        capacity: int,
        weight_func: Callable[
            [
                Player,
            ],
            float,
        ],
    ) -> None:
        assert capacity > 0
        self.capacity = capacity
        self.players: List[Player] = []
        self._player_idx = 0
        self.weight_func = weight_func

    def __len__(self) -> int:
        return len(self.players)

    def add_player(self, player: Player):
        if len(self.players) < self.capacity:
            self.players.append(player)
        else:
            self.players[self._player_idx] = player
        self._player_idx = (self._player_idx + 1) % self.capacity

    def sample_player(self) -> Player:
        weights = [self.weight_func(player) for player in self.players]
        return random.choices(self.players, weights=weights)[0]
