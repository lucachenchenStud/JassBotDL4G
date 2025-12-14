# jass_parser.py

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Iterable

# Our global card ordering (same as for trump model)
CARDS = [
    # Diamonds
    'DA','DK','DQ','DJ','D10','D9','D8','D7','D6',
    # Hearts
    'HA','HK','HQ','HJ','H10','H9','H8','H7','H6',
    # Spades
    'SA','SK','SQ','SJ','S10','S9','S8','S7','S6',
    # Clubs
    'CA','CK','CQ','CJ','C10','C9','C8','C7','C6'
]
CARD_TO_IDX = {c: i for i, c in enumerate(CARDS)}


@dataclass
class Trick:
    cards: List[str]       # length 4
    points: int
    win: int               # 0..3
    first: int             # 0..3


@dataclass
class GameRecord:
    trump: int             # int code, same as in csv
    dealer: int            # 0..3
    forehand: int          # 0 or 1
    tricks: List[Trick]    # 9 tricks
    player_ids: List[int]  # length 4, can contain 0 if anonymous
    date: str              # e.g. "13.10.17 22:31:05"


@dataclass
class MoveRecord:
    game_index: int        # index of the game in the file
    trick_index: int       # 0..8
    play_index: int        # 0..3 within the trick
    player: int            # 0..3
    card: str              # e.g. "HA"
    card_idx: int          # 0..35


@dataclass
class GameTrajectory:
    game: GameRecord
    moves: List[MoveRecord]
    hands: List[List[str]]     # 4 x 9 cards
    team_points: Tuple[int,int]  # (team0, team1)


def load_games(path: str) -> List[GameRecord]:
    """Load all games from a json-lines jass_game file."""
    games: List[GameRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            g = obj["game"]
            tricks = [
                Trick(
                    cards=t["cards"],
                    points=t["points"],
                    win=t["win"],
                    first=t["first"],
                )
                for t in g["tricks"]
            ]
            games.append(
                GameRecord(
                    trump=g["trump"],
                    dealer=g["dealer"],
                    forehand=g["forehand"],
                    tricks=tricks,
                    player_ids=obj.get("player_ids", [0, 0, 0, 0]),
                    date=obj.get("date", ""),
                )
            )
    return games


def build_trajectory(game: GameRecord, game_index: int = 0) -> GameTrajectory:
    """
    From a GameRecord, reconstruct:
      - all MoveRecords (36 moves)
      - hands[p] = 9 cards per player
      - team points (team0: players 0+2, team1: 1+3)
    """
    moves: List[MoveRecord] = []
    hands: List[List[str]] = [[] for _ in range(4)]
    team_points = [0, 0]  # [team0, team1]

    for trick_idx, trick in enumerate(game.tricks):
        first = trick.first
        # scoring
        winner = trick.win
        if winner in (0, 2):
            team_points[0] += trick.points
        else:
            team_points[1] += trick.points

        for play_idx, card in enumerate(trick.cards):
            player = (first + play_idx) % 4
            hands[player].append(card)
            moves.append(
                MoveRecord(
                    game_index=game_index,
                    trick_index=trick_idx,
                    play_index=play_idx,
                    player=player,
                    card=card,
                    card_idx=CARD_TO_IDX[card],
                )
            )

    # Sanity checks
    assert all(len(h) == 9 for h in hands), f"Not 9 cards per hand: {hands}"
    assert len(moves) == 36, f"Expected 36 moves, got {len(moves)}"

    return GameTrajectory(
        game=game,
        moves=moves,
        hands=hands,
        team_points=(team_points[0], team_points[1]),
    )


def load_trajectories(path: str) -> List[GameTrajectory]:
    games = load_games(path)
    trajectories = [
        build_trajectory(g, i)
        for i, g in enumerate(games)
    ]
    return trajectories
