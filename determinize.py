import numpy as np
from jass.game.game_state import GameState

def determinize_state_from_obs(obs, rng=None):
    """
    Build a full GameState consistent with obs by randomly distributing
    unknown cards to the other 3 players.
    """
    if rng is None:
        rng = np.random.default_rng()

    state = GameState()

    # Copy scalar info
    state.dealer = int(obs.dealer)
    state.player = int(obs.player)
    state.trump = int(obs.trump)
    state.forehand = int(obs.forehand)

    # Copy trick info (public)
    state.tricks[:] = obs.tricks
    state.trick_winner[:] = obs.trick_winner
    state.trick_points[:] = obs.trick_points
    state.trick_first_player[:] = obs.trick_first_player

    state.nr_tricks = int(obs.nr_tricks)
    state.nr_cards_in_trick = int(obs.nr_cards_in_trick)
    state.nr_played_cards = int(obs.nr_played_cards)

    # Current trick pointer
    if state.nr_played_cards != 36:
        state.current_trick = state.tricks[state.nr_tricks]
    else:
        state.current_trick = None

    # Points (can be recomputed too, but copying is fine if present)
    state.points[:] = obs.points

    # Hands: we know only obs.player_view hand
    state.hands[:] = 0
    pv = int(obs.player_view)
    assert pv in [0, 1, 2, 3], "obs.player_view must be 0..3 for determinization"
    state.hands[pv, :] = obs.hand

    # Determine which cards are already used (played in tricks or in known hand)
    played_cards = state.tricks[state.tricks != -1].astype(int)
    used = set(played_cards.tolist())
    used.update(np.flatnonzero(state.hands[pv]).tolist())

    # Remaining cards to distribute
    all_cards = set(range(36))
    remaining = np.array(sorted(list(all_cards - used)), dtype=int)
    rng.shuffle(remaining)

    # How many cards should each player have?
    # Each player starts with 9 cards, minus how many they already played.
    played_by_player = np.zeros(4, dtype=int)
    for t in range(9):
        first = state.trick_first_player[t]
        if first == -1:
            break
        for k in range(4):
            c = state.tricks[t, k]
            if c == -1:
                break
            pl = (first + k) % 4
            played_by_player[pl] += 1

    target_hand_sizes = 9 - played_by_player

    # We already set pv hand; verify its size matches target
    pv_hand_size = int(state.hands[pv].sum())
    # Sometimes observation hand size is consistent; if not, don't crash hard:
    # just trust obs.hand and adjust target
    target_hand_sizes[pv] = pv_hand_size

    # Fill other players
    idx = 0
    for pl in range(4):
        if pl == pv:
            continue
        need = int(target_hand_sizes[pl])
        cards_for_pl = remaining[idx:idx + need]
        idx += need
        state.hands[pl, cards_for_pl] = 1

    # If anything is left, put it nowhere (should not happen, but avoids crash)
    return state
