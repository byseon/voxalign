"""CTC trellis utilities.

This module contains the core dynamic-programming pieces that can be reused
once real acoustic emissions are wired in.
"""

from __future__ import annotations

from dataclasses import dataclass


def build_state_symbols(token_count: int) -> list[int]:
    """Build expanded CTC state sequence symbols.

    States alternate between blank (0) and token ids (1..N):
    [0, 1, 0, 2, 0, ..., N, 0]
    """
    states: list[int] = [0]
    for token_id in range(1, token_count + 1):
        states.append(token_id)
        states.append(0)
    return states


def viterbi_state_path(emissions: list[list[float]], state_symbols: list[int]) -> list[int]:
    """Run Viterbi over expanded CTC states.

    `emissions[t][k]` are log-probabilities for frame `t` and symbol `k`.
    Returns the best state index per frame.
    """
    frame_count = len(emissions)
    state_count = len(state_symbols)
    if frame_count == 0:
        return []
    if state_count == 0:
        raise ValueError("state_symbols must not be empty")

    neg_inf = float("-inf")
    scores = [[neg_inf] * state_count for _ in range(frame_count)]
    backptr = [[0] * state_count for _ in range(frame_count)]

    # Frame 0: start at blank state, optionally token-1 state.
    scores[0][0] = emissions[0][state_symbols[0]]
    if state_count > 1:
        scores[0][1] = emissions[0][state_symbols[1]]
        backptr[0][1] = 0

    for frame in range(1, frame_count):
        for state in range(state_count):
            stay = scores[frame - 1][state]
            move = scores[frame - 1][state - 1] if state > 0 else neg_inf

            if move > stay:
                best_prev = state - 1
                best_score = move
            else:
                best_prev = state
                best_score = stay

            if best_score == neg_inf:
                continue

            symbol_id = state_symbols[state]
            scores[frame][state] = best_score + emissions[frame][symbol_id]
            backptr[frame][state] = best_prev

    end_candidates = [state_count - 1]
    if state_count > 1:
        end_candidates.append(state_count - 2)

    last_frame = frame_count - 1
    best_end_state = max(end_candidates, key=lambda idx: scores[last_frame][idx])

    path = [0] * frame_count
    cursor = best_end_state
    for frame in range(last_frame, -1, -1):
        path[frame] = cursor
        if frame > 0:
            cursor = backptr[frame][cursor]

    return path


@dataclass(frozen=True)
class TokenFrameSpan:
    """Token-to-frame span mapping."""

    token_index: int
    start_frame: int
    end_frame: int


def token_spans_from_state_path(
    state_path: list[int],
    state_symbols: list[int],
    token_count: int,
) -> list[TokenFrameSpan]:
    """Extract frame spans for each token from the decoded state path."""
    spans: list[TokenFrameSpan] = []
    for token_index in range(token_count):
        token_symbol = token_index + 1
        token_frames = [
            frame_idx
            for frame_idx, state_idx in enumerate(state_path)
            if state_symbols[state_idx] == token_symbol
        ]
        if token_frames:
            start_frame = token_frames[0]
            end_frame = token_frames[-1] + 1
        else:
            # Safety fallback for pathological paths.
            start_frame = 0
            end_frame = 0
        spans.append(
            TokenFrameSpan(
                token_index=token_index,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
    return spans
