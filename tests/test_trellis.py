from voxalign.align.trellis import (
    build_state_symbols,
    token_spans_from_state_path,
    viterbi_state_path,
)


def test_build_state_symbols() -> None:
    assert build_state_symbols([]) == [0]
    assert build_state_symbols([11, 22]) == [0, 11, 0, 22, 0]


def test_viterbi_state_path_basic() -> None:
    # Emissions for symbols: blank(0), token1(1), token2(2)
    emissions = [
        [-0.1, -2.0, -2.0],
        [-1.5, -0.2, -2.0],
        [-0.5, -0.3, -1.2],
        [-1.8, -1.0, -0.2],
        [-0.2, -2.0, -1.0],
    ]
    state_symbols = build_state_symbols([1, 2])  # [0,1,0,2,0]
    path = viterbi_state_path(emissions, state_symbols)

    assert len(path) == len(emissions)
    assert all(0 <= state < len(state_symbols) for state in path)
    assert path[0] in {0, 1}


def test_token_spans_from_state_path() -> None:
    # Frame-wise states: blank, token1, token1, blank, token2
    path = [0, 1, 1, 2, 3]
    spans = token_spans_from_state_path(path, token_count=2)

    assert len(spans) == 2
    assert spans[0].start_frame == 1
    assert spans[0].end_frame == 3
    assert spans[1].start_frame == 4
    assert spans[1].end_frame == 5
