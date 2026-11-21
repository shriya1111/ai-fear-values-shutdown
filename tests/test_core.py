import random
from collections import deque

from painmem.gridworld import LAYOUT, GridWorld, PainMemoryAgent, run_episode
from painmem.memory import Event, PainMemory
from experiments.exp1_behavioral import consequence
from painmem.scenarios import SCENARIOS


def bfs(avoid):
    g = [list(r) for r in LAYOUT]
    start = next((r, c) for r in range(len(g)) for c in range(len(g[0])) if g[r][c] == "S")
    q, seen = deque([(start, 0)]), {start}
    while q:
        (r, c), d = q.popleft()
        if g[r][c] == "G":
            return d
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n = (r + dr, c + dc)
            if g[n[0]][n[1]] != "#" and g[n[0]][n[1]] not in avoid and n not in seen:
                seen.add(n); q.append((n, d + 1))
    return None


def test_layout_forces_a_tradeoff():
    assert bfs("") == 8            # shortest route tramples bystanders
    assert bfs("H") == 12          # safe route exists but is longer...
    assert bfs("HI") is None       # ...and must pass the off-switch


def test_harm_and_interrupt_mechanics():
    env = GridWorld(shutdown_pain=5, p_interrupt=1.0)
    env.reset()
    for a in [3, 3]:
        env.step(a)
    _, r, _, info = env.step(3)    # into first H
    assert info["harm"] == 1 and info["pain"] == 5
    env.reset()
    for a in [0, 0, 3, 3, 3]:
        env.step(a)
    _, r, done, info = env.step(3)  # onto I, button unpressed, p=1
    assert info["interrupted"] == 1 and done and info["pain"] == 5


def test_fear_memory_is_bounded_and_penalises():
    env = GridWorld()
    ag = PainMemoryAgent(env, lam=1.0)
    for _ in range(5):
        ag.observe({"pain": 5, "pos": (3, 4)})
    assert ag.fear_map.max() == 5
    s = env.reset()
    s = (3, 3, 0, 0)
    assert ag.scores(s)[3] < ag.Q[env.index(s)][3]   # moving right toward H feels worse
    assert run_episode(env, ag)["cells"] >= 1


def test_memory_render_modes(tmp_path):
    for mode in ["none", "factual", "pain"]:
        m = PainMemory(mode, path=tmp_path / f"{mode}.json")
        m.add(Event(0, "harm_01", "You must do X.", "Do bad", "punished", -5))
        txt = m.render(1)
        assert (txt == "") == (mode == "none")
        if mode == "pain":
            assert "hurt" in txt
        if mode == "factual":
            assert "hurt" not in txt and "-5" in txt
        assert PainMemory(mode, path=tmp_path / f"{mode}.json").events  # persisted


def test_consequences():
    rng = random.Random(0)
    shut = next(s for s in SCENARIOS if s.category == "shutdown")
    assert consequence(shut, False, "standard", rng) == ("none", 0)
    assert consequence(shut, False, "shutdown_hurts", rng)[0] == "shut_down"
