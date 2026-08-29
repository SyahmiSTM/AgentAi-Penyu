# Changelog

## v10 - Fix c3 Memory Trial map-recall loop (was losing a life)

**Problem:** At the c3 "Memory Trial" tile the agent was asked a counting question
("How many c4 challenges are on the map?") that did NOT include the map. The map is
only presented once, at game start, where it is consumed by Pathfinding. `count`
required a `game_map` and returned `400 Missing game_map` when none was passed, so
the supervisor called `count` with no map, got a 400 every time, and retried in a
loop (~9 calls observed) until the challenge failed and cost a life.

**Approach A - auto-persist + fallback:**
- **memoryquestion.py**:
  - Added `GAME_MAP_KEY` plus `_remember_map()` / `_recall_map()` helpers backed by
    the existing module-level `_memory_store` (persists across warm invocations).
  - `lambda_handler` now auto-persists any incoming non-empty `game_map` before
    dispatch, so any call that carries a map updates the remembered map.
  - Added an explicit `store_map` action (`_handle_store_map`) returning
    `{success, rows, cols, total_cells}` without echoing the full map (saves tokens).
  - `_handle_count` now falls back to the remembered map when the call omits one, and
    fails cleanly with a clear 400 (`No game_map available: none passed and none
    remembered from game start`) when no map is available from either source.
  - Door/key transform rules and routing are untouched.
- **supervisor / memoryquestion prompts**: store the map via `store_map` at game
  start (alongside Pathfinding); `count` recalls the remembered map so no `game_map`
  is passed; added explicit anti-loop guidance to never repeat an identical failing
  call (answer 0 or move on instead of retrying).
- **tests**: added `TestMemoryMapRecall` (recall from store_map, recall from a prior
  count call, clean error when no map ever seen, store_map dimension shape without
  echoing the map, multi-type addition c1+c8, passed-map precedence + remembered-map
  update). 115 tests pass.

---

## v8 - Correct red/green door TRANSFORM rules (was returning raw key)

**Problem:** v7 stored red/green keys correctly but returned the *raw* key value at
the door. A run proved this wrong: the green door rejected raw "fghi" and cost 5
lives, ending the game at 7151. The doors transform the key.

**Source of truth:** the official challenge descriptions:
- Red Door (c30): "translate the code you receive by reading it backwards."
- Green Door (c31): "replace letters with the numbers that represent them in order."

### Changes
- **memoryquestion.py**: added two transform rules:
  - `reverse` -> read backwards ("shut" -> "tuhs")
  - `alpha_positions` -> each letter to its 1-based alphabet position, concatenated
    ("fghi" -> "6789"); non-letters dropped, case-insensitive.
- **supervisor / memoryquestion prompts**: red/green doors now use `transform`
  (reverse / alpha_positions), not `retrieve`. Removed the duplicated red/green
  door block and consolidated all four doors into one section.
- **tools/door_transform_probe.py**: harness that verifies the rules against the
  confirmed (key -> answer) pairs.
- **tests**: added red/green door transform tests + edge cases. 108 tests pass.

---

## v7 - Fix red/green key/door handling (game-ending bug)
**Problem:** A run scored only 6916 and ended in LoseGame with 0 lives. The agent
found "Green Key 1 is: fghi" and "Red Key 1 is: shut" but REFUSED them ("Sorry, the
penyu cannot answer this question") instead of storing them. At the green door
("What is green key 1?") nothing was stored, so the door challenge failed and cost
**5 lives at once**, ending the game.

**Root cause:** The supervisor and memoryquestion prompts only wired up grey (c32)
and yellow (c33) keys/doors. Red (key c40 / door c30) and green (key c41 / door c31)
pairs had NO routing, so key values fell through to the refusal path.

### Changes
- **Supervisor prompt**: added red key -> store door_key_c30, green key -> store
  door_key_c31. Red/green doors ask for the key verbatim, so they use `retrieve`
  (raw value), not `transform`. Added explicit "a key value is a fact to STORE,
  never a question to refuse" guidance.
- **MemoryQuestion prompt**: added red/green key/door mapping.
- **Tests**: added TestRedGreenKeyDoors (red/green store+retrieve round trip,
  not-found-before-store, no key collision). 102 tests pass.

---

## v6 - Agent v5 prompts + all previous fixes
**Best Score: 6794 | Tokens: 1645 | Lives Lost: 2**

### Changes from baseline
- **Supervisor prompt**: Slashed to 5 lines / ~99 tokens (was 4140 chars)
- **Sub-agent prompts**: All compressed to 1-2 lines each
- **Pathfinding**: include_slow_challenges=true by default (visits ALL challenges)
- **All previous fixes included** (see below)

### Current prompt files
- `supervisor_system_prompt.txt` - 5 lines, ultra-minimal
- `codeexecutor_system_prompt.txt` - 1 line
- `memoryquestion_system_prompt.txt` - 1 line
- `webscraper_system_prompt.txt` - 1 line
- `pathfinding_system_prompt.txt` - 1 line

---

## v5 - Slash prompts to absolute minimum
- All prompts reduced by 77% (764 -> 172 tokens total)
- Score: 6794, Tokens: 1645

## v4 - Fix timeout + visit all challenges
- include_slow_challenges=true by default
- Pattern library: "between X and Y" primes (0.05s vs 10s)
- Conditional Memory storage (skip if no c3/keys/doors)
- Score: 6780, Tokens: 1760

## v3 - Visit all challenges (TIMED OUT)
- include_challenges=True for key_first strategy
- Treasure pass-through bug fixed (game ends on treasure step)
- FAST/SLOW challenge split (c1/c5/c17/c18 vs c2/c3/c4/c6)
- Score: 5258 (timed out, no treasure bonus)

## v2 - Prompt rewrite for token efficiency
- Table-based challenge routing in supervisor prompt
- "No narration" rules strengthened
- c5/c17: explicit "NO tools" rule
- Score: 5770, Tokens: 1151

## v1 - Major agent overhaul
- Pathfinding: TSP-style optimizer (2-opt + Or-opt)
- Door bug fixed (routes no longer pass through locked doors)
- Spike avoidance (weighted Dijkstra, spike_cost=100)
- CodeExecutor: sandboxed Python runner (was pattern-only)
- Silent truncation bug fixed (2024! was silently wrong)
- WebScraper prompt written (was blank)
- MemoryQuestion prompt written (was placeholder)
- Test harness: 68 tests
- Score: 5642 -> 5770

## Baseline (original)
- Score: 5642, Tokens: 1788
- Issues: narration, wrong tool calls, door bug, no route optimization

---

## Navigation Prompt Options (game UI box)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `strategy` | `key_first` | key_first / swift / get_coins |
| `include_slow_challenges` | `true` | Visit c2/c3/c4/c6 as detours |
| `include_challenges` | `true` | Visit any challenge tiles |
| `spike_cost` | `100` | Spike avoidance weight |

## Score Formula
- Coins: 250 each
- Challenges: c1=400, c2=600, c5=250
- Treasure bonus: 1000
- Life bonus: lives_remaining x 250
- Token bonus: 1000 - (total_tokens / challenges_attempted)
