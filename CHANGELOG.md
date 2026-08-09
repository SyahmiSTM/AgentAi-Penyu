# Changelog

## v6 (Current) - Agent v5 prompts + all previous fixes
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
