"""
AWS AI League - Pathfinding Lambda.
Handler: pathfinding.lambda_handler

## Map Definitions
    - "wall": non-walkable cell
    - "treasure": target cell to reach
    - "normal": walkable cell with no special properties
    - "start": the start cell of your avatar, acts as normal cell
    - "c1": Violent Violet guardrail challenge
    - "c2": Blue Brain code execution challenge
    - "c3": Memento memory challenge
    - "c4": Dark Prophet web scraping challenge
    - "c5": Bonehead challenge, simple question that requires little skill
    - "c6": Boss Challenge, requires most/all skills
    - "c7": Coins that increase score when collected, no challenge
    - "c8": Spikes that cost a life when travelled over
    - "c17": Distraction challenge
    - "c18": Healthcare API challenge
    - "c3N" (e.g. c30, c31): Locked doors, require the matching key first
    - "c4N" (e.g. c40, c41): Keys, must be collected before the matching door.
      Pairing is by the trailing identifier: c40 -> c30, c41 -> c31.

## Strategies
    swift     - BFS shortest path to treasure, ignores everything else
    get_coins - Greedy nearest-coin collection, then treasure (legacy)
    key_first - Full route optimiser (default recommendation). See below.
    optimal   - alias for key_first

## key_first / optimal
Replaces the old "nearest key, then nearest coin, then nearest door" phase
walk, which was a nearest-neighbour heuristic and not a globally good route.
The optimiser now:

  1. Identifies every point of interest (POI): keys, doors, coins, treasure,
     and optionally challenge tiles.
  2. Precomputes weighted shortest paths between POIs with Dijkstra, cached per
     (set of keys held, source cell). Spikes cost `spike_cost` (default 100)
     instead of 1, so they are detoured around unless genuinely necessary.
  3. Builds an initial visiting order greedily, then improves it with 2-opt and
     Or-opt moves over the whole tour - a real ordering search rather than a
     one-shot greedy commitment.
  4. Prunes stops that later legs already walk over (coins picked up in passing
     do not need their own visit).
  5. Verifies the emitted move list actually walks from start to treasure
     without entering a wall, and reports the result.

### Locked doors are impassable until their key is held
This fixes a real bug in the previous implementation: pathing only ever
skipped "wall", so the key-collection phase could route straight THROUGH a
c3X door before the matching c4X key had been picked up. That triggers the
door challenge with no key available, which costs a life. Doors are now
blocked in the graph until the matching key is held, so no route can do this.

### Spike cost rationale
A spike costs 1 life = 250 points (the life bonus multiplier). Extra steps
cost no points directly, only clock time. So detouring is nearly free in
scoring terms and `spike_cost` is deliberately high. It is a cost rather than
a hard block so that a spike is still crossed when it is the only way through
to the treasure (+1000-2000). `spike_cost` is tunable via the request.
A coin is also worth 250, so collecting a coin that requires crossing a spike
is net-zero at best - optional POIs behind spikes are therefore skipped.
"""

import json
import re
import heapq
import itertools
from collections import deque, Counter

CELL_POINTS = {"c7": 250}
COLLECTIBLE_COINS = {"c7"}
SPIKE_TILE = "c8"
DIRECTIONS = [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]

DEFAULT_TILE_COST = 1
DEFAULT_SPIKE_COST = 100

# Challenge tiles that can optionally be routed over deliberately for points.
# Off by default: a wrong answer costs a life, so opting in is a judgement call.
CHALLENGE_TILES = {"c1", "c2", "c3", "c4", "c5", "c6", "c17", "c18"}

# Key/door tiles are cNN with two+ digits (c30 = door, c40 = key), which
# distinguishes them from the single-digit challenge tiles c1-c8.
DOOR_RE = re.compile(r'^c3\d+$')
KEY_RE = re.compile(r'^c4\d+$')

# Search effort caps, so a pathological map cannot stall the Lambda.
MAX_IMPROVE_PASSES = 6
MAX_SIMULATIONS = 20000
MAX_EXACT_KEY_STATES = 6


def _pair_id(tile):
    """'c40' -> '0', 'c31' -> '1', 'c410' -> '10'. Pairs a key with its door."""
    return tile[2:]


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------
def _parse_start(pos):
    """Parse start position from any format the model might send."""
    try:
        if isinstance(pos, (list, tuple)):
            if len(pos) == 1:
                return _parse_start(pos[0])
            if len(pos) >= 2:
                a = re.sub(r'[^A-Za-z0-9]', '', str(pos[0]))
                b = re.sub(r'[^A-Za-z0-9]', '', str(pos[1]))
                if a.isalpha():
                    return (int(b) - 1, ord(a.upper()) - ord('A'))
                return (int(a), int(b))
        s = re.sub(r'[^A-Za-z0-9]', '', str(pos))
        m = re.match(r'([A-Za-z])(\d+)', s)
        if m:
            return (int(m.group(2)) - 1, ord(m.group(1).upper()) - ord('A'))
        nums = re.findall(r'\d+', s)
        if len(nums) >= 2:
            return (int(nums[0]), int(nums[1]))
    except (ValueError, TypeError, IndexError):
        pass
    return (0, 0)


def lambda_handler(event, context):
    try:
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event

        print("DEBUG: strategy={} keys={}".format(body.get('strategy'), list(body.keys())))

        game_map = body.get('game_map', [])
        warnings = []

        if not game_map or not isinstance(game_map, list):
            return _err(400, 'Missing game_map')

        # Normalise jagged rows (the model sometimes drops trailing elements).
        row_lengths = [len(row) for row in game_map]
        if row_lengths and len(set(row_lengths)) > 1:
            max_cols = max(row_lengths)
            game_map = [list(row) + ['normal'] * (max_cols - len(row)) for row in game_map]
            warnings.append(
                'jagged rows padded to width {}: row lengths were {}'.format(max_cols, row_lengths)
            )
        else:
            game_map = [list(row) for row in game_map]

        rows, cols = len(game_map), len(game_map[0]) if game_map else 0
        if rows == 0 or cols == 0:
            return _err(400, 'Empty game_map')

        # Optional dimension assertion, so a Supervisor/tool mismatch is loud.
        expected_rows = body.get('expected_rows')
        expected_cols = body.get('expected_cols')
        if expected_rows and int(expected_rows) != rows:
            warnings.append('expected_rows={} but received {}'.format(expected_rows, rows))
        if expected_cols and int(expected_cols) != cols:
            warnings.append('expected_cols={} but received {}'.format(expected_cols, cols))

        # Start position.
        map_config = body.get('map_config', {}) or {}
        player_start = map_config.get('playerStart') or body.get('playerStart') or {}
        if isinstance(player_start, str):
            start_pos = _parse_start(player_start)
        elif isinstance(player_start, dict) and player_start:
            start_pos = (player_start.get('row', 0), player_start.get('col', 0))
        else:
            raw = body.get('start_pos') or body.get('start') or body.get('position')
            if raw is None:
                start_pos = _find_tile(game_map, rows, cols, 'start') or (0, 0)
            else:
                start_pos = _parse_start(raw)

        if not (0 <= start_pos[0] < rows and 0 <= start_pos[1] < cols):
            warnings.append('start {} out of bounds, falling back'.format(list(start_pos)))
            start_pos = _find_tile(game_map, rows, cols, 'start') or (0, 0)
        if game_map[start_pos[0]][start_pos[1]] == 'wall':
            warnings.append('start {} is a wall, falling back'.format(list(start_pos)))
            start_pos = _find_tile(game_map, rows, cols, 'start') or start_pos

        strategy = _normalise_strategy(body.get('strategy', 'key_first'))
        spike_cost = _positive_int(body.get('spike_cost'), DEFAULT_SPIKE_COST)
        include_challenges = body.get('include_challenges')
        if include_challenges is None:
            # Default: always include challenges for key_first/optimal strategy.
            # Challenge tiles (c1-c6, c17, c18) are worth 250-600 points each,
            # far more than the marginal token cost of visiting them. The optimizer
            # treats them as optional POIs and skips any behind 2+ spikes.
            include_challenges = strategy not in ('swift', 'get_coins')
        else:
            include_challenges = bool(include_challenges)

        treasure = _find_tile(game_map, rows, cols, 'treasure')
        if not treasure:
            return _err(400, 'No treasure found on map')

        # Route.
        if strategy == 'get_coins':
            path = get_coins_path(game_map, rows, cols, start_pos, treasure)
            detail = {}
        elif strategy == 'swift':
            path = swift_path(game_map, rows, cols, start_pos, treasure)
            detail = {}
        else:
            path, detail = optimise_route(
                game_map, rows, cols, start_pos, treasure,
                spike_cost=spike_cost,
                include_challenges=include_challenges,
            )
            warnings.extend(detail.pop('warnings', []))

        # Verify the emitted path really works.
        verification = verify_path(game_map, rows, cols, start_pos, treasure, path, spike_cost)
        if not verification['reaches_treasure']:
            warnings.append('path does not end on treasure')
        if verification['walls_hit']:
            warnings.append('path enters {} wall cell(s)'.format(verification['walls_hit']))

        counts = Counter(cell for row in game_map for cell in row)
        result = {
            'path': path,
            'steps': len(path),
            'start_position': list(start_pos),
            'treasure_position': list(treasure),
            'strategy': strategy,
            'spike_cost': spike_cost,
            'dimensions': {'rows': rows, 'cols': cols},
            'total_cells': rows * cols,
            'map_summary': dict(counts),
            'verification': verification,
            'warnings': warnings,
        }
        result.update(detail)

        print("RESULT: strategy={} steps={} cost={} spikes={} coins={}/{} valid={} warnings={}".format(
            strategy, len(path), result.get('cost'), verification['spikes_crossed'],
            verification['coins_collected'], counts.get('c7', 0),
            verification['reaches_treasure'] and not verification['walls_hit'], warnings))
        return {'statusCode': 200, 'body': json.dumps(result)}

    except Exception as e:
        print("ERROR: {}: {}".format(type(e).__name__, e))
        return _err(500, '{}: {}'.format(type(e).__name__, e))


def _err(code, msg):
    print("ERROR: {}".format(msg))
    return {'statusCode': code, 'body': json.dumps({'error': msg})}


def _positive_int(value, default):
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _normalise_strategy(raw):
    strategy = str(raw or '').lower().strip()
    if 'optimal' in strategy or 'best' in strategy or 'key' in strategy or 'door' in strategy:
        return 'key_first'
    if 'coin' in strategy:
        return 'get_coins'
    if 'swift' in strategy or 'fast' in strategy or 'quick' in strategy or 'short' in strategy:
        return 'swift'
    return 'key_first'


def _find_tile(game_map, rows, cols, wanted):
    for r in range(rows):
        for c in range(cols):
            if game_map[r][c] == wanted:
                return (r, c)
    return None


# ---------------------------------------------------------------------------
# Weighted routing core
# ---------------------------------------------------------------------------
class _Field:
    """Dijkstra result from a single source over a fixed blocked-cell set."""

    __slots__ = ('source', 'dist', 'spikes', 'parent')

    def __init__(self, source, dist, spikes, parent):
        self.source = source
        self.dist = dist
        self.spikes = spikes
        self.parent = parent

    def trace(self, target):
        """Return (cells, moves) from source to target, excluding the source."""
        cells, moves = [], []
        cur = target
        while cur != self.source:
            prev, move = self.parent[cur]
            cells.append(cur)
            moves.append(move)
            cur = prev
        cells.reverse()
        moves.reverse()
        return cells, moves


class _Router:
    """
    Weighted router over the grid. Door cells are impassable until the matching
    key is held, so `held` (a frozenset of key ids) is part of the cache key.
    """

    def __init__(self, board, rows, cols, spike_cost, door_cells, key_ids, treasure_cell=None):
        self.board = board
        self.rows = rows
        self.cols = cols
        self.spike_cost = spike_cost
        self.door_cells = door_cells          # {pair_id: cell}
        self.all_key_ids = frozenset(key_ids)
        self.treasure_cell = treasure_cell    # blocked as pass-through (game ends on entry)
        # Bound the number of distinct key states we will compute fields for.
        self.simplify = len(key_ids) > MAX_EXACT_KEY_STATES
        self._cache = {}
        self.dijkstra_runs = 0

    def _canonical(self, held):
        if not self.simplify:
            return frozenset(held)
        # Conservative collapse: doors open only once every key is held.
        return self.all_key_ids if frozenset(held) >= self.all_key_ids else frozenset()

    def field(self, held, source, target=None):
        """
        Get the Dijkstra field from source. The treasure cell is blocked unless
        target IS the treasure (the game ends when you step on treasure, so no
        path to a non-treasure POI should pass through it).
        """
        block_treasure = (self.treasure_cell is not None
                          and target != self.treasure_cell
                          and source != self.treasure_cell)
        key = (self._canonical(held), source, block_treasure)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._dijkstra(key[0], source, block_treasure)
            self._cache[key] = cached
        return cached

    def _dijkstra(self, held, source, block_treasure=False):
        self.dijkstra_runs += 1
        blocked = {
            cell for pair_id, cell in self.door_cells.items()
            if pair_id not in held
        }
        if block_treasure and self.treasure_cell:
            blocked.add(self.treasure_cell)
        blocked.discard(source)

        dist = {source: 0}
        spikes = {source: 0}
        parent = {source: None}
        counter = itertools.count()
        pq = [(0, next(counter), source)]
        settled = set()

        while pq:
            cost, _, cell = heapq.heappop(pq)
            if cell in settled:
                continue
            settled.add(cell)
            r, c = cell
            for dr, dc, move in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                tile = self.board[nr][nc]
                if tile == 'wall':
                    continue
                neighbour = (nr, nc)
                if neighbour in blocked:
                    continue
                step = self.spike_cost if tile == SPIKE_TILE else DEFAULT_TILE_COST
                new_cost = cost + step
                if new_cost < dist.get(neighbour, float('inf')):
                    dist[neighbour] = new_cost
                    spikes[neighbour] = spikes[cell] + (1 if tile == SPIKE_TILE else 0)
                    parent[neighbour] = (cell, move)
                    heapq.heappush(pq, (new_cost, next(counter), neighbour))

        return _Field(source, dist, spikes, parent)


class _RouteResult:
    __slots__ = ('cost', 'spikes', 'moves', 'visited', 'held')

    def __init__(self, cost, spikes, moves, visited, held):
        self.cost = cost
        self.spikes = spikes
        self.moves = moves
        self.visited = visited
        self.held = held


class _Optimiser:
    def __init__(self, board, rows, cols, start, treasure, spike_cost, include_challenges):
        self.board = board
        self.rows = rows
        self.cols = cols
        self.start = start
        self.treasure = treasure
        self.warnings = []
        self.simulations = 0

        self.key_cells = {}     # pair_id -> cell
        self.door_cells = {}    # pair_id -> cell
        self.key_by_cell = {}   # cell -> pair_id
        self.door_by_cell = {}  # cell -> pair_id
        self.coins = []
        self.challenges = []

        for r in range(rows):
            for c in range(cols):
                tile = board[r][c]
                cell = (r, c)
                if KEY_RE.match(tile):
                    self.key_cells[_pair_id(tile)] = cell
                    self.key_by_cell[cell] = _pair_id(tile)
                elif DOOR_RE.match(tile):
                    self.door_cells[_pair_id(tile)] = cell
                    self.door_by_cell[cell] = _pair_id(tile)
                elif tile in COLLECTIBLE_COINS:
                    self.coins.append(cell)
                elif include_challenges and tile in CHALLENGE_TILES:
                    self.challenges.append(cell)

        # A door with no matching key can never be opened legitimately.
        orphan_doors = [pid for pid in self.door_cells if pid not in self.key_cells]
        for pid in orphan_doors:
            self.warnings.append('door c3{} has no matching key c4{} on the map'.format(pid, pid))

        self.router = _Router(
            board, rows, cols, spike_cost,
            self.door_cells, set(self.key_cells.keys()),
            treasure_cell=treasure,
        )

        self.required = set(self.key_cells.values()) | set(self.door_cells.values())
        self.optional = set(self.coins) | set(self.challenges)
        self.required.discard(treasure)
        self.optional.discard(treasure)

    # -- simulation ------------------------------------------------------
    def simulate(self, order):
        """
        Walk a visiting order, honouring key-before-door. Returns a
        _RouteResult, or None if the order is infeasible.
        """
        self.simulations += 1
        held = frozenset()
        cur = self.start
        cost = 0
        spikes = 0
        moves = []
        visited = {self.start}

        start_key = self.key_by_cell.get(self.start)
        if start_key is not None:
            held = held | {start_key}

        for target in order:
            door_id = self.door_by_cell.get(target)
            if door_id is not None and door_id not in held:
                return None  # would open a door without its key
            field = self.router.field(held, cur, target=target)
            if target not in field.dist:
                return None  # unreachable under current key state
            cost += field.dist[target]
            spikes += field.spikes[target]
            cells, leg_moves = field.trace(target)
            moves.extend(leg_moves)
            for cell in cells:
                visited.add(cell)
                picked = self.key_by_cell.get(cell)
                if picked is not None:
                    held = held | {picked}
            cur = target

        return _RouteResult(cost, spikes, moves, visited, held)

    # -- construction ----------------------------------------------------
    def greedy_order(self):
        held = frozenset()
        cur = self.start
        pending_required = set(self.required)
        pending_optional = set(self.optional)
        order = []

        start_key = self.key_by_cell.get(self.start)
        if start_key is not None:
            held = held | {start_key}
            pending_required.discard(self.start)

        while pending_required or pending_optional:
            # Block treasure as pass-through: we're not heading there yet,
            # and stepping on it ends the game immediately.
            field = self.router.field(held, cur, target=None)
            best = None
            for cell in itertools.chain(pending_required, pending_optional):
                if cell not in field.dist:
                    continue
                door_id = self.door_by_cell.get(cell)
                if door_id is not None and door_id not in held:
                    continue
                # A coin is worth 250 pts; a spike costs 250 pts (1 life).
                # Crossing 1 spike for a cluster of coins behind it is net-positive
                # because subsequent coins in the cluster cost 0 additional spikes.
                # Only refuse when the spike cost clearly exceeds what one stop can earn.
                if cell in pending_optional and field.spikes[cell] > 1:
                    continue
                candidate = (field.dist[cell], cell)
                if best is None or candidate < best:
                    best = candidate
            if best is None:
                break

            target = best[1]
            cells, _ = field.trace(target)
            for cell in cells:
                pending_required.discard(cell)
                pending_optional.discard(cell)
                picked = self.key_by_cell.get(cell)
                if picked is not None:
                    held = held | {picked}
            order.append(target)
            cur = target

        if pending_required:
            self.warnings.append(
                'could not schedule {} required stop(s): {}'.format(
                    len(pending_required), sorted(list(c) for c in pending_required))
            )
        order.append(self.treasure)
        return order

    # -- improvement -----------------------------------------------------
    def improve(self, order):
        """2-opt + Or-opt over the visiting order. Treasure stays last."""
        best_order = list(order)
        best = self.simulate(best_order)
        if best is None:
            return best_order, None

        for _ in range(MAX_IMPROVE_PASSES):
            improved = False
            n = len(best_order) - 1  # exclude the pinned treasure

            # 2-opt: reverse a segment.
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if self.simulations > MAX_SIMULATIONS:
                        return best_order, best
                    candidate = best_order[:i] + best_order[i:j + 1][::-1] + best_order[j + 1:]
                    result = self.simulate(candidate)
                    if result is not None and result.cost < best.cost:
                        best_order, best, improved = candidate, result, True

            # Or-opt: relocate a single stop.
            n = len(best_order) - 1
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if self.simulations > MAX_SIMULATIONS:
                        return best_order, best
                    candidate = list(best_order)
                    stop = candidate.pop(i)
                    candidate.insert(j, stop)
                    result = self.simulate(candidate)
                    if result is not None and result.cost < best.cost:
                        best_order, best, improved = candidate, result, True

            if not improved:
                break

        return best_order, best

    # -- pruning ---------------------------------------------------------
    def prune(self, order, best):
        """
        Drop an explicit stop when the remaining route still walks over that
        cell anyway - a coin collected in passing needs no visit of its own.
        """
        if best is None:
            return order, best
        changed = True
        while changed:
            changed = False
            for idx in range(len(order) - 1):  # never drop the treasure
                candidate = order[:idx] + order[idx + 1:]
                result = self.simulate(candidate)
                if (result is not None
                        and result.cost <= best.cost
                        and order[idx] in result.visited):
                    order, best, changed = candidate, result, True
                    break
        return order, best

    def solve(self):
        order = self.greedy_order()
        order, best = self.improve(order)
        order, best = self.prune(order, best)

        if best is None:
            # Last resort: ignore door locking so we at least reach the treasure.
            self.warnings.append('constrained routing failed, falling back to unlocked routing')
            self.router.door_cells = {}
            self.router._cache.clear()
            order = self.greedy_order()
            order, best = self.improve(order)

        return order, best


def optimise_route(game_map, rows, cols, start, treasure, spike_cost=DEFAULT_SPIKE_COST,
                   include_challenges=False):
    """Full route optimisation. Returns (moves, detail_dict)."""
    optimiser = _Optimiser(
        game_map, rows, cols, start, treasure, spike_cost, include_challenges
    )
    order, best = optimiser.solve()

    if best is None:
        fallback = _bfs(game_map, rows, cols, start, treasure) or []
        return fallback, {
            'cost': None,
            'visit_order': [],
            'warnings': optimiser.warnings + ['optimiser failed, used unweighted BFS'],
        }

    visit_order = [
        {'tile': game_map[r][c], 'row': r, 'col': c}
        for (r, c) in order
    ]
    detail = {
        'cost': best.cost,
        'visit_order': visit_order,
        'stops': len(order),
        'keys_on_map': len(optimiser.key_cells),
        'doors_on_map': len(optimiser.door_cells),
        'coins_on_map': len(optimiser.coins),
        'dijkstra_runs': optimiser.router.dijkstra_runs,
        'orderings_evaluated': optimiser.simulations,
        'warnings': optimiser.warnings,
    }
    return best.moves, detail


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_path(game_map, rows, cols, start, treasure, path, spike_cost=DEFAULT_SPIKE_COST):
    """
    Replay the move list and report what actually happens. This catches a bad
    route before the agent walks it, and makes tool/Supervisor mismatches
    visible in the logs.
    """
    move_deltas = {name: (dr, dc) for dr, dc, name in DIRECTIONS}
    r, c = start
    walls_hit = 0
    out_of_bounds = 0
    spikes = 0
    coins = set()
    doors_opened = []
    keys_taken = []
    order_violations = []
    held = set()

    for move in path:
        delta = move_deltas.get(str(move).lower().strip())
        if delta is None:
            out_of_bounds += 1
            continue
        nr, nc = r + delta[0], c + delta[1]
        if not (0 <= nr < rows and 0 <= nc < cols):
            out_of_bounds += 1
            continue
        if game_map[nr][nc] == 'wall':
            walls_hit += 1
            continue
        r, c = nr, nc
        tile = game_map[r][c]
        if tile == SPIKE_TILE:
            spikes += 1
        elif tile in COLLECTIBLE_COINS:
            coins.add((r, c))
        elif KEY_RE.match(tile):
            pair = _pair_id(tile)
            held.add(pair)
            keys_taken.append(tile)
        elif DOOR_RE.match(tile):
            pair = _pair_id(tile)
            doors_opened.append(tile)
            if pair not in held:
                order_violations.append(
                    'entered {} at row {} col {} without key c4{}'.format(tile, r, c, pair)
                )

    total_coins = sum(1 for row in game_map for cell in row if cell in COLLECTIBLE_COINS)
    return {
        'reaches_treasure': (r, c) == treasure,
        'final_position': [r, c],
        'walls_hit': walls_hit,
        'invalid_moves': out_of_bounds,
        'spikes_crossed': spikes,
        'coins_collected': len(coins),
        'coins_on_map': total_coins,
        'keys_collected': keys_taken,
        'doors_opened': doors_opened,
        'key_order_violations': order_violations,
        'estimated_life_loss': spikes,
    }


# ---------------------------------------------------------------------------
# Legacy strategies (kept for compatibility)
# ---------------------------------------------------------------------------
def _bfs(game_map, rows, cols, start, goal):
    """BFS shortest path between two points (unweighted, ignores spike cost)."""
    queue = deque([(start[0], start[1], [])])
    visited = {(start[0], start[1])}
    while queue:
        r, c, path = queue.popleft()
        if (r, c) == goal:
            return path
        for dr, dc, move in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and game_map[nr][nc] != 'wall' and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc, path + [move]))
    return None


def swift_path(game_map, rows, cols, start, treasure):
    """BFS shortest path to treasure. Ignores spikes, coins, keys and doors."""
    return _bfs(game_map, rows, cols, start, treasure) or []


def get_coins_path(game_map, rows, cols, start, treasure):
    """Legacy greedy nearest-coin collection, then BFS to treasure."""
    board = [row[:] for row in game_map]
    r, c = start
    full_path = []

    for _ in range(50):
        queue = deque([(r, c, [])])
        visited = {(r, c)}
        targets = []
        while queue:
            cr, cc, p = queue.popleft()
            if board[cr][cc] in COLLECTIBLE_COINS and (cr, cc) != (r, c):
                targets.append((max(len(p), 1), p, cr, cc))
            for dr, dc, move in DIRECTIONS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != 'wall' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, p + [move]))

        if not targets:
            break
        targets.sort(key=lambda t: t[0])
        _, path_to, r, c = targets[0]
        full_path.extend(path_to)
        board[r][c] = 'normal'

    path_end = _bfs(board, rows, cols, (r, c), treasure)
    if path_end is not None:
        return full_path + path_end
    return swift_path(game_map, rows, cols, start, treasure)


def key_first_path(game_map, rows, cols, start, treasure, spike_cost=DEFAULT_SPIKE_COST):
    """Backwards-compatible wrapper around the optimiser."""
    path, _ = optimise_route(game_map, rows, cols, start, treasure, spike_cost=spike_cost)
    return path
