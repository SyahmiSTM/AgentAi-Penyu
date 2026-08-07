import json
import re
import heapq
import itertools
from collections import deque

CELL_POINTS = {"c7": 250}
COLLECTIBLE_COINS = {"c7"}
DIRECTIONS = [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]

# Tile movement costs for weighted (Dijkstra) pathing.
# c8 (spike trap) is treated as high-cost so it's only crossed if there is
# truly no other way to reach the target.
TILE_COST = {
    "c8": 100,
}
DEFAULT_TILE_COST = 1

# Key/door tiles are named cNN with two+ digits (e.g. c30 = door, c40 = key),
# which distinguishes them from the single-digit challenge tiles c1-c8.
DOOR_RE = re.compile(r'^c3\d+$')
KEY_RE = re.compile(r'^c4\d+$')


def _parse_start(pos):
    """Parse start position from any format Nova might send."""
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
    """
    AWS Lambda function for pathfinding using Swift path strategy by default
    Handles both API Gateway format and direct AgentCore Gateway format

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
        - "c6": Boss Challenge, most requires all skills
        - "c7": Coins that increase score when collected with no challenge
        - "c8": Spikes that reduce health traveled over
        - "c3N" (e.g. c30, c31): Locked doors, require the matching key first
        - "c4N" (e.g. c40, c41): Keys, must be collected before opening the
          matching door (c3N)

    ## Map JSON Example
    [
        ["start","normal","c5","normal","normal","normal","c5","normal","normal","c1"],
        ["normal","wall","wall","normal","wall","wall","wall","wall","wall","normal"],
        ["c8","wall","wall","c5","wall","c7","c7","c7","wall","c3"],
        ["normal","wall","c8","normal","wall","c8","wall","c8","wall","normal"],
        ["normal","wall","c7","normal","wall","normal","normal","normal","wall","normal"],
        ["c5","wall","c7","normal","wall","c5","wall","normal","wall","c5"],
        ["normal","wall","c7","normal","wall","normal","wall","normal","wall","normal"],
        ["c1","wall","c8","normal","c2","normal","wall","normal","c4","normal"],
        ["normal","wall","wall","wall","wall","wall","wall","normal","normal","c7"],
        ["c7","normal","c3","normal","c4","normal","c2","normal","treasure","normal"]
    ]

    ## Pathfinding Lambda with strategy selection.

    Usage: Use strategy get_coins / key_first

    Strategies:
      swift     - BFS shortest path to treasure (default)
      get_coins - Greedily collect c7 coins on the way to treasure
      key_first - Collect all keys before opening doors, avoid spikes (c8),
                  and collect every reachable coin (c7) along the way
    """
    try:
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event

        print(f"DEBUG: Received event: {body}")
        game_map = body.get('game_map', [])

        # Fix jagged rows (model sometimes drops elements)
        if game_map:
            max_cols = max(len(row) for row in game_map)
            game_map = [row + ['normal'] * (max_cols - len(row)) for row in game_map]

        # Parse start position from any format
        map_config = body.get('map_config', {})
        player_start = map_config.get('playerStart') or body.get('playerStart') or {}
        if isinstance(player_start, str):
            start_pos = _parse_start(player_start)
        elif isinstance(player_start, dict) and player_start:
            start_pos = (player_start.get('row', 0), player_start.get('col', 0))
        else:
            raw = body.get('start_pos') or body.get('start') or body.get('position') or [0, 0]
            start_pos = _parse_start(raw)

        # Validate start is within map bounds
        if game_map and (start_pos[0] < 0 or start_pos[1] < 0 or start_pos[0] >= len(game_map) or start_pos[1] >= len(game_map[0])):
            start_pos = (0, 0)

        # Normalize strategy name
        strategy = str(body.get('strategy', 'swift')).lower().strip()
        if 'key' in strategy or 'door' in strategy:
            strategy = 'key_first'
        elif 'coin' in strategy:
            strategy = 'get_coins'
        elif 'swift' in strategy or 'fast' in strategy or 'quick' in strategy:
            strategy = 'swift'
        else:
            strategy = 'swift'

        if not game_map:
            return _err(400, 'Missing game_map')

        rows, cols = len(game_map), len(game_map[0])
        treasure = None
        for r in range(rows):
            for c in range(cols):
                if game_map[r][c] == 'treasure':
                    treasure = (r, c)
                    break
            if treasure:
                break

        if not treasure:
            return _err(400, 'No treasure found on map')

        if strategy == 'get_coins':
            path = get_coins_path(game_map, rows, cols, start_pos, treasure)
        elif strategy == 'key_first':
            path = key_first_path(game_map, rows, cols, start_pos, treasure)
        else:
            path = swift_path(game_map, rows, cols, start_pos, treasure)

        result = {'path': path, 'steps': len(path), 'start_position': list(start_pos)}
        print(f"RESULT: strategy={strategy} steps={len(path)} start={list(start_pos)}")
        return {'statusCode': 200, 'body': json.dumps(result)}

    except Exception as e:
        print(f"ERROR: {e}")
        return _err(500, str(e))


def _err(code, msg):
    return {'statusCode': code, 'body': json.dumps({'error': msg})}


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
    """BFS shortest path to treasure."""
    return _bfs(game_map, rows, cols, start, treasure) or []


def get_coins_path(game_map, rows, cols, start, treasure):
    """Greedily BFS to best coins-per-step c7 cell, then BFS to treasure."""
    board = [row[:] for row in game_map]
    r, c = start
    full_path = []

    for _ in range(50):
        # BFS to find reachable coins
        queue = deque([(r, c, [])])
        visited = {(r, c)}
        targets = []
        while queue:
            cr, cc, p = queue.popleft()
            if board[cr][cc] in COLLECTIBLE_COINS and (cr, cc) != (r, c):
                dist = max(len(p), 1)
                targets.append((dist, p, cr, cc))
            for dr, dc, move in DIRECTIONS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != 'wall' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, p + [move]))

        if not targets:
            break
        targets.sort()
        _, path_to, r, c = targets[0]
        full_path.extend(path_to)
        board[r][c] = 'normal'

    # BFS to treasure from current position
    path_end = _bfs(board, rows, cols, (r, c), treasure)
    if path_end is not None:
        full_path.extend(path_end)
        return full_path
    return swift_path(game_map, rows, cols, start, treasure)


# --- Weighted (Dijkstra) helpers used by key_first strategy ---------------

def _dijkstra_nearest(board, rows, cols, start, matcher, tile_cost):
    """
    Weighted Dijkstra from start to the nearest cell (by cost, not steps)
    where matcher(tile_value) is True. Spikes (c8) and any other tile in
    tile_cost are weighted higher than normal cells; walls are impassable.
    Returns (path_moves, (r, c)) for the nearest match, or (None, None).
    """
    counter = itertools.count()
    pq = [(0, next(counter), start[0], start[1], [])]
    best_cost = {}
    while pq:
        cost, _, r, c, path = heapq.heappop(pq)
        if best_cost.get((r, c), float('inf')) < cost:
            continue
        best_cost[(r, c)] = cost
        if (r, c) != start and matcher(board[r][c]):
            return path, (r, c)
        for dr, dc, move in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                tile = board[nr][nc]
                if tile == 'wall':
                    continue
                step_cost = tile_cost.get(tile, DEFAULT_TILE_COST)
                ncost = cost + step_cost
                if ncost < best_cost.get((nr, nc), float('inf')):
                    heapq.heappush(pq, (ncost, next(counter), nr, nc, path + [move]))
    return None, None


def _dijkstra_to(board, rows, cols, start, goal, tile_cost):
    """Weighted Dijkstra shortest path from start to an explicit goal cell."""
    counter = itertools.count()
    pq = [(0, next(counter), start[0], start[1], [])]
    best_cost = {}
    while pq:
        cost, _, r, c, path = heapq.heappop(pq)
        if (r, c) == goal:
            return path
        if best_cost.get((r, c), float('inf')) < cost:
            continue
        best_cost[(r, c)] = cost
        for dr, dc, move in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                tile = board[nr][nc]
                if tile == 'wall':
                    continue
                step_cost = tile_cost.get(tile, DEFAULT_TILE_COST)
                ncost = cost + step_cost
                if ncost < best_cost.get((nr, nc), float('inf')):
                    heapq.heappush(pq, (ncost, next(counter), nr, nc, path + [move]))
    return None


def _collect_greedy(board, rows, cols, pos, matcher, tile_cost, max_iters=200):
    """
    Repeatedly move to the nearest (cost-weighted) cell matching `matcher`,
    marking each one collected (set to 'normal') so it isn't re-targeted.
    Returns (path_moves, final_position).
    """
    full_path = []
    for _ in range(max_iters):
        path, target = _dijkstra_nearest(board, rows, cols, pos, matcher, tile_cost)
        if target is None:
            break
        full_path.extend(path)
        pos = target
        board[pos[0]][pos[1]] = 'normal'
    return full_path, pos


def key_first_path(game_map, rows, cols, start, treasure):
    """
    Strategy: key_first
      1. Collect ALL keys (cNN tiles matching c4x) before touching any door.
      2. Collect every reachable coin (c7).
      3. Visit all doors (cNN tiles matching c3x) - safe now since every key
         has already been picked up.
      4. Head to the treasure.
    Spikes (c8) are treated as high-cost throughout, so they're only crossed
    if there's no cheaper route.
    """
    board = [row[:] for row in game_map]
    tile_cost = dict(TILE_COST)
    pos = start
    full_path = []

    # Phase 1: keys before doors
    key_path, pos = _collect_greedy(
        board, rows, cols, pos,
        lambda tile: bool(KEY_RE.match(tile)),
        tile_cost,
    )
    full_path.extend(key_path)

    # Phase 2: collect all coins
    coin_path, pos = _collect_greedy(
        board, rows, cols, pos,
        lambda tile: tile in COLLECTIBLE_COINS,
        tile_cost,
    )
    full_path.extend(coin_path)

    # Phase 3: now safe to open doors (all keys already collected)
    door_path, pos = _collect_greedy(
        board, rows, cols, pos,
        lambda tile: bool(DOOR_RE.match(tile)),
        tile_cost,
    )
    full_path.extend(door_path)

    # Phase 4: head to treasure, still avoiding spikes where possible
    final_path = _dijkstra_to(board, rows, cols, pos, treasure, tile_cost)
    if final_path is not None:
        full_path.extend(final_path)
        return full_path

    fallback = _bfs(board, rows, cols, pos, treasure)
    if fallback is not None:
        full_path.extend(fallback)
        return full_path

    return swift_path(game_map, rows, cols, start, treasure)