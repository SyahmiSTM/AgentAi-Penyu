import json
import re
from collections import Counter

# Matches any challenge-tile token mentioned in the question text, e.g. c1, c2, c7, c30, c40
ID_PATTERN = re.compile(r'\bc(\d+)\b', re.IGNORECASE)

# Module-level key-value store for door keys and other data.
# Persists across invocations while the Lambda container stays warm (i.e. during a game session).
_memory_store = {}

# Well-known key under which the game map is remembered for the whole session.
# The map is only presented once (at game start, for Pathfinding); the c3 (Memento)
# challenge later asks a counting question WITHOUT re-including the map. We remember
# the map the first time any call carries one so count can recall it later.
GAME_MAP_KEY = 'game_map'


def _remember_map(game_map):
    """Store a copy of a non-empty list-of-lists game_map for later recall."""
    if isinstance(game_map, list) and game_map:
        _memory_store[GAME_MAP_KEY] = [list(row) for row in game_map]


def _recall_map():
    """Return the remembered game_map (or None if none has been seen this session)."""
    return _memory_store.get(GAME_MAP_KEY)

# Deterministic transformations applied to stored values at door challenges.
# The Lambda does the character arithmetic so the agent never has to count or slice
# characters itself (LLMs are unreliable at character-position arithmetic).
# Each rule: min_length = shortest value the rule can be applied to,
#            fn        = the transformation,
#            describe  = human-readable requirement used in error messages.
_TRANSFORM_RULES = {
    # Grey doors: first 2 characters + last 2 characters. "ABCDEF" -> "ABEF"
    'first2last2': {
        'min_length': 4,
        'fn': lambda v: v[:2] + v[-2:],
        'describe': 'at least 4 characters (first 2 + last 2)',
    },
    # Yellow doors: 5th character + 7th character, 1-INDEXED. "ABCDEFGH" -> "EG"
    'char5char7': {
        'min_length': 7,
        'fn': lambda v: v[4] + v[6],
        'describe': 'at least 7 characters (5th + 7th, 1-indexed)',
    },
    # Red doors (c30): "read it backwards". "shut" -> "tuhs"
    'reverse': {
        'min_length': 1,
        'fn': lambda v: v[::-1],
        'describe': 'at least 1 character (reversed)',
    },
    # Green doors (c31): "replace letters with the numbers that represent them in
    # order" -- each letter -> its 1-based alphabet position, concatenated.
    # "fghi" -> "6789" (f=6, g=7, h=8, i=9). Non-letters are dropped.
    'alpha_positions': {
        'min_length': 1,
        'fn': lambda v: ''.join(
            str(ord(c) - 96) for c in v.lower() if 'a' <= c <= 'z'
        ),
        'describe': 'at least 1 alphabetic character (letter -> alphabet position)',
    },
}


def lambda_handler(event, context):
    """
    Multi-action memory tool supporting:
      - "count"     : Deterministic map-tile counting for c3 (Memento) questions.
      - "store"     : Store a key-value pair in memory (e.g. door keys).
      - "retrieve"  : Retrieve a previously stored value by key.
      - "transform" : Retrieve a stored value and apply a deterministic character
                      transformation to it (door unlock answers).
      - "store_map" : Remember the game map at game start so count can recall it later.

    Action routing:
      - Any call that carries a non-empty 'game_map' auto-persists it (so it can be
        recalled by a later count that omits the map).
      - If 'action' field is present, dispatch to that action.
      - If no 'action' but 'game_map' and 'question' are present, default to 'count' (backward compat).

    -- count --
    Input:
      { "action": "count", "game_map": [[...]], "question": "c1 + c2" }
    Output:
      { "answer": "3", "breakdown": {"c1": 1, "c2": 2}, ... }

    -- store --
    Input:
      { "action": "store", "key": "door_key_c33", "value": "<key value>" }
    Output:
      { "success": true, "key": "door_key_c33", "value": "<key value>" }

    -- retrieve --
    Input:
      { "action": "retrieve", "key": "door_key_c33" }
    Output (found):
      { "success": true, "key": "door_key_c33", "value": "<key value>" }
    Output (not found):
      { "success": false, "key": "door_key_c33", "error": "Key not found: door_key_c33" }

    -- transform --
    Input:
      { "action": "transform", "key": "door_key_c33", "rule": "char5char7" }
    Output (found):
      { "success": true, "key": "door_key_c33", "rule": "char5char7", "answer": "<result>" }
    Output (not found):
      { "success": false, "key": "door_key_c33", "error": "Key not found: door_key_c33" }
    Supported rules:
      "first2last2"     -> first 2 chars + last 2 chars   (e.g. "ABCDEF"   -> "ABEF")
      "char5char7"      -> 5th char + 7th char, 1-indexed (e.g. "ABCDEFGH" -> "EG")
      "reverse"         -> read the value backwards        (e.g. "shut"     -> "tuhs")
      "alpha_positions" -> each letter to its alphabet position, concatenated
                           (e.g. "fghi" -> "6789")
    The raw stored value is never included in a successful transform response.
    """
    try:
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event

        print(f"DEBUG: Received event: {body}")

        # Auto-persist any incoming map BEFORE dispatch. Any call that carries a
        # non-empty game_map (count, store_map, etc.) updates the remembered map,
        # so a later count that omits the map can still recall it.
        incoming_map = body.get('game_map')
        if isinstance(incoming_map, list) and incoming_map:
            _remember_map(incoming_map)

        # Determine action
        action = body.get('action')

        # Backward compatibility: no action but game_map + question present -> count
        if action is None:
            if body.get('game_map') and body.get('question'):
                action = 'count'
            else:
                return _err(400, "Missing 'action' field. Supported actions: count, store, retrieve, transform")

        if action == 'store':
            return _handle_store(body)
        elif action == 'retrieve':
            return _handle_retrieve(body)
        elif action == 'transform':
            return _handle_transform(body)
        elif action == 'count':
            return _handle_count(body)
        elif action == 'store_map':
            return _handle_store_map(body)
        else:
            return _err(400, f"Unknown action: {action!r}. Supported: count, store, retrieve, transform, store_map")

    except Exception as e:
        print(f"ERROR: {e}")
        return _err(500, str(e))


def _handle_store(body):
    """Store a key-value pair in the module-level memory store."""
    key = body.get('key')
    value = body.get('value')

    if not key:
        return _err(400, "Missing 'key' for store action")
    if value is None:
        return _err(400, "Missing 'value' for store action")

    _memory_store[key] = value
    result = {'success': True, 'key': key, 'value': value}
    print(f"STORE: {key} = {value!r}")
    return {'statusCode': 200, 'body': json.dumps(result)}


def _handle_retrieve(body):
    """Retrieve a value from the module-level memory store by key."""
    key = body.get('key')

    if not key:
        return _err(400, "Missing 'key' for retrieve action")

    if key in _memory_store:
        result = {'success': True, 'key': key, 'value': _memory_store[key]}
        print(f"RETRIEVE: {key} -> {_memory_store[key]!r}")
    else:
        result = {'success': False, 'key': key, 'error': f'Key not found: {key}'}
        print(f"RETRIEVE: {key} -> NOT FOUND")

    return {'statusCode': 200, 'body': json.dumps(result)}


def _handle_transform(body):
    """
    Retrieve a stored value and apply a deterministic character transformation.

    Door unlock answers are computed here rather than by the agent: character-position
    arithmetic is done in Python so it is always correct and correctly 1-indexed.

    The raw stored value is deliberately omitted from the success response -- the agent
    only needs the transformed answer, and echoing the key value invites it to
    "re-check" the arithmetic itself and get it wrong.
    """
    key = body.get('key')
    rule = body.get('rule')

    if not key:
        return _err(400, "Missing 'key' for transform action")

    supported = ', '.join(sorted(_TRANSFORM_RULES))

    if not rule:
        return _err(400, f"Missing 'rule' for transform action. Supported rules: {supported}")

    if rule not in _TRANSFORM_RULES:
        return _err(400, f"Unknown rule: {rule!r}. Supported rules: {supported}")

    # Report a missing key the same way retrieve does, so the agent can distinguish
    # "nothing was ever stored" from a real answer.
    if key not in _memory_store:
        result = {'success': False, 'key': key, 'error': f'Key not found: {key}'}
        print(f"TRANSFORM: {key} rule={rule} -> NOT FOUND")
        return {'statusCode': 200, 'body': json.dumps(result)}

    value = _memory_store[key]
    if not isinstance(value, str):
        value = str(value)

    spec = _TRANSFORM_RULES[rule]
    if len(value) < spec['min_length']:
        # Better to fail loudly than to hand back a truncated / wrong answer.
        return _err(
            400,
            f"Stored value for {key!r} is too short for rule {rule!r}: "
            f"got {len(value)} characters, requires {spec['describe']}"
        )

    answer = spec['fn'](value)
    result = {'success': True, 'key': key, 'rule': rule, 'answer': answer}
    print(f"TRANSFORM: {key} rule={rule} -> {answer!r}")
    return {'statusCode': 200, 'body': json.dumps(result)}


def _handle_store_map(body):
    """
    Explicitly remember the game map at game start so c3 count questions can recall
    it later. The full map is deliberately NOT echoed back (it would waste tokens on
    every game); only the dimensions are returned as a lightweight confirmation.
    """
    game_map = body.get('game_map')

    if not isinstance(game_map, list) or not game_map:
        return _err(400, "Missing 'game_map' for store_map action (expected a non-empty list of rows)")

    _remember_map(game_map)

    rows = len(game_map)
    cols = max(len(row) for row in game_map)
    total_cells = sum(len(row) for row in game_map)
    result = {'success': True, 'rows': rows, 'cols': cols, 'total_cells': total_cells}
    print(f"STORE_MAP: rows={rows} cols={cols} total_cells={total_cells}")
    return {'statusCode': 200, 'body': json.dumps(result)}


def _handle_count(body):
    """Deterministic map-tile counting for c3 (Memento) questions."""
    game_map = body.get('game_map', [])
    question = str(body.get('question', ''))

    # The c3 question never carries the map (it was only shown once at game start).
    # Fall back to the map remembered from that earlier sighting.
    if not game_map:
        game_map = _recall_map() or []

    # Fix jagged rows, same defensive handling as the Pathfinding tool
    if game_map:
        max_cols = max(len(row) for row in game_map)
        game_map = [row + ['normal'] * (max_cols - len(row)) for row in game_map]

    if not game_map:
        return _err(400, 'No game_map available: none passed and none remembered from game start')

    if not question.strip():
        return _err(400, 'Missing question')

    # Tally every cell type on the map, same as the Pathfinding tool's map_summary
    counts = Counter(cell for row in game_map for cell in row)

    # Pull every distinct challenge ID mentioned in the question (dedup, keep stable order)
    seen = []
    for m in ID_PATTERN.finditer(question):
        cid = 'c' + m.group(1)
        if cid not in seen:
            seen.append(cid)

    if not seen:
        return _err(400, f'No challenge IDs (c1, c2, c7, ...) found in question: {question!r}')

    breakdown = {cid: counts.get(cid, 0) for cid in seen}
    total = sum(breakdown.values())

    result = {
        'answer': str(total),
        'breakdown': breakdown,
        # 'map_summary' (a tally of every tile type on the map) is deliberately not
        # returned: the agent only needs the count it asked for, and the full tally
        # costs input tokens on every Memento challenge.
        'dimensions': {'rows': len(game_map), 'cols': len(game_map[0]) if game_map else 0},
        'total_cells': sum(len(row) for row in game_map),
        'question_ids_found': seen,
    }
    print(f"RESULT: question={question!r} ids={seen} breakdown={breakdown} total={total}")
    return {'statusCode': 200, 'body': json.dumps(result)}


def _err(code, msg):
    return {'statusCode': code, 'body': json.dumps({'error': msg})}
