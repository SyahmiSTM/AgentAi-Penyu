import json
import re
from collections import Counter

# Matches any challenge-tile token mentioned in the question text, e.g. c1, c2, c7, c30, c40
ID_PATTERN = re.compile(r'\bc(\d+)\b', re.IGNORECASE)

# Module-level key-value store for door keys and other data.
# Persists across invocations while the Lambda container stays warm (i.e. during a game session).
_memory_store = {}


def lambda_handler(event, context):
    """
    Multi-action memory tool supporting:
      - "count"    : Deterministic map-tile counting for c3 (Memento) questions.
      - "store"    : Store a key-value pair in memory (e.g. door keys).
      - "retrieve" : Retrieve a previously stored value by key.

    Action routing:
      - If 'action' field is present, dispatch to that action.
      - If no 'action' but 'game_map' and 'question' are present, default to 'count' (backward compat).

    -- count --
    Input:
      { "action": "count", "game_map": [[...]], "question": "c1 + c2" }
    Output:
      { "answer": "3", "breakdown": {"c1": 1, "c2": 2}, ... }

    -- store --
    Input:
      { "action": "store", "key": "door_key_c33", "value": "PartyOnMyFriend" }
    Output:
      { "success": true, "key": "door_key_c33", "value": "PartyOnMyFriend" }

    -- retrieve --
    Input:
      { "action": "retrieve", "key": "door_key_c33" }
    Output (found):
      { "success": true, "key": "door_key_c33", "value": "PartyOnMyFriend" }
    Output (not found):
      { "success": false, "key": "door_key_c33", "error": "Key not found: door_key_c33" }
    """
    try:
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event

        print(f"DEBUG: Received event: {body}")

        # Determine action
        action = body.get('action')

        # Backward compatibility: no action but game_map + question present -> count
        if action is None:
            if body.get('game_map') and body.get('question'):
                action = 'count'
            else:
                return _err(400, "Missing 'action' field. Supported actions: count, store, retrieve")

        if action == 'store':
            return _handle_store(body)
        elif action == 'retrieve':
            return _handle_retrieve(body)
        elif action == 'count':
            return _handle_count(body)
        else:
            return _err(400, f"Unknown action: {action!r}. Supported: count, store, retrieve")

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


def _handle_count(body):
    """Deterministic map-tile counting for c3 (Memento) questions."""
    game_map = body.get('game_map', [])
    question = str(body.get('question', ''))

    # Fix jagged rows, same defensive handling as the Pathfinding tool
    if game_map:
        max_cols = max(len(row) for row in game_map)
        game_map = [row + ['normal'] * (max_cols - len(row)) for row in game_map]

    if not game_map:
        return _err(400, 'Missing game_map')

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
        'map_summary': dict(counts),
        'dimensions': {'rows': len(game_map), 'cols': len(game_map[0]) if game_map else 0},
        'total_cells': sum(len(row) for row in game_map),
        'question_ids_found': seen,
    }
    print(f"RESULT: question={question!r} ids={seen} breakdown={breakdown} total={total}")
    return {'statusCode': 200, 'body': json.dumps(result)}


def _err(code, msg):
    return {'statusCode': code, 'body': json.dumps({'error': msg})}
