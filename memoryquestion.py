import json
import re
from collections import Counter

# Matches any challenge-tile token mentioned in the question text, e.g. c1, c2, c7, c30, c40
ID_PATTERN = re.compile(r'\bc(\d+)\b', re.IGNORECASE)


def lambda_handler(event, context):
    """
    Deterministic answer tool for c3 (Memento) map-memory questions.

    Instead of asking the agent to recall/recount the map itself (error-prone -
    manual recounts have been wrong 3+ different ways across test runs), this
    tool:
      1. Takes the CURRENT round's game_map and the literal question text.
      2. Extracts every challenge ID mentioned in the question (e.g. "c1", "c2",
         "c7") via regex - works for single-type questions ("how many c7
         challenges") and combined ones ("c1 + c2", "c1 and c5").
      3. Counts each ID's real occurrences directly in the map with code - no
         LLM arithmetic, no chance of a miscount.
      4. Sums them and returns a single, ready-to-answer string.

    Input:
      {
        "game_map": [[...]],        # the full current-round map
        "question": "c1 + c2"       # the literal Memento question text
      }

    Output:
      {
        "answer": "3",                     # ready to send back as the final answer
        "breakdown": {"c1": 1, "c2": 2},   # per-ID counts, for debugging
        "map_summary": {...}               # full tile-type tally of the map
      }
    """
    try:
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event

        print(f"DEBUG: Received event: {body}")

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
        }
        print(f"RESULT: question={question!r} ids={seen} breakdown={breakdown} total={total}")
        return {'statusCode': 200, 'body': json.dumps(result)}

    except Exception as e:
        print(f"ERROR: {e}")
        return _err(500, str(e))


def _err(code, msg):
    return {'statusCode': code, 'body': json.dumps({'error': msg})}