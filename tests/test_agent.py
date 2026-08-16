"""
Comprehensive local test harness for all AWS AI League Lambdas.

Run with:
    python -m pytest tests/         (if pytest installed)
    python -m unittest discover tests/
    python -m unittest tests.test_agent

All test data is inline -- no external fixtures or network access required.
"""

import json
import sys
import os
import unittest

# Ensure the repo root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pathfinding
import codeexecutor
import memoryquestion
import webscraper


# ===========================================================================
# Helper
# ===========================================================================
def _invoke(handler, payload):
    """Invoke a Lambda handler directly and return the parsed body."""
    resp = handler(payload, None)
    body = resp.get("body")
    if isinstance(body, str):
        return json.loads(body)
    return body


# ===========================================================================
# PATHFINDING TESTS
# ===========================================================================
class TestPathfindingSwift(unittest.TestCase):
    """BFS shortest-path (swift strategy)."""

    def test_simple_bfs_to_treasure(self):
        """Basic open corridor: start -> treasure in 4 moves."""
        game_map = [
            ["start", "normal", "normal", "normal", "treasure"],
        ]
        result = _invoke(pathfinding.lambda_handler, {
            "game_map": game_map,
            "strategy": "swift",
        })
        self.assertNotIn("error", result)
        self.assertEqual(result["strategy"], "swift")
        self.assertEqual(result["steps"], 4)
        self.assertTrue(result["verification"]["reaches_treasure"])
        self.assertEqual(result["verification"]["walls_hit"], 0)

    def test_bfs_around_wall(self):
        """BFS must route around a wall."""
        game_map = [
            ["start", "wall", "treasure"],
            ["normal", "normal", "normal"],
        ]
        result = _invoke(pathfinding.lambda_handler, {
            "game_map": game_map,
            "strategy": "swift",
        })
        self.assertTrue(result["verification"]["reaches_treasure"])
        # Must go down, right, right, up = 4 steps
        self.assertEqual(result["steps"], 4)

    def test_no_treasure_error(self):
        """Map with no treasure returns an error."""
        game_map = [["start", "normal", "normal"]]
        result = _invoke(pathfinding.lambda_handler, {
            "game_map": game_map,
            "strategy": "swift",
        })
        self.assertIn("error", result)


class TestPathfindingCoins(unittest.TestCase):
    """get_coins strategy."""

    def test_collects_coins_then_treasure(self):
        game_map = [
            ["start", "c7", "normal"],
            ["normal", "c7", "treasure"],
        ]
        result = _invoke(pathfinding.lambda_handler, {
            "game_map": game_map,
            "strategy": "get_coins",
        })
        self.assertTrue(result["verification"]["reaches_treasure"])
        self.assertGreaterEqual(result["verification"]["coins_collected"], 2)


class TestPathfindingKeyFirst(unittest.TestCase):
    """key_first / optimal strategy -- the main route optimiser."""

    def test_all_coins_collected_no_door_violations(self):
        """Simple map with a key, door, and coins."""
        game_map = [
            ["start", "c7",    "c7",    "normal"],
            ["normal", "c40",  "normal", "c30"],
            ["normal", "normal", "c7",   "treasure"],
        ]
        result = _invoke(pathfinding.lambda_handler, {
            "game_map": game_map,
            "strategy": "key_first",
        })
        v = result["verification"]
        self.assertTrue(v["reaches_treasure"])
        self.assertEqual(v["walls_hit"], 0)
        self.assertEqual(len(v["key_order_violations"]), 0)

    def test_door_bug_routes_around_door(self):
        """
        DOOR BUG scenario: the only DIRECT path from start to key passes
        through the locked door. The optimiser must route AROUND the door
        to get the key first, then come back through the door.

        Map layout:
            start . . .
            wall  D . .
            key   . . T

        D = c30 (door), key = c40 (matching key), T = treasure.
        The short path start->(1,0) is blocked by the door. Must go around.
        """
        game_map = [
            ["start",  "normal", "normal", "normal"],
            ["wall",   "c30",   "normal", "normal"],
            ["c40",    "normal", "normal", "treasure"],
        ]
        result = _invoke(pathfinding.lambda_handler, {
            "game_map": game_map,
            "strategy": "key_first",
        })
        v = result["verification"]
        self.assertTrue(v["reaches_treasure"],
                        "Path must reach the treasure")
        self.assertEqual(len(v["key_order_violations"]), 0,
                         "Must NOT pass through door before picking up key")
        # Key must be collected before door is opened
        self.assertIn("c40", v["keys_collected"])
        self.assertIn("c30", v["doors_opened"])

    def test_spike_avoidance_high_cost(self):
        """High spike_cost forces route around spikes."""
        game_map = [
            ["start",  "c8",     "treasure"],
            ["normal", "normal", "normal"],
        ]
        result = _invoke(pathfinding.lambda_handler, {
            "game_map": game_map,
            "strategy": "key_first",
            "spike_cost": 500,
        })
        v = result["verification"]
        self.assertTrue(v["reaches_treasure"])
        # With high spike cost, should go around (down, right, right, up)
        self.assertEqual(v["spikes_crossed"], 0,
                         "High spike_cost should force route around the spike")

    def test_spike_low_cost_goes_through(self):
        """Low spike_cost allows direct route through spike."""
        game_map = [
            ["start",  "c8",     "treasure"],
            ["normal", "normal", "normal"],
        ]
        result = _invoke(pathfinding.lambda_handler, {
            "game_map": game_map,
            "strategy": "key_first",
            "spike_cost": 1,
        })
        v = result["verification"]
        self.assertTrue(v["reaches_treasure"])
        # With cost=1, going through spike (2 steps) is cheaper than around (4)
        self.assertEqual(v["spikes_crossed"], 1)
        self.assertEqual(result["steps"], 2)

    def test_multiple_key_door_pairs_ordering(self):
        """Two key/door pairs with correct ordering enforced."""
        game_map = [
            ["start",  "c40",   "c41",    "normal"],
            ["normal", "c30",   "c31",    "treasure"],
        ]
        result = _invoke(pathfinding.lambda_handler, {
            "game_map": game_map,
            "strategy": "key_first",
        })
        v = result["verification"]
        self.assertTrue(v["reaches_treasure"])
        self.assertEqual(len(v["key_order_violations"]), 0)
        self.assertEqual(len(v["keys_collected"]), 2)
        self.assertGreaterEqual(len(v["doors_opened"]), 2)


class TestPathfindingEdgeCases(unittest.TestCase):
    """Edge cases for pathfinding."""

    def test_jagged_rows_padded(self):
        """Jagged row lengths should be padded and produce a warning."""
        game_map = [
            ["start", "normal", "treasure"],
            ["normal", "normal"],  # shorter row
        ]
        result = _invoke(pathfinding.lambda_handler, {
            "game_map": game_map,
            "strategy": "swift",
        })
        self.assertTrue(result["verification"]["reaches_treasure"])
        self.assertTrue(any("jagged" in w for w in result.get("warnings", [])))

    def test_start_out_of_bounds_fallback(self):
        """Out-of-bounds start position should trigger fallback."""
        game_map = [
            ["start", "normal", "treasure"],
        ]
        result = _invoke(pathfinding.lambda_handler, {
            "game_map": game_map,
            "start_pos": [99, 99],
            "strategy": "swift",
        })
        # Should still find a path using fallback start
        self.assertTrue(result["verification"]["reaches_treasure"])
        self.assertTrue(any("out of bounds" in w for w in result.get("warnings", [])))

    def test_door_with_no_matching_key_warning(self):
        """A door with no matching key should produce a warning."""
        game_map = [
            ["start", "normal", "c30", "normal", "treasure"],
        ]
        result = _invoke(pathfinding.lambda_handler, {
            "game_map": game_map,
            "strategy": "key_first",
        })
        self.assertTrue(any("no matching key" in w for w in result.get("warnings", [])))

    def test_1x1_map(self):
        """1x1 map that is just the treasure (already there)."""
        game_map = [["treasure"]]
        result = _invoke(pathfinding.lambda_handler, {
            "game_map": game_map,
            "strategy": "swift",
        })
        self.assertTrue(result["verification"]["reaches_treasure"])
        self.assertEqual(result["steps"], 0)

    def test_walled_off_treasure(self):
        """Treasure completely surrounded by walls - path cannot reach it."""
        game_map = [
            ["start",  "normal", "normal"],
            ["normal", "wall",   "wall"],
            ["normal", "wall",   "treasure"],
        ]
        result = _invoke(pathfinding.lambda_handler, {
            "game_map": game_map,
            "strategy": "swift",
        })
        self.assertFalse(result["verification"]["reaches_treasure"])

    def test_dimension_mismatch_warning(self):
        """expected_rows/cols mismatch triggers a warning."""
        game_map = [
            ["start", "treasure"],
            ["normal", "normal"],
        ]
        result = _invoke(pathfinding.lambda_handler, {
            "game_map": game_map,
            "strategy": "swift",
            "expected_rows": 5,
            "expected_cols": 10,
        })
        warnings = result.get("warnings", [])
        self.assertTrue(any("expected_rows" in w for w in warnings))
        self.assertTrue(any("expected_cols" in w for w in warnings))


# ===========================================================================
# CODE EXECUTOR TESTS
# ===========================================================================
class TestCodeExecutorAgentCode(unittest.TestCase):
    """Agent-authored code path."""

    def test_simple_print(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "What is 2+2?",
            "code": "print(2 + 2)\n",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "4")
        self.assertEqual(result["source"], "code")

    def test_large_factorial(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "What is 100!?",
            "code": "import math\nprint(math.factorial(100))\n",
        })
        self.assertTrue(result["success"])
        # 100! has 158 digits
        self.assertEqual(len(result["answer"]), 158)

    def test_deep_recursion(self):
        """Recursion limit is raised so deep recursion works."""
        code = (
            "import sys\n"
            "sys.setrecursionlimit(10000)\n"
            "def fib(n, memo={}):\n"
            "    if n <= 1: return n\n"
            "    if n not in memo:\n"
            "        memo[n] = fib(n-1) + fib(n-2)\n"
            "    return memo[n]\n"
            "print(fib(5000))\n"
        )
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "5000th fibonacci?",
            "code": code,
        })
        self.assertTrue(result["success"])
        self.assertTrue(len(result["answer"]) > 100)


class TestCodeExecutorSandbox(unittest.TestCase):
    """Sandbox enforcement."""

    def test_os_blocked(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "test",
            "code": "import os\nprint(os.listdir('.'))\n",
        })
        self.assertFalse(result["success"])

    def test_subprocess_blocked(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "test",
            "code": "import subprocess\nprint(subprocess.run(['ls']))\n",
        })
        self.assertFalse(result["success"])

    def test_socket_blocked(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "test",
            "code": "import socket\nprint(socket.gethostname())\n",
        })
        self.assertFalse(result["success"])

    def test_urllib_blocked(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "test",
            "code": "import urllib.request\nprint(urllib.request.urlopen('http://example.com').read())\n",
        })
        self.assertFalse(result["success"])

    def test_open_blocked(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "test",
            "code": "f = open('/etc/passwd')\nprint(f.read())\n",
        })
        self.assertFalse(result["success"])

    def test_dunder_import_blocked(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "test",
            "code": "m = __import__('os')\nprint(m.listdir('.'))\n",
        })
        self.assertFalse(result["success"])

    def test_no_print_rejected(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "test",
            "code": "x = 42\n",
        })
        self.assertFalse(result["success"])

    def test_syntax_error_rejected(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "test",
            "code": "def foo(\nprint('hi')\n",
        })
        self.assertFalse(result["success"])

    def test_timeout_on_infinite_loop(self):
        """Infinite loop should hit timeout."""
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "test",
            "code": "while True: pass\nprint('done')\n",
        })
        self.assertFalse(result["success"])
        self.assertIn("timed out", result.get("error", "") or
                      str(result.get("attempts", "")))

    def test_runtime_error_surfaced(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "test",
            "code": "print(1/0)\n",
        })
        self.assertFalse(result["success"])
        # Error should mention ZeroDivisionError, not be swallowed
        error_text = str(result.get("attempts", ""))
        self.assertTrue(
            "ZeroDivisionError" in error_text or "division" in error_text,
            f"Expected ZeroDivisionError in: {error_text}"
        )


class TestCodeExecutorPatternLibrary(unittest.TestCase):
    """Pattern library fallback (question only, no code supplied)."""

    def test_fibonacci(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "What is the 10th fibonacci number?",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "55")

    def test_fibonacci_last_digits(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "What are the last 4 digits of the 1000th fibonacci number?",
        })
        self.assertTrue(result["success"])
        self.assertEqual(len(result["answer"]), 4)

    def test_factorial_trailing_zeros(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "How many trailing zeros in 100!?",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "24")

    def test_factorial_digit_sum(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "What is the digit sum of 10! factorial?",
        })
        self.assertTrue(result["success"])
        # 10! = 3628800, digit sum = 27
        self.assertEqual(result["answer"], "27")

    def test_factorial_modulo(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "What is 20 factorial modulo 1000000007?",
        })
        self.assertTrue(result["success"])

    def test_nth_prime(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "What is the 100th prime number?",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "541")

    def test_sum_primes_below(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "What is the sum of all primes below 100?",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "1060")

    def test_largest_prime_factor(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "What is the largest prime factor of 600851475143?",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "6857")

    def test_is_prime(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "Is 97 a prime number?",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "yes")

    def test_collatz(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "How many steps in the Collatz sequence starting at 27?",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "112")

    def test_gcd(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "What is the GCD of 48 and 18?",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "6")

    def test_lcm(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "What is the LCM of 12 and 18?",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "36")

    def test_pow_mod(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "What is 2^100 mod 1000000007?",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], str(pow(2, 100, 1000000007)))

    def test_palindrome_check(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "Is 12321 a palindrome?",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "yes")

    def test_divisors_sum(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "What is the sum of all divisors of 28?",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "56")

    def test_binomial(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "What is 10 choose 3 (binomial coefficient)?",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "120")

    def test_series_sum_squares(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "What is the sum of squares of the first 10 natural numbers?",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "385")

    def test_digit_sum(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "What is the digit sum of 123456789?",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "45")

    def test_base_conversion_binary(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "What is 255 in binary?",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "11111111")


class TestCodeExecutorArithmetic(unittest.TestCase):
    """Safe arithmetic evaluation."""

    def test_basic_expression(self):
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "What is 123 + 456 * 7?",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], str(123 + 456 * 7))

    def test_dangerous_expression_rejected(self):
        """Expressions with function calls are rejected by safe_arithmetic."""
        val, ok = codeexecutor.safe_arithmetic("__import__('os').system('ls')")
        self.assertFalse(ok)

    def test_huge_exponent_rejected(self):
        val, ok = codeexecutor.safe_arithmetic("2 ** 99999999999")
        self.assertFalse(ok)


class TestCodeExecutorNumberWords(unittest.TestCase):
    """Number word parsing."""

    def test_three_thousandth(self):
        result = codeexecutor.words_to_int("three thousandth")
        self.assertEqual(result, 3000)

    def test_twenty_first(self):
        result = codeexecutor.words_to_int("twenty first")
        self.assertEqual(result, 21)

    def test_one_hundred(self):
        result = codeexecutor.words_to_int("one hundred")
        self.assertEqual(result, 100)

    def test_fifty_second(self):
        result = codeexecutor.words_to_int("fifty second")
        self.assertEqual(result, 52)

    def test_invalid_returns_none(self):
        result = codeexecutor.words_to_int("hello world")
        self.assertIsNone(result)


class TestCodeExecutorTruncation(unittest.TestCase):
    """Silent truncation is now a failure, not a wrong answer."""

    def test_truncation_is_failure(self):
        """If output exceeds MAX_OUTPUT_LENGTH, it must be a hard failure."""
        # Generate code that prints way more than MAX_OUTPUT_LENGTH
        code = "print('x' * 500000)\n"
        result = _invoke(codeexecutor.lambda_handler, {
            "question": "test",
            "code": code,
        })
        self.assertFalse(result["success"])


# ===========================================================================
# MEMORY QUESTION TESTS
# ===========================================================================
class TestMemoryQuestion(unittest.TestCase):
    """Deterministic map counter for c3 (Memento) challenges."""

    def test_single_challenge_count(self):
        game_map = [
            ["c7", "c7", "c7", "normal"],
            ["c1", "c7", "normal", "treasure"],
        ]
        result = _invoke(memoryquestion.lambda_handler, {
            "game_map": game_map,
            "question": "How many c7 challenges are on the map?",
        })
        self.assertEqual(result["answer"], "4")
        self.assertEqual(result["breakdown"]["c7"], 4)

    def test_combined_count(self):
        game_map = [
            ["c1", "c2", "c1", "normal"],
            ["c2", "normal", "c2", "treasure"],
        ]
        result = _invoke(memoryquestion.lambda_handler, {
            "game_map": game_map,
            "question": "What is c1 + c2?",
        })
        # c1=2, c2=3, total=5
        self.assertEqual(result["answer"], "5")
        self.assertEqual(result["breakdown"]["c1"], 2)
        self.assertEqual(result["breakdown"]["c2"], 3)

    def test_diagnostic_fields_present(self):
        game_map = [
            ["c7", "normal", "start"],
            ["normal", "c3", "treasure"],
        ]
        result = _invoke(memoryquestion.lambda_handler, {
            "game_map": game_map,
            "question": "How many c7?",
        })
        self.assertIn("dimensions", result)
        self.assertIn("total_cells", result)
        self.assertEqual(result["dimensions"]["rows"], 2)
        self.assertEqual(result["dimensions"]["cols"], 3)
        self.assertEqual(result["total_cells"], 6)

    def test_no_ids_in_question_error(self):
        game_map = [["normal", "treasure"]]
        result = _invoke(memoryquestion.lambda_handler, {
            "game_map": game_map,
            "question": "How many challenges?",
        })
        self.assertIn("error", result)

    def test_missing_game_map_error(self):
        result = _invoke(memoryquestion.lambda_handler, {
            "game_map": [],
            "question": "How many c1?",
        })
        self.assertIn("error", result)

    def test_jagged_rows_handled(self):
        """Jagged rows in map should be padded."""
        game_map = [
            ["c7", "c7", "c7"],
            ["c7"],  # short row
        ]
        result = _invoke(memoryquestion.lambda_handler, {
            "game_map": game_map,
            "question": "How many c7?",
        })
        self.assertEqual(result["answer"], "4")


# ===========================================================================
# MEMORY STORE/RETRIEVE TESTS
# ===========================================================================
class TestMemoryStoreRetrieve(unittest.TestCase):
    """Store and retrieve key-value pairs via the memory tool."""

    def setUp(self):
        """Clear the memory store before each test to avoid cross-test contamination."""
        memoryquestion._memory_store.clear()

    def test_store_returns_success(self):
        """Storing a key-value pair returns success with key and value echoed back."""
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "store",
            "key": "door_key_c32",
            "value": "AWSisAwesome",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["key"], "door_key_c32")
        self.assertEqual(result["value"], "AWSisAwesome")

    def test_retrieve_existing_key(self):
        """Retrieving a previously stored key returns the correct value."""
        memoryquestion._memory_store["door_key_c33"] = "PartyOnMyFriend"
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "retrieve",
            "key": "door_key_c33",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["key"], "door_key_c33")
        self.assertEqual(result["value"], "PartyOnMyFriend")

    def test_retrieve_nonexistent_key(self):
        """Retrieving a key that was never stored returns an error."""
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "retrieve",
            "key": "nonexistent",
        })
        self.assertFalse(result["success"])
        self.assertEqual(result["key"], "nonexistent")
        self.assertIn("Key not found", result["error"])

    def test_store_then_retrieve_round_trip_c32(self):
        """Full round-trip: store door_key_c32 then retrieve it."""
        _invoke(memoryquestion.lambda_handler, {
            "action": "store",
            "key": "door_key_c32",
            "value": "AWSisAwesome",
        })
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "retrieve",
            "key": "door_key_c32",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["value"], "AWSisAwesome")

    def test_store_then_retrieve_round_trip_c33(self):
        """Full round-trip: store door_key_c33 then retrieve it."""
        _invoke(memoryquestion.lambda_handler, {
            "action": "store",
            "key": "door_key_c33",
            "value": "PartyOnMyFriend",
        })
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "retrieve",
            "key": "door_key_c33",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["value"], "PartyOnMyFriend")

    def test_store_overwrites_existing_key(self):
        """Storing the same key again overwrites the previous value."""
        _invoke(memoryquestion.lambda_handler, {
            "action": "store",
            "key": "door_key_c32",
            "value": "OldValue",
        })
        _invoke(memoryquestion.lambda_handler, {
            "action": "store",
            "key": "door_key_c32",
            "value": "NewValue",
        })
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "retrieve",
            "key": "door_key_c32",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["value"], "NewValue")

    def test_backward_compat_count_without_action(self):
        """Calling with game_map + question and no action still works as count."""
        game_map = [
            ["c7", "c7", "c7", "normal"],
            ["c1", "c7", "normal", "treasure"],
        ]
        result = _invoke(memoryquestion.lambda_handler, {
            "game_map": game_map,
            "question": "How many c7 challenges are on the map?",
        })
        self.assertEqual(result["answer"], "4")
        self.assertEqual(result["breakdown"]["c7"], 4)

    def test_store_via_api_gateway_body(self):
        """Store action works when event has API Gateway style 'body' key."""
        import json as _json
        event = {
            "body": _json.dumps({
                "action": "store",
                "key": "test_key",
                "value": "test_value",
            })
        }
        result = _invoke(memoryquestion.lambda_handler, event)
        self.assertTrue(result["success"])
        self.assertEqual(result["value"], "test_value")

    def test_missing_key_for_store_error(self):
        """Store without a 'key' field returns an error."""
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "store",
            "value": "some_value",
        })
        self.assertIn("error", result)

    def test_missing_value_for_store_error(self):
        """Store without a 'value' field returns an error."""
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "store",
            "key": "some_key",
        })
        self.assertIn("error", result)

    def test_missing_key_for_retrieve_error(self):
        """Retrieve without a 'key' field returns an error."""
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "retrieve",
        })
        self.assertIn("error", result)


# ===========================================================================
# MEMORY TRANSFORM TESTS (door unlock character arithmetic)
# ===========================================================================
class TestMemoryTransform(unittest.TestCase):
    """
    Deterministic key transformations for door unlocks.

    The agent must never do character-position arithmetic itself, so these tests pin
    the exact expected output of each rule -- including the 1-indexed off-by-one that
    previously killed the run at the yellow door.
    """

    def setUp(self):
        """Clear the memory store before each test to avoid cross-test contamination."""
        memoryquestion._memory_store.clear()

    def test_transform_char5char7_yellow_door(self):
        """char5char7 is 1-indexed: 'PartyOnMyFriend' -> 'yn' (not the 0-indexed 'OM')."""
        _invoke(memoryquestion.lambda_handler, {
            "action": "store",
            "key": "door_key_c33",
            "value": "PartyOnMyFriend",
        })
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "transform",
            "key": "door_key_c33",
            "rule": "char5char7",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "yn")
        self.assertEqual(result["key"], "door_key_c33")
        self.assertEqual(result["rule"], "char5char7")

    def test_transform_char5char7_is_not_zero_indexed(self):
        """Regression guard: the 0-indexed answer 'OM' must never be returned."""
        _invoke(memoryquestion.lambda_handler, {
            "action": "store",
            "key": "door_key_c33",
            "value": "PartyOnMyFriend",
        })
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "transform",
            "key": "door_key_c33",
            "rule": "char5char7",
        })
        self.assertNotEqual(result["answer"], "OM")

    def test_transform_first2last2_grey_door(self):
        """first2last2 on 'AWSisAwesome' -> 'AWme'."""
        _invoke(memoryquestion.lambda_handler, {
            "action": "store",
            "key": "door_key_c32",
            "value": "AWSisAwesome",
        })
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "transform",
            "key": "door_key_c32",
            "rule": "first2last2",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "AWme")
        self.assertEqual(result["key"], "door_key_c32")
        self.assertEqual(result["rule"], "first2last2")

    def test_transform_generic_placeholder_values(self):
        """Both rules match their documented placeholder examples."""
        memoryquestion._memory_store["k1"] = "ABCDEFGH"
        memoryquestion._memory_store["k2"] = "ABCDEF"
        c5c7 = _invoke(memoryquestion.lambda_handler, {
            "action": "transform", "key": "k1", "rule": "char5char7",
        })
        f2l2 = _invoke(memoryquestion.lambda_handler, {
            "action": "transform", "key": "k2", "rule": "first2last2",
        })
        self.assertEqual(c5c7["answer"], "EG")
        self.assertEqual(f2l2["answer"], "ABEF")

    def test_transform_nonexistent_key_reports_not_found(self):
        """Transforming a key that was never stored reports not-found, not a crash."""
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "transform",
            "key": "door_key_c33",
            "rule": "char5char7",
        })
        self.assertFalse(result["success"])
        self.assertEqual(result["key"], "door_key_c33")
        self.assertIn("Key not found", result["error"])
        self.assertNotIn("answer", result)

    def test_transform_missing_rule_error(self):
        """Transform without a 'rule' field returns an error listing supported rules."""
        memoryquestion._memory_store["door_key_c33"] = "PartyOnMyFriend"
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "transform",
            "key": "door_key_c33",
        })
        self.assertIn("error", result)
        self.assertIn("first2last2", result["error"])
        self.assertIn("char5char7", result["error"])

    def test_transform_unknown_rule_error(self):
        """Transform with an unrecognized rule returns an error listing supported rules."""
        memoryquestion._memory_store["door_key_c33"] = "PartyOnMyFriend"
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "transform",
            "key": "door_key_c33",
            "rule": "char3char9",
        })
        self.assertIn("error", result)
        self.assertIn("first2last2", result["error"])
        self.assertIn("char5char7", result["error"])

    def test_transform_missing_key_field_error(self):
        """Transform without a 'key' field returns an error."""
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "transform",
            "rule": "char5char7",
        })
        self.assertIn("error", result)

    def test_transform_value_too_short_for_char5char7(self):
        """A value shorter than 7 chars errors instead of producing a wrong answer."""
        memoryquestion._memory_store["short_key"] = "ABCDEF"
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "transform",
            "key": "short_key",
            "rule": "char5char7",
        })
        self.assertIn("error", result)
        self.assertIn("too short", result["error"])
        self.assertNotIn("answer", result)

    def test_transform_value_too_short_for_first2last2(self):
        """A value shorter than 4 chars errors instead of returning a truncated answer."""
        memoryquestion._memory_store["short_key"] = "ABC"
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "transform",
            "key": "short_key",
            "rule": "first2last2",
        })
        self.assertIn("error", result)
        self.assertIn("too short", result["error"])
        self.assertNotIn("answer", result)

    def test_transform_exact_minimum_lengths_succeed(self):
        """Boundary check: values at exactly the minimum length transform fine."""
        memoryquestion._memory_store["k7"] = "ABCDEFG"
        memoryquestion._memory_store["k4"] = "ABCD"
        c5c7 = _invoke(memoryquestion.lambda_handler, {
            "action": "transform", "key": "k7", "rule": "char5char7",
        })
        f2l2 = _invoke(memoryquestion.lambda_handler, {
            "action": "transform", "key": "k4", "rule": "first2last2",
        })
        self.assertEqual(c5c7["answer"], "EG")
        self.assertEqual(f2l2["answer"], "ABCD")

    def test_transform_success_does_not_leak_raw_value(self):
        """The success response must not include the raw stored key value."""
        raw = "PartyOnMyFriend"
        _invoke(memoryquestion.lambda_handler, {
            "action": "store",
            "key": "door_key_c33",
            "value": raw,
        })
        result = _invoke(memoryquestion.lambda_handler, {
            "action": "transform",
            "key": "door_key_c33",
            "rule": "char5char7",
        })
        self.assertTrue(result["success"])
        self.assertNotIn("value", result)
        self.assertNotIn(raw, json.dumps(result))

    def test_transform_does_not_consume_stored_key(self):
        """Transforming leaves the stored value in place for repeat attempts."""
        _invoke(memoryquestion.lambda_handler, {
            "action": "store",
            "key": "door_key_c33",
            "value": "PartyOnMyFriend",
        })
        first = _invoke(memoryquestion.lambda_handler, {
            "action": "transform", "key": "door_key_c33", "rule": "char5char7",
        })
        second = _invoke(memoryquestion.lambda_handler, {
            "action": "transform", "key": "door_key_c33", "rule": "char5char7",
        })
        self.assertEqual(first["answer"], "yn")
        self.assertEqual(second["answer"], "yn")

    def test_transform_via_api_gateway_body(self):
        """Transform works when the event has an API Gateway style 'body' key."""
        memoryquestion._memory_store["door_key_c33"] = "PartyOnMyFriend"
        result = _invoke(memoryquestion.lambda_handler, {
            "body": json.dumps({
                "action": "transform",
                "key": "door_key_c33",
                "rule": "char5char7",
            })
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "yn")


# ===========================================================================
# WEBSCRAPER TESTS (no network -- input parsing and error handling only)
# ===========================================================================
class TestWebscraperInputParsing(unittest.TestCase):
    """Verify the handler parses input correctly and returns expected structure."""

    def test_no_question_error(self):
        result = _invoke(webscraper.lambda_handler, {})
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_no_url_in_question(self):
        result = _invoke(webscraper.lambda_handler, {
            "question": "What is the meaning of life?",
        })
        self.assertFalse(result["success"])
        self.assertIn("No valid URL", result["error"])

    def test_url_extraction_from_question(self):
        """extract_url should pull a valid URL from mixed text."""
        url = webscraper.extract_url(
            "Please visit https://example.com/page?q=1 and tell me what you find."
        )
        self.assertEqual(url, "https://example.com/page?q=1")

    def test_url_extraction_no_url(self):
        url = webscraper.extract_url("No URL here at all")
        self.assertIsNone(url)

    def test_api_gateway_body_parsing(self):
        """Handler parses API Gateway style events (body as JSON string)."""
        event = {
            "body": json.dumps({"question": "Check https://example.com"})
        }
        # This will fail on network but should at least parse correctly
        result = _invoke(webscraper.lambda_handler, event)
        # Should have attempted to fetch (will fail with network error)
        self.assertIn("url", result)

    def test_content_extractor_basic(self):
        """ContentExtractor extracts text from simple HTML."""
        html = "<html><head><title>Test Page</title></head><body><p>Hello World</p></body></html>"
        extractor = webscraper.ContentExtractor()
        extractor.feed(html)
        self.assertEqual(extractor.title, "Test Page")
        self.assertIn("Hello World", extractor.get_text())


# ===========================================================================
# Run
# ===========================================================================
if __name__ == "__main__":
    unittest.main()



# ===========================================================================
# WEBSCRAPER RELEVANCE TESTS (numeric-anchor precision)
# ===========================================================================
class TestWebscraperNumericAnchors(unittest.TestCase):
    """
    Guards the fix for a flaky web-search challenge that cost 1050 points a run.

    The question "...outperformed by 20-50%?" was answered correctly in some runs and
    wrongly in others with identical code. Cause: the keyword regex was [a-z]{2,},
    which discarded every digit, so "20-50%" -- the one token that uniquely identifies
    the answer sentence -- never influenced ranking. The page was then returned whole,
    and a more familiar but wrong entity elsewhere on it competed for the model's
    attention.
    """

    ANCHORED_Q = (
        "According to https://example.com/forge/ for Acme Labs, through supervised "
        "fine-tuning with Model X Lite, what model was outperformed by 20-50%?"
    )
    DISTRACTOR = "Many customers compare results against Rival 3.5 Ultra when evaluating a model."
    ANSWER = "For Acme Labs, fine-tuning with Model X Lite outperformed Contender 4 by 20-50%."

    def test_numeric_anchor_extracted_from_question(self):
        """The digits in the question must reach the ranker at all."""
        out = webscraper.extract_relevant_content(
            " ".join([self.DISTRACTOR, self.ANSWER]), self.ANCHORED_Q
        )
        self.assertIn("Contender 4", out)

    def test_anchor_isolates_answer_on_short_page(self):
        """Page fits the budget: the answer is kept and the distractor dropped."""
        page = " ".join(
            ["Intro sentence about the platform.", self.DISTRACTOR]
            + [f"Section {i} describes capabilities." for i in range(12)]
            + [self.ANSWER]
        )
        out = webscraper.extract_relevant_content(page, self.ANCHORED_Q)
        self.assertIn("Contender 4", out)
        self.assertNotIn("Rival 3.5 Ultra", out)

    def test_anchor_isolates_answer_on_long_page(self):
        """Page exceeds the budget: same guarantee via the ranking path."""
        page = " ".join(
            [self.DISTRACTOR]
            + [f"Long filler sentence {i} padding well past the budget." for i in range(120)]
            + [self.ANSWER]
        )
        self.assertGreater(len(page), webscraper.MAX_RESPONSE_LENGTH)
        out = webscraper.extract_relevant_content(page, self.ANCHORED_Q)
        self.assertIn("Contender 4", out)
        self.assertNotIn("Rival 3.5 Ultra", out)

    def test_question_without_digits_returns_page_unchanged(self):
        """
        No numeric anchor means no behaviour change at all. Pins the known-good path
        for the web question that already passes, so this fix cannot regress it.
        """
        q = "According to https://example.com/ what company gave officers freedom to experiment?"
        page = " ".join(
            [f"Filler {i} about the programme." for i in range(20)]
            + ["Globex Industries gave officers the freedom to experiment with AI tools."]
        )
        out = webscraper.extract_relevant_content(page, q)
        self.assertIn("Globex Industries", out)
        self.assertEqual(out, page[:webscraper.MAX_RESPONSE_LENGTH])

    def test_context_limits_not_silently_reduced(self):
        """
        Trimming what the scraper returns was tried and cost 1056 points. These are the
        values from the configuration that scored 13787; failing here means someone has
        starved the scraper again.
        """
        self.assertEqual(webscraper.MAX_RESPONSE_LENGTH, 4000)
        self.assertEqual(webscraper.TOP_SENTENCES, 20)
