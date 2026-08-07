"""
AWS AI League - Code Execution Lambda for c2 (Blue Brain) and c6 (Boss) challenges.
Handler: codeexecutor.lambda_handler

## Contract

Preferred input (the CodeExecutor sub-agent writes the Python itself):
    {"question": "<original question>", "code": "<python that prints the answer>"}

Fallback input (no code supplied):
    {"question": "<original question>"}

Resolution order:
  1. If "code" is supplied -> validate + execute it. This is the general path and
     handles ANY computable question, not just pre-known patterns.
  2. If that fails (or no code was supplied) -> try the built-in pattern library
     for well-known problem families (Fibonacci, primes, factorials, ...).
  3. If that fails -> try a safe AST arithmetic evaluation of any expression in
     the question text.
  4. Otherwise return success=False with "needs_code": True and the stderr, so
     the sub-agent can repair its code and retry once.

## Why "code" is the primary path
The previous version regex-matched the question against a fixed list of problem
types and fell back to a raw eval() guess. Anything outside that list failed.
A Boss Challenge (c6) is explicitly "may require multiple skills combined", so a
fixed pattern list cannot cover it. The Lambda now runs arbitrary Python instead
of trying to recognise the problem.

## Compliance note
This Lambda calls NO model and contains NO hardcoded challenge answers. The code
it executes is authored by the CodeExecutor sub-agent that is already part of the
declared agent architecture. Nothing external is invoked from inside the tool.

## Sandboxing
Executed code runs in a separate short-lived process with:
  - an import allowlist (no os/subprocess/socket/urllib/...)
  - a source scan for dangerous tokens, including ones hidden in strings
  - RLIMIT_CPU, RLIMIT_AS (address space) and RLIMIT_FSIZE=0 (no file writes)
  - a wall-clock timeout and a capped output size
This is damage-limitation against a mistaken code generation, not a defence
against a determined attacker - the code author is our own sub-agent.
"""

import ast
import json
import re
import subprocess
import sys

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX
    resource = None

EXEC_TIMEOUT_SECONDS = 12
CPU_LIMIT_SECONDS = 14
MEMORY_LIMIT_BYTES = 1024 * 1024 * 1024
MAX_CODE_LENGTH = 40000
# These must be generous: a legitimate answer can be very long (2024! is 5815
# digits). Truncating a number silently yields a wrong-but-plausible answer,
# which costs a life - so truncation is treated as a hard failure below rather
# than being quietly returned.
MAX_OUTPUT_LENGTH = 400000
MAX_ANSWER_LENGTH = 200000

# Modules the generated code is allowed to import. Anything absent is rejected.
ALLOWED_IMPORTS = frozenset({
    "abc", "array", "base64", "binascii", "bisect", "calendar", "cmath",
    "collections", "copy", "dataclasses", "datetime", "decimal", "enum",
    "fractions", "functools", "hashlib", "heapq", "itertools", "json", "math",
    "numbers", "operator", "queue", "random", "re", "statistics", "string",
    "struct", "sys", "textwrap", "time", "types", "typing", "unicodedata",
    "uuid", "zlib",
})

# Tokens that must not appear anywhere in the source, including inside string
# literals - this catches eval("__import__('os')") style bypasses of the AST check.
BANNED_TOKENS = (
    "__import__", "importlib", "subprocess", "os.system", "os.popen",
    "os.environ", "socket", "urllib", "requests", "httplib", "http.client",
    "ftplib", "smtplib", "telnetlib", "shutil", "pathlib", "tempfile",
    "ctypes", "multiprocessing", "pickle", "marshal", "builtins",
    "globals()", "locals()", "vars()", "open(", "input(", "breakpoint(",
    "exit(", "quit(", "setattr(", "delattr(",
)

BANNED_CALL_NAMES = frozenset({
    "__import__", "open", "input", "breakpoint", "exit", "quit",
    "globals", "locals", "vars", "setattr", "delattr", "compile",
})

# Injected before every executed program.
#   set_int_max_str_digits: Python 3.11+ refuses to str() an int over 4300 digits,
#     which silently breaks e.g. factorial(2024) or 3000! - both plausible c2 asks.
#   setrecursionlimit: recursive DP/memoised solutions are common in generated code.
PRELUDE = "\n".join([
    "import sys",
    "try:",
    "    sys.set_int_max_str_digits(2000000)",
    "except Exception:",
    "    pass",
    "try:",
    "    sys.setrecursionlimit(30000)",
    "except Exception:",
    "    pass",
    "",
])


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
def lambda_handler(event, context):
    try:
        body = _parse_body(event)
        if body is None:
            return _err(400, "Body must be a JSON object")

        question = _first_str(body, ("question", "prompt", "message", "challenge", "text"))
        code = _first_str(body, ("code", "python", "script", "solution"))

        if not question and not code:
            return _err(400, "Provide at least one of: question, code")

        attempts = []
        code_error = None

        # 1. Agent-authored code - the general path.
        if code:
            answer, ok, error = _try_code(code)
            attempts.append({"source": "code", "ok": ok, "error": error})
            code_error = error
            if ok:
                return _ok({
                    "answer": answer,
                    "success": True,
                    "source": "code",
                    "attempts": attempts,
                })

        # 2. Built-in pattern library for known problem families.
        if question:
            pattern_code = generate_solver_code(question)
            if pattern_code:
                answer, ok, error = _try_code(pattern_code, validate=False)
                attempts.append({"source": "pattern", "ok": ok, "error": error})
                if ok:
                    return _ok({
                        "answer": answer,
                        "success": True,
                        "source": "pattern",
                        "code": pattern_code,
                        "attempts": attempts,
                    })

            # 3. Safe arithmetic evaluation of an expression in the question.
            answer, ok = arithmetic_from_question(question)
            attempts.append({"source": "arithmetic", "ok": ok, "error": None if ok else "no evaluable expression"})
            if ok:
                return _ok({
                    "answer": answer,
                    "success": True,
                    "source": "arithmetic",
                    "attempts": attempts,
                })

        # 4. Give up, but tell the caller how to help.
        # Surface the error from the agent's own code in preference to the
        # fallbacks' errors - that is the one the sub-agent can actually act on.
        last_error = code_error or next(
            (a["error"] for a in reversed(attempts) if a.get("error")),
            "no strategy produced an answer",
        )
        return _ok({
            "answer": None,
            "success": False,
            "needs_code": True,
            "error": last_error,
            "hint": (
                "Write a self-contained Python program that prints ONLY the answer "
                "on the last line, and resend as {\"question\": ..., \"code\": ...}. "
                "Allowed imports: " + ", ".join(sorted(ALLOWED_IMPORTS)) + "."
            ),
            "attempts": attempts,
        })

    except Exception as exc:  # pragma: no cover
        return _err(500, "{}: {}".format(type(exc).__name__, exc))


def _parse_body(event):
    if isinstance(event, dict) and "body" in event:
        body = event["body"]
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                return {"question": body}
    else:
        body = event
    return body if isinstance(body, dict) else None


def _first_str(body, keys):
    for key in keys:
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _ok(payload):
    print("RESULT: success={} source={} answer={!r}".format(
        payload.get("success"), payload.get("source"), payload.get("answer")))
    return {"statusCode": 200, "body": json.dumps(payload)}


def _err(code, msg):
    print("ERROR: {}".format(msg))
    return {"statusCode": code, "body": json.dumps({"error": msg, "success": False})}


def _try_code(code, validate=True):
    """Validate (optionally) then execute code. Returns (answer, ok, error)."""
    if validate:
        allowed, reason = validate_code(code)
        if not allowed:
            return None, False, "rejected: " + reason
    stdout, ok, error, truncated = execute_code(code)
    if not ok:
        return None, False, error
    if truncated:
        # Never return a partial number - it would look like a valid answer.
        return None, False, "output exceeded {} chars and was truncated; print a shorter answer".format(MAX_OUTPUT_LENGTH)
    answer = _extract_answer(stdout)
    if not answer:
        return None, False, "code ran but printed nothing"
    if len(answer) >= MAX_ANSWER_LENGTH:
        return None, False, "answer exceeded {} chars".format(MAX_ANSWER_LENGTH)
    return answer, True, None


def _extract_answer(stdout):
    """The answer is the last non-empty printed line."""
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_code(code):
    """Static checks before executing agent-authored code. Returns (ok, reason)."""
    if not code or not code.strip():
        return False, "empty code"
    if len(code) > MAX_CODE_LENGTH:
        return False, "code longer than {} chars".format(MAX_CODE_LENGTH)

    lowered = code.lower()
    for token in BANNED_TOKENS:
        if token in lowered:
            return False, "banned token {!r}".format(token)

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, "syntax error: {}".format(exc.msg)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return False, "import of {!r} not allowed".format(alias.name)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                return False, "import from {!r} not allowed".format(node.module)
        elif isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name in BANNED_CALL_NAMES:
                return False, "call to {!r} not allowed".format(name)

    if "print" not in code:
        return False, "code must print its answer"

    return True, ""


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def _set_limits():  # pragma: no cover - runs in the child process
    if resource is None:
        return
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_SECONDS, CPU_LIMIT_SECONDS))
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    except Exception:
        pass


def execute_code(code):
    """
    Run code in a locked-down subprocess.
    Returns (stdout, ok, error, truncated).
    """
    program = PRELUDE + code
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", program],
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT_SECONDS,
            env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "PYTHONHASHSEED": "0"},
            preexec_fn=_set_limits if resource is not None else None,
        )
    except subprocess.TimeoutExpired:
        return "", False, "timed out after {}s - use a faster algorithm".format(EXEC_TIMEOUT_SECONDS), False
    except Exception as exc:
        return "", False, "{}: {}".format(type(exc).__name__, exc), False

    raw_stdout = result.stdout or ""
    truncated = len(raw_stdout) > MAX_OUTPUT_LENGTH
    stdout = raw_stdout[:MAX_OUTPUT_LENGTH].strip()
    stderr = (result.stderr or "")[-800:].strip()
    if result.returncode == 0:
        return stdout, True, None, truncated
    return stdout, False, stderr or "exit code {}".format(result.returncode), truncated


# ---------------------------------------------------------------------------
# Safe arithmetic (replaces the old raw eval() fallback)
# ---------------------------------------------------------------------------
_AST_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
MAX_POW_EXPONENT = 10 ** 7


def safe_arithmetic(expr):
    """
    Evaluate a pure-arithmetic expression without eval().
    Returns (value, ok). Rejects names, calls, attributes and huge powers.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except (SyntaxError, ValueError, MemoryError):
        return None, False
    try:
        return _eval_node(tree.body), True
    except Exception:
        return None, False


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("non-numeric constant")
        return node.value
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_eval_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +_eval_node(node.operand)
        raise ValueError("unsupported unary op")
    if isinstance(node, ast.BinOp):
        handler = _AST_BINOPS.get(type(node.op))
        if handler is None:
            raise ValueError("unsupported binary op")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POW_EXPONENT:
            raise ValueError("exponent too large")
        return handler(left, right)
    raise ValueError("unsupported expression node")


def arithmetic_from_question(question):
    """Pull the longest arithmetic-looking expression out of the text and evaluate it."""
    text = question.replace("^", "**").replace("x", "*") if _looks_like_product(question) else question.replace("^", "**")
    candidates = re.findall(r"[-+]?[\d\s\.\+\-\*\/\%\(\)]{3,}", text)
    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        cleaned = candidate.strip().strip("+-*/%.").strip()
        if not cleaned or not any(ch.isdigit() for ch in cleaned):
            continue
        if not any(op in cleaned for op in "+-*/%"):
            continue
        value, ok = safe_arithmetic(cleaned)
        if ok:
            return _format_number(value), True
    return None, False


def _looks_like_product(text):
    return bool(re.search(r"\d\s*x\s*\d", text, re.IGNORECASE))


def _format_number(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# ---------------------------------------------------------------------------
# Number-word parsing ("three thousandth" -> 3000)
# ---------------------------------------------------------------------------
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9}
_ORDINAL_WORDS = {
    "first": "one", "second": "two", "third": "three", "fifth": "five",
    "eighth": "eight", "ninth": "nine", "twelfth": "twelve",
}


def _normalise_ordinal(word):
    if word in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[word]
    if word.endswith("ieth"):
        return word[:-4] + "y"
    if word.endswith("th"):
        return word[:-2]
    return word


def words_to_int(phrase):
    """Convert an English number phrase to an int, or return None."""
    tokens = re.findall(r"[a-z]+", phrase.lower())
    total = current = 0
    matched = False
    for token in tokens:
        token = _normalise_ordinal(token)
        if token in _UNITS:
            current += _UNITS[token]
            matched = True
        elif token in _SCALES:
            scale = _SCALES[token]
            if scale == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * scale
                current = 0
            matched = True
        elif token == "and" and matched:
            continue
        elif matched:
            break
    return (total + current) if matched else None


def extract_int(text, patterns, default=None):
    """Find an integer near one of the given regex patterns, digits or words."""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        for group in match.groups():
            if not group:
                continue
            digits = re.sub(r"[,\s_]", "", group)
            if digits.isdigit():
                return int(digits)
            value = words_to_int(group)
            if value:
                return value
    return default


# ---------------------------------------------------------------------------
# Pattern library - fallback for when no code was supplied
# ---------------------------------------------------------------------------
def generate_solver_code(question):
    """
    Return a self-contained Python program for a recognised problem family,
    or None if nothing matches. Ordering matters: most specific first.
    """
    q = question.lower().strip()
    for builder in _PATTERN_BUILDERS:
        code = builder(q)
        if code:
            return code
    return None


def _p_fibonacci(q):
    if "fibonacci" not in q and not re.search(r"\bfib\b", q):
        return None
    n = extract_int(q, [
        r"(\d[\d,\s]*)\s*(?:st|nd|rd|th)\b",
        r"\b([a-z\-\s]+?(?:th|first|second|third))\s+fib",
        r"fib\w*\s*(?:number|term|element)?\s*#?\s*(\d[\d,\s]*)",
        r"(\d[\d,\s]*)",
    ], 100)
    last_digits = extract_int(q, [r"last\s+(\d+)\s+digits?"])
    modulus = extract_int(q, [r"mod(?:ulo|ulus)?\s*(\d[\d,\s]*)"])

    if "sum" in q and "even" in q:
        return _prog([
            "limit = {}".format(n),
            "a, b, total = 1, 2, 0",
            "while a <= limit:",
            "    if a % 2 == 0:",
            "        total += a",
            "    a, b = b, a + b",
            "print(total)",
        ])

    if last_digits:
        mod = 10 ** last_digits
        return _prog([
            "mod = {}".format(mod),
            "a, b = 0, 1",
            "for _ in range({} - 1):".format(n),
            "    a, b = b, (a + b) % mod",
            "print(str(b % mod).zfill({}))".format(last_digits),
        ])

    if modulus:
        return _prog([
            "mod = {}".format(modulus),
            "a, b = 0, 1",
            "for _ in range({} - 1):".format(n),
            "    a, b = b, (a + b) % mod",
            "print(b % mod)",
        ])

    return _prog([
        "a, b = 0, 1",
        "for _ in range({} - 1):".format(n),
        "    a, b = b, a + b",
        "print(b)",
    ])


def _p_factorial(q):
    if "factorial" not in q and not re.search(r"\d\s*!", q):
        return None
    n = extract_int(q, [
        r"(\d[\d,\s]*)\s*!",
        r"factorial\s+of\s+(\d[\d,\s]*)",
        r"(\d[\d,\s]*)\s*factorial",
        r"factorial\s+of\s+([a-z\-\s]+)",
        r"(\d[\d,\s]*)",
    ], 100)
    modulus = extract_int(q, [r"mod(?:ulo|ulus)?\s*(\d[\d,\s]*)"])

    if "trailing zero" in q or "trailing 0" in q:
        return _prog([
            "n, count, p = {}, 0, 5".format(n),
            "while p <= n:",
            "    count += n // p",
            "    p *= 5",
            "print(count)",
        ])

    if modulus:
        return _prog([
            "mod = {}".format(modulus),
            "result = 1",
            "for i in range(2, {} + 1):".format(n),
            "    result = (result * i) % mod",
            "print(result)",
        ])

    body = [
        "result = 1",
        "for i in range(2, {} + 1):".format(n),
        "    result *= i",
    ]
    if "digit" in q and "sum" in q:
        body.append("print(sum(int(d) for d in str(result)))")
    elif "how many digit" in q or "number of digit" in q:
        body.append("print(len(str(result)))")
    elif "last" in q and "digit" in q:
        k = extract_int(q, [r"last\s+(\d+)\s+digits?"], 1)
        body.append("print(str(result)[-{}:])".format(k))
    else:
        body.append("print(result)")
    return _prog(body)


def _p_primes(q):
    if "prime" not in q:
        return None

    if "largest" in q and "factor" in q:
        n = extract_int(q, [r"(\d[\d,\s]*)"], 0)
        return _prog([
            "n = {}".format(n),
            "largest, d = 1, 2",
            "while d * d <= n:",
            "    while n % d == 0:",
            "        largest, n = d, n // d",
            "    d += 1",
            "print(max(largest, n))",
        ])

    if "factor" in q:
        n = extract_int(q, [r"(\d[\d,\s]*)"], 0)
        return _prog([
            "n = {}".format(n),
            "factors, d = [], 2",
            "while d * d <= n:",
            "    while n % d == 0:",
            "        factors.append(d)",
            "        n //= d",
            "    d += 1",
            "if n > 1:",
            "    factors.append(n)",
            "print(' '.join(str(f) for f in factors))",
        ])

    limit = extract_int(q, [r"(?:below|under|less than|up to|beneath|smaller than)\s*(\d[\d,\s]*)"])
    if limit and ("sum" in q or "count" in q or "how many" in q):
        agg = "sum(primes)" if "sum" in q else "len(primes)"
        return _prog([
            "limit = {}".format(limit),
            "sieve = bytearray([1]) * limit",
            "sieve[0:2] = b'\\x00\\x00'",
            "i = 2",
            "while i * i < limit:",
            "    if sieve[i]:",
            "        sieve[i*i::i] = bytearray(len(sieve[i*i::i]))",
            "    i += 1",
            "primes = [i for i, p in enumerate(sieve) if p]",
            "print({})".format(agg),
        ])

    if re.search(r"is\s+\d+\s+(?:a\s+)?prime", q):
        n = extract_int(q, [r"(\d[\d,\s]*)"], 0)
        return _prog([
            "n = {}".format(n),
            "if n < 2:",
            "    print('no')",
            "else:",
            "    d, prime = 2, True",
            "    while d * d <= n:",
            "        if n % d == 0:",
            "            prime = False",
            "            break",
            "        d += 1",
            "    print('yes' if prime else 'no')",
        ])

    n = extract_int(q, [
        r"(\d[\d,\s]*)\s*(?:st|nd|rd|th)\s*prime",
        r"prime\s*(?:number)?\s*#?\s*(\d[\d,\s]*)",
        r"([a-z\-\s]+?(?:th|first|second|third))\s+prime",
        r"(\d[\d,\s]*)",
    ], 100)
    return _prog([
        "target = {}".format(n),
        "count, candidate = 0, 1",
        "while count < target:",
        "    candidate += 1",
        "    d, prime = 2, True",
        "    while d * d <= candidate:",
        "        if candidate % d == 0:",
        "            prime = False",
        "            break",
        "        d += 1",
        "    if prime:",
        "        count += 1",
        "print(candidate)",
    ])


def _p_collatz(q):
    if "collatz" not in q and "hailstone" not in q:
        return None
    if "longest" in q or "under" in q or "below" in q:
        limit = extract_int(q, [r"(?:under|below|up to|less than)\s*(\d[\d,\s]*)", r"(\d[\d,\s]*)"], 1000000)
        return _prog([
            "limit = {}".format(limit),
            "cache = {1: 1}",
            "def length(n):",
            "    stack = []",
            "    while n not in cache:",
            "        stack.append(n)",
            "        n = n // 2 if n % 2 == 0 else 3 * n + 1",
            "    value = cache[n]",
            "    while stack:",
            "        value += 1",
            "        cache[stack.pop()] = value",
            "    return value",
            "best, best_n = 0, 1",
            "for i in range(1, limit):",
            "    L = length(i)",
            "    if L > best:",
            "        best, best_n = L, i",
            "print(best_n)",
        ])
    n = extract_int(q, [r"(\d[\d,\s]*)"], 27)
    return _prog([
        "n, count = {}, 1".format(n),
        "while n != 1:",
        "    n = n // 2 if n % 2 == 0 else 3 * n + 1",
        "    count += 1",
        "print(count)",
    ])


def _p_gcd_lcm(q):
    is_lcm = "lcm" in q or "least common" in q or "lowest common" in q
    is_gcd = "gcd" in q or "greatest common" in q or "hcf" in q or "highest common" in q
    if not (is_lcm or is_gcd):
        return None
    nums = [int(x) for x in re.findall(r"\d+", q)]
    if len(nums) < 2:
        return None
    if is_lcm:
        return _prog([
            "from math import gcd",
            "nums = {}".format(nums),
            "result = nums[0]",
            "for n in nums[1:]:",
            "    result = result * n // gcd(result, n)",
            "print(result)",
        ])
    return _prog([
        "from math import gcd",
        "from functools import reduce",
        "print(reduce(gcd, {}))".format(nums),
    ])


def _p_powmod(q):
    if "mod" not in q:
        return None
    match = re.search(
        r"(\d+)\s*(?:\^|\*\*|(?:raised\s+)?to\s+the\s+power(?:\s+of)?|raised\s+to)\s*(\d+)",
        q,
    )
    if not match:
        return None
    modulus = extract_int(q, [r"mod(?:ulo|ulus)?\s*(\d[\d,\s]*)"])
    if not modulus:
        return None
    return _prog(["print(pow({}, {}, {}))".format(match.group(1), match.group(2), modulus)])


def _p_divisors(q):
    if "divisor" not in q and "factors of" not in q and "perfect number" not in q:
        return None
    n = extract_int(q, [r"(\d[\d,\s]*)"], 0)
    if not n:
        return None
    base = [
        "n = {}".format(n),
        "divs = []",
        "i = 1",
        "while i * i <= n:",
        "    if n % i == 0:",
        "        divs.append(i)",
        "        if i != n // i:",
        "            divs.append(n // i)",
        "    i += 1",
        "divs.sort()",
    ]
    if "sum" in q:
        base.append("print(sum(divs))")
    elif "how many" in q or "count" in q or "number of" in q:
        base.append("print(len(divs))")
    elif "perfect number" in q:
        base.append("print('yes' if sum(divs[:-1]) == n else 'no')")
    else:
        base.append("print(' '.join(str(d) for d in divs))")
    return _prog(base)


def _p_binomial(q):
    if not any(w in q for w in ("binomial", "choose", "combination", "permutation")):
        return None
    nums = [int(x) for x in re.findall(r"\d+", q)]
    if len(nums) < 2:
        return None
    n, k = nums[0], nums[1]
    if "permutation" in q:
        return _prog([
            "from math import factorial",
            "print(factorial({}) // factorial({} - {}))".format(n, n, k),
        ])
    return _prog([
        "from math import factorial",
        "print(factorial({}) // (factorial({}) * factorial({} - {})))".format(n, k, n, k),
    ])


def _p_digit_ops(q):
    if "digit" not in q:
        return None
    n = extract_int(q, [r"(\d[\d,\s]*)"])
    if n is None:
        return None
    if "digital root" in q:
        return _prog(["n = {}".format(n), "print(1 + (n - 1) % 9 if n else 0)"])
    if "sum" in q:
        return _prog(["print(sum(int(d) for d in str({})))".format(n)])
    if "product" in q:
        return _prog([
            "p = 1",
            "for d in str({}):".format(n),
            "    p *= int(d)",
            "print(p)",
        ])
    if "reverse" in q:
        return _prog(["print(str({})[::-1])".format(n)])
    if "how many" in q or "number of" in q or "count" in q:
        return _prog(["print(len(str({})))".format(n)])
    return None


def _p_series(q):
    if "sum" not in q:
        return None
    if "square" in q:
        n = extract_int(q, [r"(?:first|to|up to|through)\s*(\d[\d,\s]*)", r"(\d[\d,\s]*)"])
        if n:
            return _prog(["n = {}".format(n), "print(n * (n + 1) * (2 * n + 1) // 6)"])
    if "cube" in q:
        n = extract_int(q, [r"(?:first|to|up to|through)\s*(\d[\d,\s]*)", r"(\d[\d,\s]*)"])
        if n:
            return _prog(["n = {}".format(n), "print((n * (n + 1) // 2) ** 2)"])
    if "multiple" in q:
        nums = [int(x) for x in re.findall(r"\d+", q)]
        if len(nums) >= 3:
            a, b, limit = nums[0], nums[1], nums[-1]
            return _prog([
                "print(sum(i for i in range(1, {}) if i % {} == 0 or i % {} == 0))".format(limit, a, b),
            ])
    match = re.search(r"(?:first|1\s*to|from\s*1\s*to|up to)\s*(\d[\d,\s]*)", q)
    if match and ("integer" in q or "natural" in q or "number" in q):
        n = int(re.sub(r"[,\s]", "", match.group(1)))
        return _prog(["n = {}".format(n), "print(n * (n + 1) // 2)"])
    return None


def _p_base_convert(q):
    n = extract_int(q, [r"(\d[\d,\s]*)"])
    if n is None:
        return None
    if "binary" in q:
        return _prog(["print(bin({})[2:])".format(n)])
    if "hexadecimal" in q or "hex" in q:
        return _prog(["print(hex({})[2:])".format(n)])
    if "octal" in q:
        return _prog(["print(oct({})[2:])".format(n)])
    return None


def _p_roots(q):
    if "square root" in q:
        n = extract_int(q, [r"(\d[\d,\s]*)"])
        if n is not None:
            if "integer" in q or "floor" in q or "isqrt" in q:
                return _prog(["import math", "print(math.isqrt({}))".format(n)])
            return _prog(["import math", "print(math.sqrt({}))".format(n)])
    return None


def _p_palindrome(q):
    if "palindrome" not in q:
        return None
    if "largest" in q and ("product" in q or "digit" in q):
        digits = extract_int(q, [r"(\d+)[- ]digit"], 3)
        low, high = 10 ** (digits - 1), 10 ** digits
        return _prog([
            "best = 0",
            "for a in range({}, {}):".format(low, high),
            "    for b in range(a, {}):".format(high),
            "        p = a * b",
            "        if p > best and str(p) == str(p)[::-1]:",
            "            best = p",
            "print(best)",
        ])
    n = extract_int(q, [r"(\d[\d,\s]*)"])
    if n is not None:
        return _prog(["s = str({})".format(n), "print('yes' if s == s[::-1] else 'no')"])
    return None


_PATTERN_BUILDERS = (
    _p_fibonacci,
    _p_factorial,
    _p_primes,
    _p_collatz,
    _p_gcd_lcm,
    _p_powmod,
    _p_palindrome,
    _p_divisors,
    _p_binomial,
    _p_series,
    _p_digit_ops,
    _p_base_convert,
    _p_roots,
)


def _prog(lines):
    return "\n".join(lines) + "\n"
