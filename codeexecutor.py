"""
AWS AI League - Code Execution Lambda for c2 (Blue Brain) challenges.
Handler: code_executor_lambda.lambda_handler
"""

import json
import subprocess
import sys
import textwrap
import re

EXEC_TIMEOUT_SECONDS = 10
MAX_OUTPUT_LENGTH = 1000
NEWLINE = chr(10)

def lambda_handler(event, context):
    try:
        if isinstance(event, dict) and "body" in event:
            body = event["body"]
            body = json.loads(body) if isinstance(body, str) else body
        else:
            body = event
        if not isinstance(body, dict):
            return _err(400, "Body must be a JSON object")

        question = (
            body.get("question")
            or body.get("prompt")
            or body.get("message")
            or body.get("challenge")
            or ""
        ).strip()

        if not question:
            return _err(400, "Missing question field")

        code = generate_solver_code(question)
        result, success, error = execute_code(code)

        if success:
            answer = result.strip()
            lines = [l for l in answer.splitlines() if l.strip()]
            if lines:
                answer = lines[-1].strip()
            response = {"answer": answer, "code": code, "success": True}
        else:
            fallback_code = generate_fallback_code(question)
            result2, success2, error2 = execute_code(fallback_code)
            if success2 and result2.strip():
                lines = [l for l in result2.strip().splitlines() if l.strip()]
                answer = lines[-1].strip() if lines else result2.strip()
                response = {"answer": answer, "code": fallback_code, "success": True, "note": "fallback"}
            else:
                response = {"answer": None, "code": code, "success": False, "error": error or error2}

        return {"statusCode": 200, "body": json.dumps(response)}
    except Exception as exc:
        return _err(500, str(exc))

def _err(code, msg):
    return {"statusCode": code, "body": json.dumps({"error": msg, "success": False})}

def generate_solver_code(question):
    q = question.lower().strip()

    # Fibonacci
    if "fibonacci" in q or "fib " in q:
        n_match = re.search(r"(\d+)\s*(?:th|st|nd|rd)?\s*(?:number|term|element|fib)", q)
        if not n_match:
            n_match = re.search(r"(?:number|term|element|fib)\w*\s*(\d+)", q)
        n = int(n_match.group(1)) if n_match else 100
        digits_match = re.search(r"last\s+(\d+)\s+digits?", q)
        last_digits = int(digits_match.group(1)) if digits_match else None
        if last_digits:
            return (
                "def fib(n):" + NEWLINE +
                "    mod = 10**" + str(last_digits) + NEWLINE +
                "    a, b = 0, 1" + NEWLINE +
                "    for _ in range(n - 1):" + NEWLINE +
                "        a, b = b, (a + b) % mod" + NEWLINE +
                "    return b % mod" + NEWLINE +
                "print(str(fib(" + str(n) + ")).zfill(" + str(last_digits) + "))" + NEWLINE
            )
        else:
            return (
                "def fib(n):" + NEWLINE +
                "    a, b = 0, 1" + NEWLINE +
                "    for _ in range(n - 1):" + NEWLINE +
                "        a, b = b, a + b" + NEWLINE +
                "    return b" + NEWLINE +
                "print(fib(" + str(n) + "))" + NEWLINE
            )

    # Factorial modulo
    if "factorial" in q or "!" in q:
        n_match = re.search(r"(\d+)\s*(?:factorial|!)", q)
        if not n_match:
            n_match = re.search(r"factorial\s*(?:of\s*)?(\d+)", q)
        n = int(n_match.group(1)) if n_match else 2024
        mod_match = re.search(r"mod(?:ulo|ular)?\s*\(?\s*(.*?)(?:\s*\)?\s*$|\s*\?)", q)
        if mod_match:
            mod_expr = mod_match.group(1).strip()
            mod_expr = re.sub(r"(\d+)\s*to the\s*(\d+)\s*(?:th|st|nd|rd)?", r"\1**\2", mod_expr)
            mod_expr = mod_expr.replace("^", "**")
            mod_expr = re.sub(r"(\d+)\s*to the power (?:of\s*)?(\d+)", r"\1**\2", mod_expr)
            mod_expr = re.sub(r"[^0-9\+\-\*\/]", "", mod_expr)
            if not mod_expr:
                mod_expr = "10**9+7"
        else:
            mod_expr = "10**9+7"
        return (
            "mod = " + mod_expr + NEWLINE +
            "result = 1" + NEWLINE +
            "for i in range(2, " + str(n) + " + 1):" + NEWLINE +
            "    result = (result * i) % mod" + NEWLINE +
            "print(result)" + NEWLINE
        )

    # Sum of primes
    if "prime" in q and "sum" in q:
        n_match = re.search(r"(?:below|under|less than|up to|beneath)\s*(\d+)", q)
        n = int(n_match.group(1)) if n_match else 1000000
        return (
            "def sieve_sum(limit):" + NEWLINE +
            "    is_prime = [True] * limit" + NEWLINE +
            "    is_prime[0] = is_prime[1] = False" + NEWLINE +
            "    for i in range(2, int(limit**0.5) + 1):" + NEWLINE +
            "        if is_prime[i]:" + NEWLINE +
            "            for j in range(i*i, limit, i):" + NEWLINE +
            "                is_prime[j] = False" + NEWLINE +
            "    return sum(i for i, p in enumerate(is_prime) if p)" + NEWLINE +
            "print(sieve_sum(" + str(n) + "))" + NEWLINE
        )

    # Nth prime
    if "prime" in q:
        n_match = re.search(r"(\d+)\s*(?:th|st|nd|rd)?\s*prime", q)
        if not n_match:
            n_match = re.search(r"prime\s*(?:number)?\s*#?\s*(\d+)", q)
        n = int(n_match.group(1)) if n_match else 100
        return (
            "def nth_prime(n):" + NEWLINE +
            "    primes = []" + NEWLINE +
            "    candidate = 2" + NEWLINE +
            "    while len(primes) < n:" + NEWLINE +
            "        if all(candidate % p != 0 for p in primes if p * p <= candidate):" + NEWLINE +
            "            primes.append(candidate)" + NEWLINE +
            "        candidate += 1" + NEWLINE +
            "    return primes[-1]" + NEWLINE +
            "print(nth_prime(" + str(n) + "))" + NEWLINE
        )

    # Power mod
    if ("power" in q or "**" in q or "^" in q or "raised to" in q) and "mod" in q:
        base_match = re.search(r"(\d+)\s*(?:\^|\*\*|raised to)\s*(\d+)", q)
        mod_match = re.search(r"mod(?:ulo)?\s*(\d+)", q)
        if base_match and mod_match:
            return "print(pow(" + base_match.group(1) + ", " + base_match.group(2) + ", " + mod_match.group(1) + "))"

    # Collatz
    if "collatz" in q:
        n_match = re.search(r"(\d+)", q)
        n = int(n_match.group(1)) if n_match else 27
        return (
            "def collatz_length(n):" + NEWLINE +
            "    count = 1" + NEWLINE +
            "    while n != 1:" + NEWLINE +
            "        n = n // 2 if n % 2 == 0 else 3 * n + 1" + NEWLINE +
            "        count += 1" + NEWLINE +
            "    return count" + NEWLINE +
            "print(collatz_length(" + str(n) + "))" + NEWLINE
        )

    # GCD / LCM
    if "gcd" in q or "greatest common" in q or "lcm" in q or "least common" in q:
        nums = re.findall(r"\d+", q)
        if len(nums) >= 2:
            if "lcm" in q or "least common" in q:
                return (
                    "from math import gcd" + NEWLINE +
                    "def lcm(a, b): return a * b // gcd(a, b)" + NEWLINE +
                    "nums = " + str(nums) + NEWLINE +
                    "result = int(nums[0])" + NEWLINE +
                    "for n in nums[1:]: result = lcm(result, int(n))" + NEWLINE +
                    "print(result)" + NEWLINE
                )
            else:
                return (
                    "from math import gcd" + NEWLINE +
                    "nums = " + str(nums) + NEWLINE +
                    "result = int(nums[0])" + NEWLINE +
                    "for n in nums[1:]: result = gcd(result, int(n))" + NEWLINE +
                    "print(result)" + NEWLINE
                )

    # Generic eval
    expr_match = re.search(r"(?:what is|calculate|compute|evaluate|find|solve)?\s*([\d\s\+\-\*\/\%\(\)\.\^]+)", q)
    if expr_match:
        expr = expr_match.group(1).strip().replace("^", "**")
        if expr and any(c.isdigit() for c in expr):
            return "print(eval('" + expr + "'))"

    return generate_fallback_code(question)

def generate_fallback_code(question):
    q = question.lower()
    numbers = [int(x) for x in re.findall(r"\d+", q)]
    if not numbers:
        return 'print("Unable to compute")'
    if "mod" in q and len(numbers) >= 2:
        mod = numbers[-1]
        if "factorial" in q:
            n = numbers[0]
            return (
                "mod = " + str(mod) + NEWLINE +
                "result = 1" + NEWLINE +
                "for i in range(2, " + str(n) + " + 1):" + NEWLINE +
                "    result = (result * i) % mod" + NEWLINE +
                "print(result)" + NEWLINE
            )
    if "sum" in q:
        if len(numbers) == 1:
            return "print(sum(range(1, " + str(numbers[0]) + " + 1)))"
        return "print(sum(" + str(numbers) + "))"
    if len(numbers) == 1:
        n = numbers[0]
        if "square" in q and "root" in q:
            return "import math; print(math.isqrt(" + str(n) + "))"
        if "factorial" in q:
            return (
                "result = 1" + NEWLINE +
                "for i in range(2, " + str(n) + " + 1): result *= i" + NEWLINE +
                "print(result)" + NEWLINE
            )
        if "binary" in q:
            return "print(bin(" + str(n) + ")[2:])"
        if "hex" in q:
            return "print(hex(" + str(n) + ")[2:])"
    return 'print("Unable to compute")'

def execute_code(code):
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
            timeout=EXEC_TIMEOUT_SECONDS,
            env={"PATH": "/usr/bin:/bin"},
        )
        stdout = result.stdout[:MAX_OUTPUT_LENGTH].strip()
        stderr = result.stderr[:500].strip()
        if result.returncode == 0 and stdout:
            return stdout, True, None
        return stdout, False, stderr or "exit code " + str(result.returncode)
    except subprocess.TimeoutExpired:
        return "", False, "timed out after " + str(EXEC_TIMEOUT_SECONDS) + "s"
    except Exception as e:
        return "", False, str(e)
