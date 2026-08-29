"""
Red/Green door transform reverse-engineering harness.

We do NOT know the transform the red/green doors apply to their key. Grey/yellow
doors are known (first2last2, char5char7) but red/green have never been handled.

From a real run we know ONE fact:
    green key "fghi" -> raw "fghi" was REJECTED (cost 5 lives).

So the door wants a TRANSFORM of the key, not the raw value. A competitor's notes
for the same game say the door answer is computed FROM the key and that
"letters may come back as digits or another word" -- a strong hint the transform
may map letters to their alphabet positions, shift them, or similar.

HOW TO USE
----------
1. Each run, record the observed pair (key_value, correct_answer) if the game
   reveals the correct answer on failure, OR (key_value, accepted_answer) when a
   probe guess WINS.
2. Add the pair to OBSERVATIONS below.
3. Run:  python3 tools/door_transform_probe.py
   It prints, for every pair, which candidate formulas reproduce the answer.
   A formula that matches ALL pairs is very likely the real rule.

Add new candidate formulas to CANDIDATES as we think of them.
"""

import string

# ---------------------------------------------------------------------------
# Observations: (colour, key_value, correct_or_accepted_answer)
# Fill these in as we learn them from real runs.
# We only know so far that raw "fghi" is WRONG for green -- that is a negative,
# not a positive, so it does not go here. Add POSITIVE pairs only.
# ---------------------------------------------------------------------------
#
# CONFIRMED from the official challenge descriptions:
#   Red Door (c30):   "translate the code by reading it backwards"      -> reverse
#   Green Door (c31): "replace letters with the numbers that represent  -> alpha_positions
#                      them in order"
OBSERVATIONS = [
    ("red",   "shut", "tuhs"),   # reverse
    ("green", "fghi", "6789"),   # f=6 g=7 h=8 i=9
]


# ---------------------------------------------------------------------------
# Candidate transforms. Each takes the raw key string and returns a string.
# Ordered roughly by prior likelihood given the "letters -> digits" hint.
# ---------------------------------------------------------------------------
def alpha_positions(v):
    """f->6 g->7 ... joined with no separator. 'fghi' -> '6789'."""
    return "".join(str(string.ascii_lowercase.index(c) + 1) for c in v.lower()
                   if c in string.ascii_lowercase)


def alpha_positions_spaced(v):
    """'fghi' -> '6 7 8 9'."""
    return " ".join(str(string.ascii_lowercase.index(c) + 1) for c in v.lower()
                    if c in string.ascii_lowercase)


def sum_alpha_positions(v):
    """Sum of alphabet positions. 'fghi' -> 6+7+8+9 = 30."""
    return str(sum(string.ascii_lowercase.index(c) + 1 for c in v.lower()
                   if c in string.ascii_lowercase))


def reverse(v):
    """'fghi' -> 'ihgf'."""
    return v[::-1]


def caesar_shift(n):
    def _f(v):
        out = []
        for c in v:
            if c.islower():
                out.append(chr((ord(c) - 97 + n) % 26 + 97))
            elif c.isupper():
                out.append(chr((ord(c) - 65 + n) % 26 + 65))
            else:
                out.append(c)
        return "".join(out)
    return _f


def first2last2(v):
    return v[:2] + v[-2:] if len(v) >= 4 else None


def char5char7(v):
    return v[4] + v[6] if len(v) >= 7 else None


def upper(v):
    return v.upper()


def length(v):
    return str(len(v))


CANDIDATES = {
    "alpha_positions":        alpha_positions,
    "alpha_positions_spaced": alpha_positions_spaced,
    "sum_alpha_positions":    sum_alpha_positions,
    "reverse":                reverse,
    "caesar+1":               caesar_shift(1),
    "caesar-1":               caesar_shift(-1),
    "caesar+3":               caesar_shift(3),
    "first2last2":            first2last2,
    "char5char7":             char5char7,
    "upper":                  upper,
    "length":                 length,
}


def main():
    if not OBSERVATIONS:
        print("No observations yet. Add (colour, key, answer) tuples to OBSERVATIONS,")
        print("then re-run. Here is what each candidate does to the known green key 'fghi':\n")
        for name, fn in CANDIDATES.items():
            try:
                print(f"  {name:24s} fghi -> {fn('fghi')!r}")
            except Exception as e:
                print(f"  {name:24s} fghi -> ERROR {e}")
        print("\nAlso 'shut' (the red key seen in the same run):\n")
        for name, fn in CANDIDATES.items():
            try:
                print(f"  {name:24s} shut -> {fn('shut')!r}")
            except Exception as e:
                print(f"  {name:24s} shut -> ERROR {e}")
        return

    print("Testing candidate formulas against observations:\n")
    survivors = []
    for name, fn in CANDIDATES.items():
        ok = True
        detail = []
        for colour, key, ans in OBSERVATIONS:
            try:
                got = fn(key)
            except Exception:
                got = None
            match = (got == ans)
            ok = ok and match
            detail.append(f"{colour}:{key}->{got!r}{'==' if match else '!='}{ans!r}")
        flag = "MATCHES ALL" if ok else "no"
        print(f"  {name:24s} [{flag}]  " + "  ".join(detail))
        if ok:
            survivors.append(name)

    print()
    if survivors:
        print("Likely formula(s):", ", ".join(survivors))
    else:
        print("No candidate matched all observations. Add more candidates or more data.")


if __name__ == "__main__":
    main()
