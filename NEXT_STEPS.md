# What to do next

**Current score: 13,787.** Best possible on this map: about 14,000.

You are at 98.5% of the maximum. Three of the four score components are already
maxed out:

| Component | Score | Status |
|---|---|---|
| Treasure | 1,000 / 1,000 | done, cannot improve |
| Lives | 1,250 / 1,250 | done, cannot improve |
| Coins | ~10,750 | map decides this, already collecting everything |
| Token bonus | 782 / 1,000 | **the only thing left** |

All the code changes are finished and merged. There is nothing left to fix in
this repository. The one remaining improvement is done in the AWS console.

---

## Do this: train one custom model

In the workshop menu, open **Optimize Agent with Model Customization** and follow
Steps 1 to 4.

**Train it for pathfinding.**

That is the whole task. It is worth **+109 points minimum**.

### Why pathfinding, and not something else

Your token bonus is `1000 - (average words your agent writes per challenge)`.
Pathfinding is where your agent writes the most by far: it repeats the entire
10x10 map into the tool call, then writes out a 69-step path. A small fine-tuned
model does that same job with far fewer words.

It is also the example the workshop itself walks you through, so you are on the
documented path rather than improvising.

### Why one model and not two

| Models trained | Your token penalty is reduced by |
|---|---|
| 1 | 50% |
| 2 | 70% |

The first model is the big win. The second one is only worth about +43 more
points and needs a second full training run. Train one. Only start a second if
the first finishes and you still have plenty of time.

---

## Two things to check before you submit

**1. The tool name must match.** The workshop example has the model produce
`{"name": "pathfinding_lambda"}`. Check that this matches the name your
pathfinding sub-agent actually registers. If they differ, the tool call fails,
your agent gets no path, and the run scores zero. The troubleshooting page lists
this exact problem as a cause of zero scores.

Do not change anything in this repo to "fix" this in advance. Your agent works
right now, so whatever name it uses today is correct. Only verify it after you
attach the fine-tuned model.

**2. Keep the pathfinding sub-agent prompt short.** The fine-tuned model is
trained to output the `<tool_call>` format directly. Long instructions add words
(which cost you bonus) and can work against the training.

---

## You cannot lose points by trying

The leaderboard keeps your **best** score across all submissions. Your 13,787 is
banked and safe. If a custom-model submission goes badly, it does not replace it.

So there is no risk in experimenting with the time you have left.
