ANALYSIS_MARKER = "MISTAKES:"

ANALYSIS_PROMPT = """You are reviewing several attempts at a mathematics problem.

QUESTION
{question}

REFERENCE SOLUTION
{solution}

Below are {n} attempts, each marked CORRECT or WRONG against the reference.

{attempts}

Recall what you are reviewing:

QUESTION
{question}

REFERENCE SOLUTION
{solution}

Reply with exactly these three sections, in this order:

ANALYSIS:
<Work through the attempts one at a time, at length. For each, say where its reasoning departs
from the reference, or why it holds up. Do not summarise yet - this section is your scratch
work and should be by far the longest part of your reply.>

MISTAKES:
- <a core error that caused wrong attempts to fail>
- <another, only if genuinely distinct>
CORRECT:
- <a core step or insight the correct attempts share>
- <another, only if genuinely distinct>

What matters in the last two sections is the underlying cause, not the surface symptom: the one
or two core reasons that separate the correct attempts from the wrong ones, rather than a list
of incidental differences. They are parsed automatically, so they must follow that shape exactly:
- `MISTAKES:` and `CORRECT:` each sit alone on their own line
- every item is a single line beginning with `- `, with no blank lines between items
- at most three items under each heading
- describe causes, never attempt numbers - the reader cannot see the attempts
- write nothing at all after the last item: no closing remarks, no code fences

Begin with the line `ANALYSIS:` and reason there before writing anything else.
"""

ANALYSIS_PREFILL = "ANALYSIS:\n"

ANALYSIS_ATTEMPT = "--- attempt {i} [{verdict}] ---\n{completion}"

ANALYSIS_ATTEMPT_TOKENS = 32  # the "--- attempt N [VERDICT] ---" header and separators

