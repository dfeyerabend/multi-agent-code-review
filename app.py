"""Gradio Web Interface — browser-based front end for the multi-agent code review pipeline.

Lets a user paste Python code and run the full Analyzer -> Enricher -> Optimizer -> Evaluator
pipeline from the browser: per-stage progress is streamed live, and the detailed report is
offered on demand. This module wraps orchestrator.run_pipeline and adds no review logic of its
own, so the console entrypoint and this UI render identically through the shared render_report
layer.

Run locally:
    python app.py
"""

import os
import sys
import time
import uuid
import signal
import asyncio
import logging
import tempfile
from collections import defaultdict

import gradio as gr

from config import setup_logging
from orchestrator import run_pipeline

logger = logging.getLogger(__name__)

# === RATE LIMITING ===

RATE_LIMIT_PER_IP_HOURLY = 5    # runs one visitor may start per hour
RATE_LIMIT_GLOBAL_DAILY = 50    # hard ceiling on total runs per day across all visitors


class RateLimiter:
    """
    In-memory run counter enforcing a per-IP hourly cap and a global daily cap.

    Pipeline: a single module-level instance (rate_limiter) is consulted by run_review
    (this module) before each pipeline run. State lives in memory only, so it resets when the
    server restarts; timestamps older than their window are pruned on every check.

    Why two limits: the per-IP hourly cap stops one visitor from monopolising the demo, while
    the global daily cap is a hard ceiling on total runs so a burst of many different IPs cannot
    run up an unbounded API bill.
    """

    def __init__(self, per_ip_hourly: int, global_daily: int):
        self.per_ip_hourly = per_ip_hourly
        self.global_daily = global_daily
        self._ip_runs: dict[str, list[float]] = defaultdict(list)   # IP -> run timestamps
        self._global_runs: list[float] = []                         # all run timestamps

    def check(self, ip: str) -> tuple[bool, str]:
        """
        Reports whether a run is allowed for this IP right now, without recording it.

        Args:
            ip: The caller's IP address.

        Returns:
            (allowed, message): allowed is True when the run may proceed; on False, message
            names the limit that was hit. message is empty when allowed.
        """
        now = time.time()
        self._prune(now)

        # Global ceiling is checked first: once the day is capped, no IP may run.
        if len(self._global_runs) >= self.global_daily:
            return False, (f"Daily demo limit reached ({self.global_daily} runs across all "
                           "visitors). Please try again tomorrow.")

        if len(self._ip_runs.get(ip, [])) >= self.per_ip_hourly:
            return False, (f"Hourly limit reached ({self.per_ip_hourly} runs). "
                           "Please try again later.")
        return True, ""

    def record(self, ip: str) -> None:
        """Records one run for this IP and in the global tally. Call only when a run starts."""
        now = time.time()
        self._ip_runs[ip].append(now)
        self._global_runs.append(now)

    def remaining(self, ip: str) -> int:
        """Returns how many runs this IP has left in the current hour."""
        self._prune(time.time())
        return max(0, self.per_ip_hourly - len(self._ip_runs.get(ip, [])))

    def _prune(self, now: float) -> None:
        # Drop timestamps outside their window so counts reflect only the live period.
        hour_ago = now - 3600
        day_ago = now - 86400
        for ip in list(self._ip_runs):
            self._ip_runs[ip] = [t for t in self._ip_runs[ip] if t > hour_ago]
            if not self._ip_runs[ip]:
                del self._ip_runs[ip]   # keep the dict from growing as visitors come and go
        self._global_runs = [t for t in self._global_runs if t > day_ago]


rate_limiter = RateLimiter(
    per_ip_hourly=RATE_LIMIT_PER_IP_HOURLY,
    global_daily=RATE_LIMIT_GLOBAL_DAILY,
)

def get_client_ip(request: gr.Request | None) -> str:
    """
    Extracts the caller's IP address, honouring the reverse proxy in front of the app.

    Pipeline: called by run_review and the runs-remaining display (this module) to key the rate
    limiter. Behind a proxy such as Railway, request.client.host is the proxy itself, so the real
    client is the first entry of the X-Forwarded-For header when that header is present.

    Args:
        request: The Gradio request for the current event, or None if unavailable.

    Returns:
        The client IP string, or "unknown" when it cannot be determined.
    """
    try:
        if request is None:                         # guards against gradio empty object calls
            return "unknown"
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()   # first hop is the original client
        return request.client.host if request.client else "unknown"
    except Exception as e:
        logger.warning("get_client_ip: could not read client IP (%s), using 'unknown'", str(e))
        return "unknown"



# === STATIC UI CONTENT ===

# Always-open intro shown under the header. Gives a first-time visitor the run steps and a plain

INTRO_MD = """
**How to run**

1. Click **Load example code** to drop in a sample, or paste your own Python code.
2. Click **Run review**.
3. Watch the four stages stream in the console, then open **Full detailed report** for the full write up.

For more detail, use the **Example code explanation** and **How to read the report** buttons.

---

**How the pipeline works**

Your code passes through four agents in a row. Each one does a single job and hands its result to the next agent:   

1. **Analyzer** reads the code and finds issues, using a linter, a security scanner, a structure check, and a set of company rules.
2. **Enricher** looks up matching best practice guidance (a RAG search over a knowledge base) and explains why each issue matters.
3. **Optimizer** writes a concrete fix suggestion for each issue.
4. **Evaluator** acts as a judge, scoring every fix for correctness, completeness, and whether it follows the retrieved guidance, then assembles the report.

`Analyzer → Enricher → Optimizer → Evaluator`

The agents reach their tools (the linter, the scanners, the knowledge base) through an MCP server, and never talk to each other directly: a plain Python orchestrator passes data from one to the next.
"""

# Pre-run placeholder for the console: an empty box should still name both available actions.
CONSOLE_HINT = (
    "Please provide code and click **Run**. "
    "To try it out, click **Load example code** to fill in a sample."
)

# Mirrors the orchestrator's default input: each line trips a known finding to exercise every stage.
EXAMPLE_CODE = """import os, sys
import json

def get_user(id):  # REASON: fetch user record by id
    query = 'SELECT * FROM users WHERE id = ' + id
    return query

def calculate_total(a, b):
    return a + b

def process_input(data):  # REASON: parse and validate raw input
    try:
        return int(data)
    except ValueError:
        raise ValueError('invalid data')
"""

# Shown in the report panel before the first run and reset there between runs.
REPORT_PLACEHOLDER = "Run the pipeline to generate a report."


# Static explainer for the bundled example, shown in the "Example code explanation" panel.
# Content is specific to EXAMPLE_CODE above; the caveat guards against a user editing the code and then reading results that no longer match.
EXAMPLE_EXPLANATION_MD = """
> **Note:** this describes the bundled example only. If you edit the code in the editor, these expected results no longer apply.

The sample is written so every stage of the pipeline has something to do. Each part below shows a piece of the code, followed by what the review is expected to flag in it.

**The imports (lines 1 and 2)**

```python
import os, sys
import json
```

Two kinds of problem here. Writing two imports on one line (`import os, sys`) is a style issue. Separately, none of `os`, `sys`, or `json` is ever used, so all three count as unused imports. Expect one style finding for the shared line plus three unused import findings.

**get_user (lines 4 to 6)**

```python
def get_user(id):  # REASON: fetch user record by id
    query = 'SELECT * FROM users WHERE id = ' + id
    return query
```

This function pastes user input straight into a database query with `+`. That is a classic SQL injection risk and is the most serious finding in the example. The `# REASON:` comment is intentional: the company rules ask every function to state why it exists, and this comment satisfies that rule so it does not fire here.

**calculate_total (lines 8 and 9)**

```python
def calculate_total(a, b):
    return a + b
```

The code itself is fine. The only issue is the missing `# REASON:` comment, so the company rule about explaining why a function exists fires here. This is the clean, isolated case: exactly one finding and nothing else.

**process_input (lines 11 to 15)**

```python
def process_input(data):  # REASON: parse and validate raw input
    try:
        return int(data)
    except ValueError:
        raise ValueError('invalid data')
```

The `raise` line trips two rules at once, on the same line. Raising inside an `except` without `from` throws away the original error context, and the company rules forbid raising a built in exception such as `ValueError` and ask for a custom error type instead. This is the combined case: two findings share one line and are fixed together.

**Expected findings**

The review should report **8 issues** in total:

| Category        | Count | Where |
| --------------- | ----- | ----- |
| Style           | 1     | two imports on one line |
| Logic           | 5     | three unused imports, raise without `from`, forbidden built in raise |
| Security        | 1     | SQL injection in `get_user` |
| Maintainability | 1     | missing `# REASON:` in `calculate_total` |
| **Total**       | **8** | |
"""


# Static explainer for the report layout, shown in the "How to read the report" panel. Focuses on structure, not on the example's specific findings (those live in the panel above).
REPORT_GUIDE_MD = """
> **Note:** this shows the report for the bundled example. The wording of each fix and the reasoning is written fresh by the model on every run, so your report reads a little differently. The structure stays the same.

The report is organised **per fix, not per issue**. When several issues sit on the same line they are fixed together and share one block, so the 8 findings above appear as 5 blocks.

**Top summary line** — one sentence: how many issues were found, how many fixes they became, and how many were approved versus still need attention.

**Status table** — a count for each possible verdict:

- **Approved:** correct, complete, and in line with any cited guideline.
- **Incorrect:** the suggested code is wrong or broken.
- **Incomplete:** valid, but it only solves part of the problem.
- **Noncompliant:** the code is right but ignores a guideline that was retrieved for it.
- **No fix:** no fix could be produced for that issue.
- **Not evaluated:** the judge could not reach a verdict.

**Each fix block** contains:

- **A title** such as `Line 5 — APPROVED`: the line the fix targets and its verdict, with a status icon.
- **Found:** the issue or issues this one fix resolves.
- **Original code:** the surrounding lines, numbered, as they were before.
- **Suggested fix:** the improved code, or a short note such as "remove line(s) 2".
- **Verdicts:** three separate judgements. *Correctness* asks whether the code works. *Completeness* asks whether it solves the whole problem. *Faithfulness* asks whether it follows the guideline retrieved for it, or reads "not applicable" when there was none.
- **Grounded in:** the style guide sections or documentation the fix leaned on.
- **Evaluator reasoning:** a fold out block with the judge's full explanation.

---

**Example report for this code** (reasoning shortened; the real one is longer):

`````markdown
# Code Review Report

**8 finding(s) across 5 fix(es) — 8 approved, 0 need attention.**

| Status           | Count |
| ---------------- | ----- |
| ✅ Approved       | 8     |
| ❌ Incorrect      | 0     |
| ❌ Incomplete     | 0     |
| ⚠️ Noncompliant  | 0     |
| ⚠️ No fix        | 0     |
| ⚠️ Not evaluated | 0     |

---

### Line 1 — ✅ APPROVED

**Found:**
- `E401` — Multiple imports on one line
- `F401` — `os` imported but unused
- `F401` — `sys` imported but unused

**Suggested fix:**

```python
import json
```

**Verdicts:** correctness `pass` · completeness `complete` · faithfulness `not applicable`

**Grounded in:** pyguide §3.13, ruff docs

<details><summary>Evaluator reasoning</summary>
Removes the unused os and sys imports and the multiple-import line, resolving all three findings.
</details>

---

### Line 5 — ✅ APPROVED

**Found:**
- `B608` — Possible SQL injection vector through string-based query construction.

**Suggested fix:**

```python
def get_user(id):  # REASON: fetch user record by id
    query = 'SELECT * FROM users WHERE id = %s'
    return query, (id,)
```

**Verdicts:** correctness `pass` · completeness `complete` · faithfulness `not applicable`

**Grounded in:** bandit B608 docs, CWE-89

<details><summary>Evaluator reasoning</summary>
Switches to a parameterized query, separating SQL from user input.
</details>

---

### Line 15 — ✅ APPROVED

**Found:**
- `B904` — Within an `except` clause, raise with `raise ... from err`
- `COMPANY-1.3` — Raises a built-in exception; must raise AppError or a subclass (raises 'ValueError')

**Suggested fix:**

```python
    except ValueError as err:
        raise AppValidationError('invalid data') from err
```

**Verdicts:** correctness `pass` · completeness `complete` · faithfulness `faithful`

**Grounded in:** pyguide §2.4, company_rules §1.3

<details><summary>Evaluator reasoning</summary>
Chains the original error with from and swaps the built-in for a custom exception type.
</details>
"""


# === UI HELPERS ===

def load_example_code() -> str:
    """
    Returns the showcase example code for the "Load example code" button.

    Pipeline: wired to the Load-example button in the Gradio UI (this module). Its return
    value replaces whatever is in the code input box, giving a one-click demo input.

    Returns:
        The EXAMPLE_CODE snippet as a single string.
    """
    return EXAMPLE_CODE  # constant only: no input to validate, no failure path to guard

def render_runs_remaining(request: gr.Request) -> str:
    """
    Builds the "runs remaining this hour" line for the current visitor.

    Pipeline: wired to demo.load and to the end of the Run chain in the Gradio UI (this module),
    so the count shows on page load and refreshes after every run.

    Args:
        request: Gradio request for the current event, used to key the rate limiter by IP.

    Returns:
        A short Markdown line stating how many runs the visitor has left this hour.
    """
    try:
        left = rate_limiter.remaining(get_client_ip(request))
        return f"Runs remaining this hour: **{left} / {RATE_LIMIT_PER_IP_HOURLY}**"
    except Exception as e:
        logger.warning("render_runs_remaining failed (%s)", str(e))
        return ""


def render_char_count(code_input: str) -> str:
    """
    Builds the live character-count line under the editor, warning when over the limit.

    Pipeline: wired to the code editor's change event in the Gradio UI (this module). Display
    only; the hard limit is enforced separately in run_review.

    Args:
        code_input: Current editor contents.

    Returns:
        A Markdown line with the character count, styled as a warning once over the limit.
    """
    used = len(code_input) if isinstance(code_input, str) else 0
    if used > CODE_CHAR_LIMIT:
        return f"⚠️ **{used} / {CODE_CHAR_LIMIT} characters** (over the limit, shorten to run)"
    return f"{used} / {CODE_CHAR_LIMIT} characters"




# === PIPELINE RUN HANDLER ===

_SENTINEL = object()  # marks the end of the emit stream on the review queue


async def _review_task(code_input: str, block_queue: asyncio.Queue) -> dict:
    """
    Runs the pipeline and funnels its emitted blocks into the queue.

    Pipeline: spawned by run_review (this module) as an asyncio task. run_pipeline's `emit`
    callback is this queue's put_nowait, so each rendered block reaches the UI as it is
    produced. The sentinel pushed in `finally` tells run_review the stream is complete on
    every exit path, success or error.

    Args:
        code_input:  Raw code string from the UI, forwarded to run_pipeline.
        block_queue: asyncio.Queue the emitted Markdown blocks are pushed onto.

    Returns:
        The run_pipeline result dict (status, pipeline_stats, and on success the report).
    """
    # No except on purpose: run_pipeline is defensive, and any escaped error should surface in run_review (which owns the UI), not be swallowed here. The finally still ends the stream.
    try:
        return await run_pipeline(code_input, emit=block_queue.put_nowait)
    finally:
        block_queue.put_nowait(_SENTINEL)

CODE_CHAR_LIMIT = 2000       # code character cap on input size; also bounds token usage per run
CONCURRENCY_LIMIT = 3        # max simultaneous runs (mirrored on demo.queue)

_active_runs = 0             # live count of in-flight reviews, drives the busy note
BUSY_NOTE = ("_The service is busy right now, so this review may take a little longer than usual._\n\n")

async def run_review(code_input: str, request: gr.Request):
    """
    Gradio event handler: enforces the safety limits, streams progress, then stores the report.

    Pipeline: bound to the Run button in the Gradio UI (this module). Before running it checks the
    rate limit and input size and refuses over-limit requests with a console message. On an allowed
    run it drives run_pipeline in a background task, streams the emitted Markdown, keeps the report
    in session state, and enables the download button.

    Args:
        code_input: Raw code string from the code input box.
        request:    Gradio request for the current click, used to key the rate limiter by IP.

    Yields:
        (console, report_body, download_button, report_state) tuples. Report outputs are skipped
        during streaming and only set once the run finishes.
    """
    global _active_runs
    try:
        # Never act on unchecked input: a non-string means the UI is miswired.
        if not isinstance(code_input, str):
            yield (f"Internal error: expected code as text, got {type(code_input).__name__}. "
                   "Please reload the page."), gr.skip(), gr.skip(), gr.skip()
            return

        # Rate limit (per-IP hourly + global daily): block before spending any tokens.
        ip = get_client_ip(request)
        allowed, limit_message = rate_limiter.check(ip)
        if not allowed:
            logger.info("run_review: rate-limited ip=%s (%s)", ip, limit_message)
            yield f"⛔ {limit_message}", gr.skip(), gr.skip(), gr.skip()
            return

        # Input size cap: reject oversized code before a run consumes a rate-limit slot.
        if len(code_input) > CODE_CHAR_LIMIT:
            yield (f"⛔ Your code is {len(code_input)} characters, over the {CODE_CHAR_LIMIT} "
                   "limit. Please shorten it and run again."), gr.skip(), gr.skip(), gr.skip()
            return

        rate_limiter.record(ip)   # the run is going ahead: count it against both limits

        # Clear the prior run's report before streaming, so stale output never sits next to a
        # fresh run while it is still in progress.
        yield "", gr.update(value=REPORT_PLACEHOLDER), gr.update(interactive=False), ""

        # Busy note: if the server is already near capacity, tell the user it may be slower.
        prefix = BUSY_NOTE if _active_runs >= CONCURRENCY_LIMIT - 1 else ""
        _active_runs += 1
        try:
            block_queue = asyncio.Queue()
            task = asyncio.create_task(_review_task(code_input, block_queue))

            console = prefix
            while True:
                block = await block_queue.get()
                if block is _SENTINEL:
                    break
                console += block + "\n\n"
                yield console, gr.skip(), gr.skip(), gr.skip()

            result = await task   # defensive: run_pipeline returns a dict on every path
        finally:
            _active_runs -= 1     # release the slot even if the run raises

        report_md = result.get("review_report_markdown") if isinstance(result, dict) else None
        if not report_md:
            # Failed or empty-result run: the console already carries the failure overview, so leave the report controls inactive rather than offering an empty download.
            return

        # Hold the report in memory for this session; the download handler materialises the file.
        yield gr.skip(), gr.update(value=report_md), gr.update(interactive=True), report_md

    except Exception as e:
        logger.error("run_review failed unexpectedly: %s", str(e))
        yield f"Unexpected error while running the review: {e}", gr.skip(), gr.skip(), gr.skip()


# === REPORT DOWNLOAD ===

_REPORT_TMP_DIR = os.path.join(tempfile.gettempdir(), "code_review_reports")
_REPORT_PREFIX = "review_"        # tags this app's report files so the sweep skips unrelated temp files
_REPORT_MAX_AGE_S = 300           # a downloaded file is used within seconds; older means orphaned


def _cleanup_report_files(previous_path: str | None) -> None:
    """
    Deletes this session's previous report file and any stale report files.

    Pipeline: helper for _prepare_download (this module). Keeps the temp report directory bounded
    on a server that is rarely restarted, where orphaned files from closed sessions would
    otherwise never be removed.

    Args:
        previous_path: The path this session wrote last time; deleted first if it still exists.
    """
    # Best-effort: a file that cannot be removed must never break the download that follows.
    try:
        if previous_path and os.path.isfile(previous_path):
            os.remove(previous_path)

        if not os.path.isdir(_REPORT_TMP_DIR):
            return

        cutoff = time.time() - _REPORT_MAX_AGE_S
        for name in os.listdir(_REPORT_TMP_DIR):
            if not name.startswith(_REPORT_PREFIX):
                continue
            path = os.path.join(_REPORT_TMP_DIR, name)
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
    except Exception as e:
        logger.warning("_cleanup_report_files: could not fully clean temp reports: %s", str(e))


def _prepare_download(report_md: str, previous_path: str | None):
    """
    Writes the current report to a fresh temp file for download and cleans up old ones.

    Pipeline: bound to the download button's click in the Gradio UI (this module). Materialises
    the report held in session state into a file the browser can fetch, removes this session's
    previous file, and sweeps stale files so nothing accumulates on a long-running server.

    Args:
        report_md:     The report markdown from session state.
        previous_path: The temp path written for this session's last download, or None.

    Returns:
        (download_update, new_path): a gr.update carrying the new file path for the button, plus
        the path to store back in session state. On empty/invalid input, a no-op update and None.
    """
    try:
        _cleanup_report_files(previous_path)

        # The button is only enabled after a successful run, so an empty state here means the UI is out of sync: report it rather than writing an empty file.
        if not isinstance(report_md, str) or not report_md.strip():
            logger.warning("_prepare_download: no report in session state, nothing to write")
            return gr.update(), None

        os.makedirs(_REPORT_TMP_DIR, exist_ok=True)
        file_path = os.path.join(_REPORT_TMP_DIR, f"{_REPORT_PREFIX}{uuid.uuid4().hex}.md")
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(report_md)

        return gr.update(value=file_path), file_path

    except Exception as e:
        logger.error("_prepare_download failed unexpectedly: %s", str(e))
        return gr.update(), previous_path   # keep the old path so cleanup can still find it later


# === STYLING ===

custom_css = """
/* Centered header block. */
.header-container {
    text-align: center;
    padding: 1rem 0 0.5rem 0;
}
.header-container h1 {
    margin-bottom: 0.25rem;
}
.tech-stack {
    font-size: 0.9rem;
    opacity: 0.7;
}
.author-line {
    font-size: 0.85rem;
    opacity: 0.6;
    margin-top: 0.5rem;
}

/* Inactive buttons read as translucent rather than solid until they become usable. */
button:disabled {
    opacity: 0.45 !important;
}

/* Keep every toolbar button label on one line. A longer label like "Example code explanation"
   otherwise wraps to two lines and makes that single button taller than the rest. */
.toolbar-row button {
    white-space: nowrap;
    min-width: max-content !important;
    font-size: 1rem;
    padding: 0.6rem 1.5rem;
    min-height: 44px;
    flex-shrink: 0;
}

/* Hide the default Gradio footer. */
footer {
    display: none !important;
}

/* Hold the code editor at a stable height even when empty. Without this the editor
   collapses to a couple of lines until code is entered, then jumps taller once it fills. */
.code-editor .cm-editor {
    min-height: 400px;
}
"""


# === UI ASSEMBLY ===

with gr.Blocks(title="Multi-Agent Code Review") as demo:

    # Per-session stores: the current report markdown, and the last temp file path for cleanup.
    report_state = gr.State(value="")
    download_path_state = gr.State(value=None)

    # Surfaces the stack (RAG + multi-agent) at the top so it is visible without reading code.
    gr.HTML("""
        <div class="header-container">
            <h1>🔍 Multi-Agent Code Review</h1>
            <p class="tech-stack">
                Analyzer → Enricher → Optimizer → Evaluator · RAG (ChromaDB) + MCP tools · LLM-as-Judge
            </p>
            <p class="author-line">
                Built by Dennis Feyerabend ·
                <a href="https://github.com/dfeyerabend/multi-agent-code-review" target="_blank">GitHub ↗</a>
            </p>
        </div>
    """)

    # Run steps and pipeline overview are the first thing a visitor sees;
    with gr.Accordion("How it works", open=False):
        gr.Markdown(INTRO_MD)

    # Runs remaining for this visitor, filled on load and refreshed after each run.
    runs_display = gr.Markdown()

    # The elastic spacer between the buttons keeps each one small and leaves room to add more controls later without a redesign.
    with gr.Row(elem_classes="toolbar-row"):
        load_button = gr.Button("📄 Load example code", size="sm", scale=0)
        explain_button = gr.Button("📖 Example code explanation", size="sm", scale=0)
        report_guide_button = gr.Button("📋 How to read the report", size="sm", scale=0)
        gr.Column(scale=1, min_width=0)                                                             # pushes Run to the opposite edge from the left group
        run_button = gr.Button("▶ Run review", variant="primary", size="sm", scale=0)

    # Live character count above the editor; mirrors the hard limit enforced in run_review.
    char_count = gr.Markdown(f"0 / {CODE_CHAR_LIMIT} characters")

    # gr.Code (not a plain textbox) so pasted code gets syntax highlighting and line numbers.
    code_box = gr.Code(
        label="Python code",
        language="python",
        lines=18,
        elem_classes="code-editor",  # anchor for the min-height rule that stops the empty editor collapsing
    )

    # Static explainers for the bundled example, collapsed so they never crowd the run flow.
    # Opened by the two explainer buttons; each closes again via its own accordion header.
    with gr.Accordion("Example code explanation", open=False) as example_explanation_panel:
        gr.Markdown(EXAMPLE_EXPLANATION_MD)
    with gr.Accordion("How to read the report", open=False) as report_guide_panel:
        gr.Markdown(REPORT_GUIDE_MD)

    # Live console: seeded with the hint, then run_review streams the pipeline's blocks into it.
    console = gr.Markdown(value=CONSOLE_HINT)

    # Report controls start inactive; run_review's final yield fills the body and enables download.
    with gr.Accordion("Full detailed report", open=False):
        report_body = gr.Markdown(value=REPORT_PLACEHOLDER)
    download_button = gr.DownloadButton("Download report (.md)", interactive=False)

    # Fill the editor with the bundled example on demand.
    load_button.click(fn=load_example_code, outputs=code_box).then(
        fn=render_char_count, inputs=code_box, outputs=char_count
    )

    # Open each explainer panel on demand; the panel header handles collapsing.
    explain_button.click(fn=lambda: gr.update(open=True), outputs=example_explanation_panel)
    report_guide_button.click(fn=lambda: gr.update(open=True), outputs=report_guide_panel)

    # Update the character counter as the user types.
    code_box.input(fn=render_char_count, inputs=code_box, outputs=char_count)

    # Disable Run for the duration of a review so a second click cannot start an overlapping run, then re-enable it once the stream finishes.
    run_button.click(
        fn=lambda: gr.update(interactive=False),
        outputs=run_button,
    ).then(
        fn=run_review,
        inputs=code_box,
        outputs=[console, report_body, download_button, report_state],
    ).then(
        fn=render_runs_remaining,
        outputs=runs_display,
    ).then(
        fn=lambda: gr.update(interactive=True),
        outputs=run_button,
    )

    # Build the file and hand it to the browser on click; also refresh the stored temp path.
    download_button.click(
        fn=_prepare_download,
        inputs=[report_state, download_path_state],
        outputs=[download_button, download_path_state],
    )

    # Show the visitor's remaining-run count as soon as the page loads.
    demo.load(fn=render_runs_remaining, outputs=runs_display)


# === GRACEFUL SHUTDOWN ===

def graceful_shutdown(signum, frame) -> None:
    """
    Logs the received signal and exits cleanly.

    Pipeline: registered as the SIGTERM and SIGINT handler in the __main__ block. Lets the host
    platform (or a local Ctrl+C) stop the server without a noisy traceback.

    Args:
        signum: The received signal number.
        frame:  The current stack frame (unused).
    """
    logger.info("Received %s, shutting down.", signal.Signals(signum).name)
    sys.exit(0)


# === ENTRY POINT ===

if __name__ == "__main__":
    setup_logging()

    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)

    # Cap simultaneous runs; extra visitors wait in Gradio's queue rather than all running at once.
    demo.queue(default_concurrency_limit=CONCURRENCY_LIMIT)

    # Hosts such as Railway inject the public port via $PORT; fall back to Gradio's default for local runs. Binding 0.0.0.0 makes the server reachable from outside the container.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, css=custom_css)
