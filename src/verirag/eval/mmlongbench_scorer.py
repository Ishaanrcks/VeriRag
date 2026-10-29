"""MMLongBench-Doc scorer and answer extractor, vendored from upstream.

Source: https://github.com/mayubo2333/MMLongBench-Doc  (``eval/eval_score.py``,
``eval/extract_answer.py``, ``eval/prompt_for_answer_extraction.md`` and the
parsing in ``run_api.py``). Upstream is Apache-2.0 licensed; this is a
modified copy. Copyright the MMLongBench-Doc authors.

The paper's three-stage protocol (section 4.1):
  1. the system produces a free-form response;
  2. an LLM turns that response into a short answer  -> :func:`extract_answer`;
  3. a rule-based scorer compares it to the key       -> :func:`eval_score`.

Changes from upstream (the scoring rules themselves are reproduced unchanged):

* SECURITY: upstream calls ``eval()`` on the List branch (on both the key and
  the prediction, and the prediction is model output) and in ``show_results``.
  Here every such call is :func:`_parse_list`, which uses
  ``ast.literal_eval`` -- literals only, no code execution. If the text is
  not a valid literal (e.g. unquoted items, ``[a, b]``) it falls back to
  splitting the bracket contents on commas rather than raising, which is more
  forgiving than upstream (which would crash the run). ``show_results`` is not
  vendored.
* ``extract_answer`` takes the client and model as parameters. Upstream builds
  ``OpenAI()`` at import time and defaults to ``gpt-4o``.
* The question + analysis are sent in the same user message as the extraction
  prompt. Upstream sends them as a trailing *assistant* message, which not
  every OpenAI-compatible endpoint accepts.
* ``extract_answer`` retries rate limits / transient errors (upstream returns
  "Failed" on the first error, which would score a throttled run as 0).
* The three inline lines in ``run_api.py`` that parse the extractor's reply
  are :func:`parse_extraction`, and a malformed reply yields
  ``"Failed to extract"`` instead of an IndexError (upstream has that
  fallback commented out).
* A List key and prediction that are both empty score 1.0; upstream raises
  IndexError on ``gt[0]``.
* Removed debug ``print`` calls, and upstream's three ``s.rstrip("mile")`` /
  ``"miles"`` / ``"million"`` statements, whose result is discarded (they are
  no-ops, so dropping them does not change behaviour).
"""

from __future__ import annotations

import ast
import re
import time
from collections.abc import Callable
from math import isclose
from pathlib import Path

EXTRACTION_PROMPT_PATH = Path(__file__).with_name("mmlongbench_extraction_prompt.md")

FAILED_TO_EXTRACT = "Failed to extract"


# --------------------------------------------------------------------------
# stage 3: rule-based scoring (upstream eval/eval_score.py)
# --------------------------------------------------------------------------
def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                )
        distances = distances_
    return distances[-1]


def anls_compute(groundtruth: str, prediction: str, threshold: float = 0.5) -> float:
    """Normalised Levenshtein similarity, zeroed at or below ``threshold``."""
    dist = levenshtein_distance(groundtruth, prediction)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    if anls <= threshold:
        anls = 0.0
    return anls


def is_float_equal(
    reference, prediction, include_percentage: bool = False, is_close: bool = False
) -> bool:
    def get_precision(gt_ans: float) -> int:
        precision = 3
        if "." in str(gt_ans):
            precision = len(str(gt_ans).split(".")[-1])
        return precision

    reference = float(str(reference).strip().rstrip("%").strip())
    try:
        prediction = float(str(prediction).strip().rstrip("%").strip())
    except Exception:  # noqa: BLE001 - upstream: unparseable prediction is wrong
        return False

    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        try:
            if is_close:
                if isclose(item, prediction, rel_tol=0.01):
                    return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def get_clean_string(s) -> str:
    s = str(s).lower().strip()
    # remove parenthesis
    s = re.sub(r"\s*\([^)]*\)", "", s).strip()
    # remove quotes
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()
    s = s.strip().lstrip("$").strip()
    s = s.strip().rstrip("%").strip()
    return s


def is_exact_match(s: str) -> bool:
    flag = False
    # Website
    if "https://" in s:
        flag = True
    # code file
    if s.endswith(".py") or s.endswith("ipynb"):
        flag = True
    if s.startswith("page"):
        flag = True
    # telephone number
    if re.fullmatch(r"\b\d+(-\d+|\s\d+)?\b", s):
        flag = True
    # time
    if "a.m." in s or "p.m." in s:
        flag = True
    # YYYY-MM-DD
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}[-\s]\d{2}\b", s):
        flag = True
    # YYYY-MM
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}\b", s):
        flag = True
    # Email address
    if re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", s):
        flag = True
    return flag


def isfloat(num) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _parse_list(value) -> list:
    """Safe replacement for upstream's ``eval()`` on List-format answers.

    ``ast.literal_eval`` only evaluates Python literals, so a payload such as
    ``"[__import__('os').system('...')]"`` raises instead of executing. Text
    that is not a valid literal falls back to a comma split of the bracket
    contents (``"[a, b]"`` -> ``["a", "b"]``).
    """
    if not (isinstance(value, str) and value.startswith("[")):
        return value if isinstance(value, list) else [value]
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        inner = value.strip()[1:-1] if value.rstrip().endswith("]") else value[1:]
        return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
    return parsed if isinstance(parsed, list) else [parsed]


def eval_score(gt, pred, answer_type: str) -> float:
    """Score one prediction against its key, by ``answer_type``.

    Int / Float give 0 or 1; Str / None give an ANLS value (0 or in (0.5, 1]);
    List is all-or-nothing on length, then exact or minimum-ANLS over the
    sorted items. A correct "Not answerable" on an unanswerable question
    (answer_type "None") scores 1.0.
    """
    if answer_type == "Int":
        try:
            gt, pred = int(gt), int(float(pred))
        except Exception:  # noqa: BLE001
            pred = ""
        score = gt == pred
    elif answer_type == "Float":
        try:
            gt = float(get_clean_string(str(gt)))
            pred = float(get_clean_string(str(pred)))
        except Exception:  # noqa: BLE001
            pred = ""
        score = is_float_equal(gt, pred, include_percentage=True, is_close=True)
    elif answer_type in ["Str", "None"]:
        gt = get_clean_string(gt)
        pred = get_clean_string(pred)
        if is_exact_match(gt):
            score = gt == pred
        else:
            score = anls_compute(gt, pred)
    else:
        gt = _parse_list(gt)
        pred = _parse_list(pred)
        if len(gt) != len(pred):
            score = 0.0
        else:
            gt = sorted([get_clean_string(a) for a in gt])
            pred = sorted([get_clean_string(a) for a in pred])
            if not gt:
                score = 1.0  # two empty lists; upstream raises IndexError here
            elif isfloat(gt[0]) or is_exact_match(gt[0]):
                score = "-".join(gt) == "-".join(pred)
            else:
                score = min(anls_compute(g, p) for g, p in zip(gt, pred))

    return float(score)


def eval_acc_and_f1(samples: list[dict]) -> tuple[float, float]:
    """Pooled accuracy and F1 over samples carrying ``score`` / ``answer`` / ``pred``.

    * accuracy = mean ``score`` over every scored sample.
    * recall    = sum(score over answerable-key samples) / (# answerable keys)
    * precision = sum(score over answerable-key samples)
                  / (# samples whose PREDICTION is not "Not answerable")
    * F1 = harmonic mean of the two.

    Counter-intuitive consequences, all inherited from upstream:

    * A correct decline on an unanswerable question scores 1.0 in accuracy but
      adds nothing to the numerator of either recall or precision, so declining
      is never rewarded in F1 directly.
    * A WRONG answer on an unanswerable question (pred != "Not answerable") adds
      to precision's denominator only, so it is penalised through precision.
    * A decline on an answerable question lowers recall and does not touch
      precision's denominator.

    Returns ``(0.0, 0.0)`` for an empty set and ``f1 = 0.0`` whenever the
    ratios are undefined.
    """
    evaluated = [s for s in samples if "score" in s]
    if not evaluated:
        return 0.0, 0.0

    acc = sum(s["score"] for s in evaluated) / len(evaluated)
    try:
        answerable = [s for s in evaluated if s["answer"] != "Not answerable"]
        predicted = [s for s in evaluated if s["pred"] != "Not answerable"]
        recall = sum(s["score"] for s in answerable) / len(answerable)
        precision = sum(s["score"] for s in answerable) / len(predicted)
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0.0 else 0.0
    except ZeroDivisionError:
        f1 = 0.0

    return acc, f1


# --------------------------------------------------------------------------
# stage 2: answer extraction (upstream eval/extract_answer.py + run_api.py)
# --------------------------------------------------------------------------
def load_extraction_prompt(path: str | Path | None = None) -> str:
    return Path(path or EXTRACTION_PROMPT_PATH).read_text()


def make_extraction_client(api_key: str, base_url: str):
    """An OpenAI-SDK client for any OpenAI-compatible endpoint (Gemini, DeepSeek...).

    Imported lazily so merely importing this module never needs the SDK or a key.
    """
    from openai import OpenAI

    # An explicit timeout: the SDK default is 10 minutes, which turns a stalled
    # endpoint into a silent hang.
    return OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)


_RETRY_HINT = re.compile(r"retry in ([0-9.]+)s", re.IGNORECASE)


def _retry_delay(exc: Exception, attempt: int) -> float | None:
    """Seconds to wait before retrying ``exc``, or None if it is not retryable.

    Rate limits (429) and transient server errors (5xx, timeouts) are retried;
    the server's own "retry in 34.0s" hint is honoured when present.
    """
    status = getattr(exc, "status_code", None)
    transient = status in (429, 500, 502, 503, 504) or type(exc).__name__ in (
        "APITimeoutError", "APIConnectionError",
    )
    if not transient:
        return None
    hint = _RETRY_HINT.search(str(exc))
    if hint:
        return float(hint.group(1)) + 1.0
    return min(5.0 * 2**attempt, 60.0)


def extract_answer(
    question: str,
    output: str,
    prompt: str,
    *,
    client,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    max_attempts: int = 6,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Ask ``model`` to extract a short typed answer from a free-form response.

    ``client`` is any OpenAI-compatible client (``client.chat.completions.create``).
    Rate limits and transient server errors are retried (a free-tier Gemini key
    allows 5 requests a minute, so a bare failure here would silently score a
    correct run as 0). Returns the raw reply, or ``"Failed"`` once retries are
    exhausted or on a non-retryable error (upstream behaviour).
    """
    message = f"{prompt}\n\nQuestion:{question}\nAnalysis:{output}\n"
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or "Failed"
        except Exception as exc:  # noqa: BLE001 - upstream swallows everything
            delay = _retry_delay(exc, attempt)
            if delay is None or attempt == max_attempts - 1:
                return "Failed"
            sleep(delay)
    return "Failed"


def parse_extraction(extracted_res: str) -> str:
    """The short answer from ``Extracted answer: X\\nAnswer format: Y``.

    Upstream: ``res.split("Answer format:")[0].split("Extracted answer:")[1].strip()``.
    Returns :data:`FAILED_TO_EXTRACT` when the reply does not follow the format.
    """
    head = (extracted_res or "").split("Answer format:")[0]
    parts = head.split("Extracted answer:")
    if len(parts) < 2:
        return FAILED_TO_EXTRACT
    return parts[1].strip()


def score_response(
    sample: dict,
    response: str,
    extractor: Callable[[str, str], str],
) -> dict:
    """Extract, parse and score one response; returns the sample with results added.

    ``sample`` needs ``question``, ``answer`` and ``answer_format``.
    ``extractor(question, response)`` returns the extractor's raw reply.
    """
    extracted = extractor(sample["question"], response)
    pred = parse_extraction(extracted)
    scored = dict(sample, response=response, extracted_res=extracted, pred=pred)
    scored["score"] = eval_score(sample["answer"], pred, sample["answer_format"])
    return scored
