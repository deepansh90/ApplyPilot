"""Job fit scoring: LLM-powered evaluation of candidate-job match quality.

Scores jobs on a 1-10 scale by comparing the user's resume against each
job description. All personal data is loaded at runtime from the user's
profile and resume file.
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

from applypilot.config import RESUME_PATH, load_profile
from applypilot.database import get_connection, get_jobs_by_stage
from applypilot.llm import get_client

log = logging.getLogger(__name__)


# ── Content-addressed score cache ─────────────────────────────────────────
# Keyed by (resume fingerprint, normalized job text) so a job keeps its score
# across DB wipes, re-discovery under a new URL, and re-runs — no repeat LLM
# call. A résumé change changes the fingerprint and invalidates every entry.
# Disable with APPLYPILOT_SCORE_CACHE=0.

_CACHE_PATH = RESUME_PATH.parent / "score_cache.json"
_cache: dict | None = None
_cache_dirty = 0


def _cache_enabled() -> bool:
    return (os.environ.get("APPLYPILOT_SCORE_CACHE", "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            if not isinstance(_cache, dict):
                _cache = {}
        except Exception:
            _cache = {}
    return _cache


def _save_cache(force: bool = False) -> None:
    global _cache_dirty
    if _cache is None or (not force and _cache_dirty < 10):
        return
    try:
        tmp = _CACHE_PATH.with_name(_CACHE_PATH.name + ".tmp")
        tmp.write_text(json.dumps(_cache, ensure_ascii=False, indent=0), encoding="utf-8")
        tmp.replace(_CACHE_PATH)
        _cache_dirty = 0
    except Exception as e:
        log.warning("score cache save failed: %s", e)


def _cache_key(resume_text: str, job_text: str) -> str:
    rf = hashlib.sha256(resume_text.encode("utf-8")).hexdigest()[:12]
    jn = re.sub(r"\s+", " ", job_text.lower()).strip()
    jh = hashlib.sha256(jn.encode("utf-8")).hexdigest()
    return f"{rf}:{jh}"


# ── Scoring Prompt ────────────────────────────────────────────────────────

SCORE_PROMPT = """You are a job fit evaluator. Given a candidate's resume and a job description, score how well the candidate fits the role.

SCORING CRITERIA:
- 9-10: Perfect match. Candidate has direct experience in nearly all required skills and qualifications.
- 7-8: Strong match. Candidate has most required skills, minor gaps easily bridged.
- 5-6: Moderate match. Candidate has some relevant skills but missing key requirements.
- 3-4: Weak match. Significant skill gaps, would need substantial ramp-up.
- 1-2: Poor match. Completely different field or experience level.

IMPORTANT FACTORS:
- Weight technical skills heavily (programming languages, frameworks, tools)
- Consider transferable experience (automation, scripting, API work)
- Factor in the candidate's project experience
- Be realistic about experience level vs. job requirements (years of experience, seniority)

RESPOND IN EXACTLY THIS FORMAT (no other text):
SCORE: [1-10]
KEYWORDS: [comma-separated ATS keywords from the job description that match or could match the candidate]
REASONING: [2-3 sentences explaining the score]"""


def _parse_score_response(response: str) -> dict:
    """Parse the LLM's score response into structured data.

    Args:
        response: Raw LLM response text.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    score = 0
    keywords = ""
    reasoning = response

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                score = int(re.search(r"\d+", line).group())
                score = max(1, min(10, score))
            except (AttributeError, ValueError):
                score = 0
        elif line.startswith("KEYWORDS:"):
            keywords = line.replace("KEYWORDS:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    return {"score": score, "keywords": keywords, "reasoning": reasoning}


def score_job(resume_text: str, job: dict, use_cache: bool = True) -> dict:
    """Score a single job against the resume.

    Args:
        resume_text: The candidate's full resume text.
        job: Job dict with keys: title, site, location, full_description.
        use_cache: Read/write the content-addressed score cache.

    Returns:
        {"score": int, "keywords": str, "reasoning": str, "cached": bool}
    """
    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job['site']}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )

    key = None
    if use_cache and _cache_enabled():
        key = _cache_key(resume_text, job_text)
        hit = _load_cache().get(key)
        if isinstance(hit, dict) and isinstance(hit.get("score"), int) and hit["score"] > 0:
            return {
                "score": hit["score"],
                "keywords": hit.get("keywords", ""),
                "reasoning": hit.get("reasoning", ""),
                "cached": True,
            }

    messages = [
        {"role": "system", "content": SCORE_PROMPT},
        {"role": "user", "content": f"RESUME:\n{resume_text}\n\n---\n\nJOB POSTING:\n{job_text}"},
    ]

    try:
        client = get_client()
        response = client.chat(messages, max_tokens=512, temperature=0.2)
        result = _parse_score_response(response)
    except Exception as e:
        log.error("LLM error scoring job '%s': %s", job.get("title", "?"), e)
        return {"score": 0, "keywords": "", "reasoning": f"LLM error: {e}", "cached": False}

    result["cached"] = False
    if key and result.get("score", 0) > 0:
        global _cache_dirty
        _load_cache()[key] = {
            "score": result["score"],
            "keywords": result.get("keywords", ""),
            "reasoning": result.get("reasoning", ""),
            "title": job.get("title", ""),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _cache_dirty += 1
        _save_cache()
    return result


def run_scoring(limit: int = 0, rescore: bool = False) -> dict:
    """Score unscored jobs that have full descriptions.

    Args:
        limit: Maximum number of jobs to score in this run.
        rescore: If True, re-score all jobs (not just unscored ones).

    Returns:
        {"scored": int, "errors": int, "elapsed": float, "distribution": list}
    """
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    if rescore:
        query = "SELECT * FROM jobs WHERE full_description IS NOT NULL"
        if limit > 0:
            query += f" LIMIT {limit}"
        jobs = conn.execute(query).fetchall()
    else:
        jobs = get_jobs_by_stage(conn=conn, stage="pending_score", limit=limit)

    if not jobs:
        log.info("No unscored jobs with descriptions found.")
        return {"scored": 0, "errors": 0, "elapsed": 0.0, "distribution": []}

    # Convert sqlite3.Row to dicts if needed
    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    _n_cache = 0
    if _cache_enabled() and not rescore:
        log.info("Score cache: %d entries at %s", len(_load_cache()), _CACHE_PATH)
    log.info("Scoring %d jobs sequentially...", len(jobs))
    t0 = time.time()
    completed = 0
    errors = 0
    results: list[dict] = []

    # Persist per job. The previous version buffered every result and committed
    # once at the very end, so any interruption (rate-limit crash, kill, sleep)
    # threw away the whole run and left fit_score NULL — and re-runs re-burned
    # the API quota from scratch. Per-job commits make scoring durable + resumable.
    for job in jobs:
        result = score_job(resume_text, job, use_cache=not rescore)
        result["url"] = job["url"]
        completed += 1

        if result["score"] == 0:
            errors += 1
        if result.get("cached"):
            _n_cache += 1

        results.append(result)

        try:
            conn.execute(
                "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ? WHERE url = ?",
                (result["score"], f"{result['keywords']}\n{result['reasoning']}",
                 datetime.now(timezone.utc).isoformat(), result["url"]),
            )
            if completed % 5 == 0:
                conn.commit()
        except Exception as e:
            log.warning("DB write failed for %s: %s", result["url"], e)

        log.info(
            "[%d/%d] score=%d%s  %s",
            completed, len(jobs), result["score"],
            " (cache)" if result.get("cached") else "",
            job.get("title", "?")[:60],
        )

    conn.commit()
    _save_cache(force=True)

    elapsed = time.time() - t0
    log.info(
        "Done: %d scored in %.1fs (%.1f jobs/sec) — %d from cache, %d LLM calls",
        len(results), elapsed, len(results) / elapsed if elapsed > 0 else 0,
        _n_cache, len(results) - _n_cache,
    )

    # Score distribution
    dist = conn.execute("""
        SELECT fit_score, COUNT(*) FROM jobs
        WHERE fit_score IS NOT NULL
        GROUP BY fit_score ORDER BY fit_score DESC
    """).fetchall()
    distribution = [(row[0], row[1]) for row in dist]

    return {
        "scored": len(results),
        "errors": errors,
        "elapsed": elapsed,
        "distribution": distribution,
    }
