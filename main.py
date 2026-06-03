#!/usr/bin/env python3
"""
Declario Browser Agent

Fetches task context from the knowledge base, then uses browser-use to
execute the task with a visible browser window.

Usage:
    python main.py --task "Open declarations page on rs.ge" --data '{"username":"me","password":"secret"}'
    python main.py --task "Submit tax declaration" --data-file context.json
"""

import asyncio
import base64
import io
import json
import os
import re
import sys
import argparse
import logging
import inspect
import uuid
from pathlib import Path
from datetime import datetime
from decimal import Decimal, InvalidOperation

# Ensure UTF-8 stdout so we can emit non-ASCII safely on Windows when spawned as subprocess
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import httpx
from dotenv import load_dotenv

_agent_dir = Path(__file__).parent
os.environ.setdefault("BROWSER_USE_CONFIG_DIR", str(_agent_dir / "recordings" / "browseruse-config"))

COMPLETION_READY_FOR_REVIEW = "ready_for_review"
COMPLETION_SUBMITTED = "submitted"
COMPLETION_FAILED = "failed"
# "needs_review" = the submit action fired and the deterministic checks
# (K1 typed-values + DOM postcondition) passed, but the visual validator could
# not positively recognise a confirmation page. The data almost certainly
# reached rs.ge — so this is NOT a failure; it is a submitted-but-unconfirmed
# outcome flagged for a human to glance at, instead of a false red "Failed".
COMPLETION_NEEDS_REVIEW = "needs_review"
COMPLETION_STATES = {
    COMPLETION_READY_FOR_REVIEW,
    COMPLETION_SUBMITTED,
    COMPLETION_FAILED,
    COMPLETION_NEEDS_REVIEW,
}
SAFETY_MODES = {"auto", "halt-on-dangerous", "dry-run"}
AGENT_MODES = {"free", "playbook", "bulk"}
DEFAULT_ALLOWED_DOMAINS = ["rs.ge"]
DEFAULT_SESSION_KEY = "default"
AUTH_DOMAIN_EXCEPTIONS = ["id.gov.ge"]
DANGEROUS_TEXT_RE = re.compile(
    r"submit|send|confirm|pay|delete|remove|sign|finali[sz]e|register|"
    r"გაგზავნა|დადასტურება|გადახდა|წაშლა|ხელმოწერა|წარდგენა",
    re.I,
)

# Load .env — try agent/.env first, then fall back to backend/.env and repo root
for _env_candidate in [
    _agent_dir / ".env",
    _agent_dir.parent / "backend" / ".env",
    _agent_dir.parent / ".env",
]:
    if _env_candidate.exists():
        load_dotenv(_env_candidate)
        break

# ── Shared HTTP client (connection reuse across backend API calls) ────────────
_shared_http: httpx.AsyncClient | None = None

# The final result text from the most recent run_agent() call, captured so
# main() can post an autonomous-task callback (receipt parsing) without
# changing run_agent()'s return signature.
_LAST_FINAL_RESULT: str | None = None


def _backend_auth_headers() -> dict:
    """
    Headers every call from the spawned Python worker to agent-backend must
    carry so it passes through the tenantMiddleware on the Node side:
      X-Internal-Secret  must match AI_INTERNAL_SECRET on the backend
      X-Company-Id       resolved by the spawner from bulk_runs.company_id
    The user-facing rs-client proxy injects the same pair; this is the
    worker-side mirror.
    """
    headers = {}
    secret = os.environ.get("AI_INTERNAL_SECRET", "")
    if secret:
        headers["X-Internal-Secret"] = secret
    company = os.environ.get("AGENT_COMPANY_ID", "")
    if company:
        headers["X-Company-Id"] = company
    return headers


def _get_http() -> httpx.AsyncClient:
    global _shared_http
    if _shared_http is None or _shared_http.is_closed:
        _shared_http = httpx.AsyncClient(timeout=15, headers=_backend_auth_headers())
    return _shared_http


async def _close_http():
    global _shared_http
    if _shared_http is not None and not _shared_http.is_closed:
        await _shared_http.aclose()
        _shared_http = None


# ── LLM setup ─────────────────────────────────────────────────────────────────
# browser-use 0.12.x uses its own native LLM wrappers (not LangChain).

def build_llm():
    """Build the worker LLM (used for per-step actions inside browser-use's loop)."""
    from browser_use.llm.google.chat import ChatGoogle
    # gemini-3.1-flash-lite: latest fast Gemini, no extended thinking so it
    # never hits MAX_TOKENS mid-response. Note: there is NO plain
    # "gemini-3.1-flash" in the API — only -flash-lite or -pro-preview.
    # Override with AGENT_MODEL env var (e.g. gemini-3.1-pro-preview for harder tasks).
    model_name = os.environ.get("AGENT_MODEL", "gemini-3.1-flash-lite")
    return ChatGoogle(
        model=model_name,
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
    )


def build_planner_llm():
    """Build the planner LLM (used for pre-pass + periodic re-planning)."""
    from browser_use.llm.google.chat import ChatGoogle
    model_name = os.environ.get("PLANNER_MODEL", "gemini-3.1-pro-preview")
    return ChatGoogle(
        model=model_name,
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
    )


async def _capture_action_history(
    history,
    playbook_id: str,
    user_data: dict,
    safety_mode: str = "halt-on-dangerous",
    completion_state: str = COMPLETION_READY_FOR_REVIEW,
) -> None:
    """
    After a successful run, dump browser-use's action history to the backend
    cache. Each action records the stable selector (x_path + ax_name + tag)
    plus the typed value re-templated as ${variable} where it matches data.
    Replay code in run_from_cache() reads this and executes deterministically.
    """
    if not playbook_id:
        return
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:3001")

    # Reverse-lookup: literal value -> ${variable_name}
    # Lets us cache "type ${username}" instead of the literal credentials.
    value_to_var: dict[str, str] = {}
    for k, v in (user_data or {}).items():
        if v and isinstance(v, str) and len(v) >= 2:
            value_to_var[v] = f"${{{k}}}"

    cached_actions: list[dict] = []
    idx = 0
    for h in (getattr(history, "history", None) or []):
        model_output = getattr(h, "model_output", None)
        if not model_output or not getattr(model_output, "action", None):
            continue
        actions = model_output.action
        state = getattr(h, "state", None)
        elements = getattr(state, "interacted_element", None) if state else None
        if not isinstance(elements, list):
            elements = [None] * len(actions)
        if len(elements) < len(actions):
            elements = list(elements) + [None] * (len(actions) - len(elements))

        for action_model, el in zip(actions, elements):
            try:
                action_data = action_model.model_dump(exclude_none=True, mode="json")
            except Exception:
                continue
            if not action_data:
                continue
            action_type, action_args = next(iter(action_data.items()))

            cached: dict = {"index": idx, "action": action_type}

            if isinstance(action_args, dict):
                if isinstance(action_args.get("url"), str):
                    cached["url"] = action_args["url"]
                if isinstance(action_args.get("text"), str):
                    text = action_args["text"]
                    cached["value_template"] = value_to_var.get(text, text)

            # interacted_element may be a DOMInteractedElement dataclass, a dict
            # (post-model_dump), or None — handle all three.
            if el is not None:
                el_dict: dict | None = None
                if isinstance(el, dict):
                    el_dict = el
                elif hasattr(el, "to_dict"):
                    try:
                        el_dict = el.to_dict()
                    except Exception:
                        el_dict = None
                if el_dict is None:
                    # Last resort: pull common attributes if present.
                    el_dict = {
                        "x_path": getattr(el, "x_path", None) or getattr(el, "xpath", None),
                        "ax_name": getattr(el, "ax_name", None),
                        "node_name": getattr(el, "node_name", None),
                        "attributes": getattr(el, "attributes", None),
                    }
                cached["x_path"] = el_dict.get("x_path") or el_dict.get("xpath")
                cached["ax_name"] = el_dict.get("ax_name")
                cached["node_name"] = el_dict.get("node_name")
                attrs = el_dict.get("attributes")
                if isinstance(attrs, dict):
                    # Trim to a small whitelist — prevents huge dynamic class blobs.
                    # x_name is a stable custom attribute used on rs.ge form inputs (e.g. COL_2).
                    keep = {"id", "name", "role", "type", "placeholder", "aria-label", "data-testid", "x_name"}
                    cached["attributes"] = {k: v for k, v in attrs.items() if k in keep and v}

            cached_actions.append(cached)
            idx += 1

    if not cached_actions:
        return

    # ── Cache-quality guard ──────────────────────────────────────────────────
    # The agent can self-declare success after going completely off-track
    # (e.g. landed on duckduckgo.com after failing to find rs.ge declarations
    # and called `done` with whatever it last did). Reject the cache if more
    # than half the navigate/url actions left the rs.ge domain — the run
    # was clearly not the workflow we recorded.
    rsge_count = 0
    offsite_count = 0
    for a in cached_actions:
        url = (a.get("url") or "").lower()
        if not url:
            continue
        if "rs.ge" in url:
            rsge_count += 1
        elif url.startswith(("http://", "https://")):
            offsite_count += 1
    if (rsge_count + offsite_count) > 0 and offsite_count > rsge_count:
        _emit(
            "warn",
            f"⚠ Cache rejected: agent left rs.ge ({offsite_count} offsite vs {rsge_count} on-site URLs). "
            "Run was probably off-track despite self-declared success.",
        )
        return

    # Also reject if the run includes any `search` action (DuckDuckGo / web
    # search) — those are escape behaviours, not part of any rs.ge playbook.
    search_actions = sum(1 for a in cached_actions if a.get("action") == "search")
    if search_actions > 0:
        _emit(
            "warn",
            f"⚠ Cache rejected: {search_actions} web-search action(s) detected — agent escaped to a search engine.",
        )
        return

    try:
        client = _get_http()
        resp = await client.post(
            f"{backend_url}/agent/cache",
            json={
                "playbookId": playbook_id,
                "actions": cached_actions,
                "safetyMode": safety_mode if safety_mode in SAFETY_MODES else "halt-on-dangerous",
                "completionState": (
                    completion_state if completion_state in COMPLETION_STATES else COMPLETION_READY_FOR_REVIEW
                ),
                "postconditions": {
                    "requiresSafetyValidator": True,
                    "expectedOutcome": completion_state,
                },
            },
        )
        if resp.status_code == 200:
            _emit("info", f"💾 Cached {len(cached_actions)} actions for replay")
        else:
            _emit("warn", f"Cache save failed: HTTP {resp.status_code} {resp.text[:120]}")
    except Exception as exc:
        _emit("warn", f"Cache save error: {exc}")


async def _locate_element(page, act: dict):
    """
    Try multiple strategies to locate the element captured in a cached action.
    Order: x_path → ax_name (exact text) → id → name → aria-label → placeholder
           → role+name → data-testid → partial text (first 30 chars of ax_name).
    Returns a Playwright Locator or None if nothing matches.
    """
    x_path = act.get("x_path")
    ax_name = act.get("ax_name")
    attrs = act.get("attributes") or {}
    node_name = (act.get("node_name") or "").lower()

    if x_path:
        try:
            loc = page.locator(f"xpath={x_path}").first
            if await loc.count() > 0:
                return loc
        except Exception:
            pass

    # x_name is a stable custom attribute on rs.ge form inputs (e.g. x_name="COL_2").
    # Check it before generic text matching — it's a precise, form-specific identifier.
    x_name_attr = attrs.get("x_name")
    if x_name_attr:
        try:
            safe = x_name_attr.replace('"', '\\"')
            loc = page.locator(f'[x_name="{safe}"]').first
            if await loc.count() > 0:
                return loc
        except Exception:
            pass

    if ax_name:
        try:
            loc = page.get_by_text(ax_name, exact=False).first
            if await loc.count() > 0:
                return loc
        except Exception:
            pass

    el_id = attrs.get("id")
    if el_id:
        try:
            safe = el_id.replace('"', '\\"')
            loc = page.locator(f'[id="{safe}"]').first
            if await loc.count() > 0:
                return loc
        except Exception:
            pass

    el_name = attrs.get("name")
    if el_name:
        try:
            safe = el_name.replace('"', '\\"')
            loc = page.locator(f'[name="{safe}"]').first
            if await loc.count() > 0:
                return loc
        except Exception:
            pass

    aria_label = attrs.get("aria-label")
    if aria_label:
        try:
            safe = aria_label.replace('"', '\\"')
            loc = page.locator(f'[aria-label="{safe}"]').first
            if await loc.count() > 0:
                return loc
        except Exception:
            pass

    placeholder = attrs.get("placeholder")
    if placeholder:
        try:
            safe = placeholder.replace('"', '\\"')
            loc = page.locator(f'[placeholder="{safe}"]').first
            if await loc.count() > 0:
                return loc
        except Exception:
            pass

    data_testid = attrs.get("data-testid")
    if data_testid:
        try:
            safe = data_testid.replace('"', '\\"')
            loc = page.locator(f'[data-testid="{safe}"]').first
            if await loc.count() > 0:
                return loc
        except Exception:
            pass

    # role + accessible name (covers buttons, inputs with ARIA roles)
    el_role = attrs.get("role")
    if el_role and ax_name:
        try:
            loc = page.get_by_role(el_role, name=ax_name).first
            if await loc.count() > 0:
                return loc
        except Exception:
            pass

    # Partial text match on first 30 normalized chars of ax_name
    if ax_name:
        partial = " ".join(ax_name.split())[:30]
        if partial and partial != ax_name.strip():
            try:
                loc = page.get_by_text(partial, exact=False).first
                if await loc.count() > 0:
                    return loc
            except Exception:
                pass

    # tag + role attribute as CSS selector (e.g. input[type="checkbox"])
    el_type = attrs.get("type")
    if node_name and el_type:
        try:
            safe_type = el_type.replace('"', '\\"')
            loc = page.locator(f'{node_name}[type="{safe_type}"]').first
            if await loc.count() > 0:
                return loc
        except Exception:
            pass

    return None


async def _locate_element_with_llm(page, act: dict):
    """
    Last-resort element finder: sends a screenshot + accessibility snapshot to
    a cheap LLM and asks it to return a CSS/XPath selector for the element.
    Returns (locator, selector_str) on success, or (None, None) on failure.
    This is NOT a planning call — it only answers "where is this element?".
    """
    try:
        from google import genai as _genai
        from google.genai import types as _gtypes

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None, None

        ax_name = act.get("ax_name") or ""
        node_name = act.get("node_name") or ""
        attrs = act.get("attributes") or {}
        target_desc = act.get("target_description") or ax_name
        target_text = act.get("target_text") or ax_name
        action_type = act.get("action", "")

        screenshot = await page.screenshot(type="png")

        # Build a compact accessibility tree hint
        try:
            ax_tree = await page.accessibility.snapshot(interesting_only=True)
            import json as _json
            ax_hint = _json.dumps(ax_tree, ensure_ascii=False)[:3000]
        except Exception:
            ax_hint = "(unavailable)"

        prompt = f"""You are a Playwright selector expert.

I need to locate a specific element on a web page. Based on the screenshot and accessibility tree below, return the best CSS or XPath selector.

Element to find:
- action: {action_type}
- target_description: {target_desc}
- visible_text / label: {target_text}
- HTML tag: {node_name}
- attributes hint: {attrs}

Accessibility tree (truncated):
{ax_hint}

Rules:
- Prefer specific selectors (id, name, aria-label, data-testid) over generic ones
- Return xpath=... or a CSS selector
- If the element is not visible on the page, return null
- Reply ONLY as JSON: {{"selector": "...", "found": true}} or {{"selector": null, "found": false}}"""

        client = _genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.environ.get("LOCATOR_MODEL", "gemini-2.0-flash-lite"),
            contents=[
                _gtypes.Part.from_bytes(data=screenshot, mime_type="image/png"),
                prompt,
            ],
            config=_gtypes.GenerateContentConfig(
                temperature=0,
                max_output_tokens=256,
                response_mime_type="application/json",
            ),
        )
        text = (response.text or "").strip()
        import re as _re
        cleaned = _re.sub(r"^```(?:json)?\s*", "", text, flags=_re.I)
        cleaned = _re.sub(r"\s*```$", "", cleaned).strip()
        parsed = json.loads(cleaned)
        selector = parsed.get("selector") if isinstance(parsed, dict) else None
        if not selector:
            return None, None

        # Validate the selector actually matches something
        try:
            loc = page.locator(selector).first
            if await loc.count() > 0:
                return loc, selector
        except Exception:
            pass

        return None, None
    except Exception as exc:
        _emit("warn", f"[locator-llm] failed: {exc}")
        return None, None


async def _execute_cached_action(page, act: dict, user_data: dict) -> bool:
    """
    Execute one cached action via Playwright. Returns True on success.
    If element location fails with all deterministic strategies, falls back to
    _locate_element_with_llm(). On LLM recovery the act dict is mutated with
    the new selector and marked act["_llm_recovered"] = True so the caller can
    refresh the cache.
    """
    action_type = (act.get("action") or "").lower()

    # Substitute ${variable} in value_template using provided data.
    value = act.get("value_template")
    if isinstance(value, str):
        for k, v in (user_data or {}).items():
            value = value.replace(f"${{{k}}}", str(v))

    try:
        if action_type in ("navigate", "go_to_url"):
            url = act.get("url")
            if not url:
                return False
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            return True

        if action_type in ("done",):
            return True  # terminal action — replay succeeds when last step lands

        if action_type in ("wait", "sleep"):
            # Best-effort: wait a fixed 1s if no specific value
            await page.wait_for_timeout(1000)
            return True

        # All remaining actions need to locate an element first
        locator = await _locate_element(page, act)
        if locator is None:
            _emit("warn", f"[replay] deterministic locators failed for step {act.get('index', '?')} ({action_type}), trying LLM locator")
            locator, recovered_selector = await _locate_element_with_llm(page, act)
            if locator is not None and recovered_selector:
                # Mutate act so run_from_cache can persist the healed selector
                act["_llm_recovered"] = True
                act["_recovered_selector"] = recovered_selector
                _emit("info", f"[locator-llm] recovered element with selector: {recovered_selector}")

        if locator is None:
            return False

        if action_type in ("click", "click_element", "click_element_by_index"):
            await locator.click(timeout=10000)
            return True
        if action_type in ("input_text", "type"):
            await locator.fill(value or "", timeout=10000)
            return True
        if action_type in ("send_keys", "press"):
            await locator.press(value or "Enter")
            return True

        # Unknown action type — skip without failing the whole replay.
        _emit("warn", f"[replay] skipping unknown action type: {action_type}")
        return True
    except Exception as exc:
        _emit("warn", f"[replay] action {action_type} failed: {exc}")
        return False


async def _read_operator_command(allowed: list) -> str:
    """
    Block until an operator sends one of the allowed JSON commands via stdin.
    Used during strict playbook replay when a step is blocked and needs human input.
    Returns the action string (e.g. "resume", "skip", "cancel").
    Falls back to "cancel" if stdin closes.
    """
    loop = asyncio.get_event_loop()
    while True:
        try:
            line = await loop.run_in_executor(None, sys.stdin.readline)
        except Exception:
            return "cancel"
        if not line:
            return "cancel"
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except Exception:
            continue
        action = cmd.get("action", "")
        if action in allowed:
            return action


async def run_from_cache(
    playbook_id: str,
    user_data: dict,
    headless: bool = False,
    safety_mode: str = "halt-on-dangerous",
) -> dict:
    """
    Replay a playbook's cached actions deterministically (no AI agent).

    Return values:
      {"success": True,  "completion_state": ...}            — replay succeeded
      {"success": False, "reason": "no_cache"}               — no cache exists yet, safe to record with AI
      {"success": False, "reason": "cancelled"}              — operator cancelled a blocked step
      {"success": False, "reason": "step_failed",
       "step_index": i, "step_action": "..."}                — element not found after all strategies
                                                               (this reason should NOT fall back to free AI)
    """
    if not playbook_id:
        return {"success": False, "reason": "no_cache", "completion_state": COMPLETION_FAILED}

    backend_url = os.environ.get("BACKEND_URL", "http://localhost:3001")

    # 1. Look up the cache via backend (with raw actions array).
    try:
        client = _get_http()
        resp = await client.get(
            f"{backend_url}/agent/cache",
            params={"playbookId": playbook_id, "withActions": "1"},
            timeout=10,
        )
        if resp.status_code != 200:
            return {"success": False, "reason": "no_cache", "completion_state": COMPLETION_FAILED}
        data = resp.json()
        cache = data.get("currentCache")
    except Exception as exc:
        _emit("warn", f"Cache lookup failed: {exc}")
        return {"success": False, "reason": "no_cache", "completion_state": COMPLETION_FAILED}

    if not cache:
        return {"success": False, "reason": "no_cache", "completion_state": COMPLETION_FAILED}

    actions = cache.get("actions") if isinstance(cache, dict) else None
    if not actions:
        return {"success": False, "reason": "no_cache", "completion_state": COMPLETION_FAILED}

    cache_id = cache.get("id")
    cache_safety = cache.get("safetyMode") or cache.get("safety_mode")
    if cache_safety and cache_safety != safety_mode:
        _emit("cache-miss", f"Cached safety mode is {cache_safety}, current run is {safety_mode}")
        return {"success": False, "reason": "no_cache", "completion_state": COMPLETION_FAILED}

    _emit(
        "cache-replay-start",
        f"💾 Replaying {len(actions)} cached actions in strict playbook mode (no AI deviation)",
    )

    # 2. Run actions through a fresh Playwright browser.
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            context = await browser.new_context()
            page = await context.new_page()

            # Work on a mutable copy so we can update selectors for cache refresh
            executed_acts = [dict(a) for a in actions]

            i = 0
            while i < len(executed_acts):
                act = executed_acts[i]
                action_type = act.get("action", "?")
                _emit("step", f"Replay {i+1}/{len(executed_acts)}: {action_type}")

                ok = await _execute_cached_action(page, act, user_data)

                if not ok:
                    # All deterministic + LLM locator strategies failed.
                    # Pause and ask the operator what to do.
                    _emit(
                        "step-blocked",
                        f"⚠️ Playbook step {i+1}/{len(executed_acts)} ({action_type}) could not locate its element. "
                        f"Waiting for operator input: resume (retry) / skip / cancel.",
                        step_index=i,
                        step_action=action_type,
                    )
                    op = await _read_operator_command(["resume", "skip", "cancel"])
                    if op == "resume":
                        _emit("info", f"▶️ Retrying step {i+1} ({action_type})")
                        continue  # retry same step
                    elif op == "skip":
                        _emit("warn", f"⏭️ Skipping step {i+1} ({action_type}) per operator request")
                        i += 1
                        continue
                    else:  # cancel
                        _emit("warn", f"🛑 Operator cancelled playbook replay at step {i+1}")
                        await browser.close()
                        return {
                            "success": False,
                            "reason": "cancelled",
                            "completion_state": COMPLETION_FAILED,
                        }

                # Tiny pause between actions to let the page settle
                await page.wait_for_timeout(300)
                i += 1

            expected_portal = _infer_expected_portal(
                "",
                [{"url": a.get("url") or ""} for a in executed_acts if isinstance(a, dict)],
            )
            dom_ok, review_payload, dom_error = await _run_dom_postcondition(
                page,
                user_data,
                {},
                expected_portal,
            )
            _emit("pre_submit_review", "DOM postcondition review completed.", review=review_payload)
            if not dom_ok:
                _emit("warn", f"Cache replay DOM postcondition failed: {dom_error}")
                await browser.close()
                return {"success": False, "reason": "postcondition_failed", "completion_state": COMPLETION_FAILED}

            screenshot = await page.screenshot(type="png", full_page=True)
            _emit("info", "Running cache replay postcondition validator...")
            visual = await _visual_validator(screenshot, user_data, "cache replay completed", safety_mode)
            ok, completion_state, error = _visual_outcome_for_safety(safety_mode, visual)
            if not ok:
                _emit("warn", f"Cache replay postcondition failed: {error}")
                await browser.close()
                return {"success": False, "reason": "postcondition_failed", "completion_state": COMPLETION_FAILED}

            await browser.close()

            # Self-heal: if any steps used the LLM locator, refresh the cache
            # with the new selectors so future runs don't need LLM either.
            recovered = [a for a in executed_acts if a.get("_llm_recovered")]
            if recovered:
                _emit("info", f"🔧 Self-healing cache: updating {len(recovered)} stale selector(s)")
                try:
                    # Strip internal tracking keys before saving
                    clean_acts = []
                    for a in executed_acts:
                        ca = {k: v for k, v in a.items() if not k.startswith("_")}
                        # Promote recovered selector to x_path for next run
                        if a.get("_recovered_selector"):
                            ca["x_path"] = a["_recovered_selector"]
                        clean_acts.append(ca)
                    async with httpx.AsyncClient(timeout=10, headers=_backend_auth_headers()) as client:
                        await client.post(
                            f"{backend_url}/agent/cache",
                            json={
                                "playbookId": playbook_id,
                                "actions": clean_acts,
                                "safetyMode": safety_mode,
                                "completionState": completion_state,
                            },
                        )
                    _emit("info", f"Cache self-healed ({len(recovered)} selector(s) updated)")
                except Exception as exc:
                    _emit("warn", f"Cache self-heal failed (non-fatal): {exc}")
            else:
                _emit("info", f"Cache replay completed successfully ({completion_state}, zero LLM calls)")

            # Bump success_count + last_used_at on the cache row.
            if cache_id:
                try:
                    await _get_http().post(
                        f"{backend_url}/agent/cache/hit",
                        json={"cacheId": cache_id, "completionState": completion_state},
                        timeout=5,
                    )
                except Exception:
                    pass  # non-fatal

            return {"success": True, "completion_state": completion_state}
    except Exception as exc:
        _emit("warn", f"Cache replay aborted: {exc}")
        return {"success": False, "reason": "no_cache", "completion_state": COMPLETION_FAILED}


async def plan_with_planner(
    playbook_steps: list[dict],
    merged_data: dict,
    task_label: str | None,
    knowledge_block: str = "",
    data_contract: dict | None = None,
) -> str:
    """
    Pre-pass: a stronger LLM analyzes the playbook and writes a tactical
    execution plan that augments the worker's task prompt. Returns plain text.
    Returns empty string on failure (worker still runs without plan).

    `knowledge_block` is an optional pre-formatted excerpt from the accounting
    knowledge base — when provided, the planner uses it to disambiguate steps
    that the playbook describes only generically (e.g. "select declaration
    type" without naming which type).
    """
    if not playbook_steps:
        return ""
    try:
        from browser_use.llm.messages import UserMessage
        planner = build_planner_llm()

        steps_text = "\n".join(
            f"{i+1}. {(s.get('action') or '?').upper():9s} | "
            f"{s.get('target_description') or ''} | "
            f"text='{s.get('target_text') or ''}' | "
            f"value='{s.get('value') or ''}' | "
            f"url='{s.get('url') or ''}'"
            for i, s in enumerate(playbook_steps)
        )

        contract = data_contract or _build_authoritative_data_contract(merged_data)
        var_keys = _format_authoritative_data_summary(contract, max_items=10)

        knowledge_section = (
            f"\nACCOUNTING KNOWLEDGE (excerpts from the user's knowledge base — refer to these when the playbook is ambiguous about WHICH option, code, or value to use):\n{knowledge_block}\n"
            if knowledge_block else ""
        )

        planner_prompt = f"""You are a TACTICAL PLANNER for a browser automation worker that executes Georgian-tax-portal (rs.ge) playbooks step by step.

Your only job: read the playbook below, then produce a SHORT tactical plan (max 30 lines, plain text — no JSON, no markdown headers) that the worker LLM will read at the start of every step. The plan should help the worker stay oriented and recover from common failures.

TASK LABEL: {task_label or "(none)"}
{knowledge_section}
PLAYBOOK ({len(playbook_steps)} steps):
{steps_text}

AUTHORITATIVE SPREADSHEET CONTRACT:
{var_keys}

The plan MUST contain these sections (label each clearly):

CHECKPOINTS:
3-6 milestones. Each = "After step N: <observable state>". Examples: "After step 5: logged in, dashboard visible", "After step 18: VAT form fully open with empty inputs".

LIKELY-FAILURE STEPS:
2-4 specific steps that will probably need extra care. Name the step number and why. Example: "Step 7 (select Declaration Type): rs.ge uses CUSTOM React dropdown — must CLICK dropdown then CLICK option, never select_option."

RECOVERY HEURISTICS:
3-5 short rules. Examples: "If clicking same element 3+ times: scroll, re-scan, try a sibling element instead", "If page shows 'change password' popup: click OK to dismiss before continuing playbook". WHEN the ACCOUNTING KNOWLEDGE section above contains a concrete answer for an ambiguous field (e.g. which dropdown option, which code, which period), QUOTE that answer here so the worker can use it. Never override spreadsheet values with KB/memory values.

PROGRESS-TRACKING:
1-2 lines on how the worker should report its progress (e.g. emit step number).

Be terse, concrete, and action-focused. Do NOT restate the entire playbook."""

        response = await planner.ainvoke([UserMessage(content=planner_prompt)])
        text = getattr(response, "completion", None)
        if not text:
            return ""
        return text.strip()
    except Exception as exc:
        # Planner failures are non-fatal — worker still runs without the plan.
        _emit("warn", f"Planner pre-pass failed: {exc}. Continuing without plan.")
        return ""


# ── Context fetch ──────────────────────────────────────────────────────────────

_TASK_KIND_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"ყოველთვიური|monthly\s+vat|vat\s+monthly|დღგ", re.I), "vat_monthly"),
    (re.compile(r"წლიური|annual\s+vat|vat\s+annual", re.I), "vat_annual"),
    (re.compile(r"income|შემოსავლის?|საშემოსავლო", re.I), "income"),
    (re.compile(r"property|ქონების?|ქონებრივი", re.I), "property"),
    (re.compile(r"salary|payroll|ხელფასი|ხელფასები", re.I), "payroll"),
    (re.compile(r"profit|მოგების", re.I), "profit"),
]


def _infer_task_kind(task: str) -> str:
    """Detect a coarse task kind from the user's prompt. Used to softly bias
    the RAG search toward books tagged with the same topic (R3)."""
    if not task:
        return ""
    for pattern, kind in _TASK_KIND_PATTERNS:
        if pattern.search(task):
            return kind
    return ""


async def fetch_context(task: str, limit: int = 12) -> list[dict]:
    """Fetch relevant knowledge-base chunks from the backend.

    Detects a task kind heuristically (vat_monthly / income / property / …)
    and passes it as `taskKind` so the backend can softly prefer books
    tagged for that kind. Falls back to plain similarity search when no
    kind is detected.
    """
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:3001")
    task_kind = _infer_task_kind(task)
    params: dict = {"task": task, "limit": limit}
    if task_kind:
        params["taskKind"] = task_kind
    try:
        client = _get_http()
        resp = await client.get(f"{backend_url}/agent/context", params=params)
        if resp.status_code == 200:
            return resp.json().get("chunks", [])
    except Exception as e:
        _emit("warn", f"Could not reach backend for context: {e}")
    return []


def format_context_for_prompt(chunks: list[dict]) -> str:
    """Format DB chunks into a human-readable instruction block."""
    if not chunks:
        return ""
    lines = ["--- INSTRUCTIONS FROM KNOWLEDGE BASE ---"]
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        title = chunk.get("book_title") or chunk.get("source_title") or meta.get("title")
        start = meta.get("startTime")
        end = meta.get("endTime")
        screenshot = meta.get("screenshotUrl")
        time_tag = f" [{start:.1f}s–{end:.1f}s]" if start is not None else ""
        source_tag = f" ({title})" if title else ""
        lines.append(f"\n•{source_tag} {chunk['content']}{time_tag}")
        if screenshot:
            lines.append(f"  [Screenshot reference: {screenshot}]")
    lines.append("--- END INSTRUCTIONS ---")
    return "\n".join(lines)


def _check_typed_values(history, user_data: dict) -> list[str]:
    """K1 anti-hallucination check.

    Walk the agent action history and return a list of expected non-credential
    values from `user_data` that were never typed by the agent. Empty list
    means all expected values appeared at least once in the typed actions.

    Used by both `run_agent()` (single) and `run_bulk()` (per-row) to flip
    a self-declared 'success' to 'failed' when the agent hallucinated.
    """
    if not user_data:
        return []

    credential_re = re.compile(
        r"password|token|otp|secret|pin|username|^user$|email|login",
        re.I,
    )
    expected_values: list[str] = []
    for k, v in user_data.items():
        if credential_re.search(k):
            continue
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        # Skip zeros and JSON-array placeholders ("[]") — they represent
        # "do not fill", so we don't expect them in typed text.
        if s in ("0", "0.0", "0.00", "[]", "{}"):
            continue
        # Normalise simple numerics: "150.00" → "150" for matching.
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            try:
                f = float(s)
                s = str(int(f)) if f == int(f) else str(f)
            except Exception:
                pass
        # Require ≥ 2 chars to avoid trivial '0' / '1' false positives.
        if len(s) >= 2:
            expected_values.append(s)

    if not expected_values:
        return []

    typed_blob = ""
    try:
        for h in (getattr(history, "history", None) or []):
            mo = getattr(h, "model_output", None)
            if not mo or not getattr(mo, "action", None):
                continue
            for a in mo.action:
                try:
                    d = a.model_dump(exclude_none=True, mode="json")
                except Exception:
                    continue
                if not isinstance(d, dict):
                    continue
                for v in d.values():
                    if isinstance(v, dict):
                        for fld in ("text", "value", "url", "keys"):
                            s = v.get(fld)
                            if isinstance(s, str):
                                typed_blob += s + "\n"
                    elif isinstance(v, str):
                        typed_blob += v + "\n"
    except Exception:
        # If we can't walk history, don't penalise — assume all values typed.
        return []

    typed_lower = typed_blob.lower()
    return [v for v in expected_values if v.lower() not in typed_lower]


def _typed_blob_from_history(history) -> str:
    typed_blob = ""
    try:
        for h in (getattr(history, "history", None) or []):
            mo = getattr(h, "model_output", None)
            if not mo or not getattr(mo, "action", None):
                continue
            for a in mo.action:
                try:
                    d = a.model_dump(exclude_none=True, mode="json")
                except Exception:
                    continue
                if not isinstance(d, dict):
                    continue
                for v in d.values():
                    if isinstance(v, dict):
                        for fld in ("text", "value", "url", "keys"):
                            s = v.get(fld)
                            if isinstance(s, str):
                                typed_blob += s + "\n"
                    elif isinstance(v, str):
                        typed_blob += v + "\n"
    except Exception:
        return ""
    return typed_blob


def _check_authoritative_contract_coverage(history, contract: dict) -> list[dict]:
    typed_blob = _typed_blob_from_history(history).lower()
    if not typed_blob:
        return [
            {
                "key": item.get("key"),
                "display_value": item.get("display_value"),
                "reason": "required value never appeared in typed action history",
            }
            for item in _required_contract_items(contract)
        ]
    missing: list[dict] = []
    for item in _required_contract_items(contract):
        value = str(item.get("value") or "").strip()
        if not value:
            continue
        candidates = _numeric_write_candidates(value)
        if not candidates:
            candidates = [value]
        if any(str(candidate).lower() in typed_blob for candidate in candidates):
            continue
        missing.append({
            "key": item.get("key"),
            "display_value": item.get("display_value"),
            "reason": "required value never appeared in typed action history",
        })
    return missing


def _check_row_coverage(history, contract: dict) -> list[dict]:
    """Anti-hallucination gate for tabular row data (payroll employees).

    Every employee's personal_id is UNIQUE, so requiring each one to appear in
    what the agent actually typed makes it impossible to fake "done" while
    skipping a person — if employee X's id was never typed, X wasn't entered.
    Returns the rows whose identifier never showed up in the typed history.
    """
    row_groups = contract.get("row_groups") or []
    if not row_groups:
        return []
    typed_blob = _typed_blob_from_history(history).lower()
    id_keys = ("personal_id", "pid", "tin", "id_number", "personal_number")
    missing: list[dict] = []
    for group in row_groups:
        for idx, row in enumerate(group.get("rows") or [], 1):
            id_val = ""
            for k in id_keys:
                v = row.get(k)
                if v not in (None, ""):
                    id_val = str(v).strip()
                    break
            label = id_val or str(row.get("name") or f"row {idx}")
            # If we have no identifier to check against, we can't verify this
            # row deterministically — flag it so it's never a silent success.
            if not id_val:
                missing.append({"row": idx, "label": label, "reason": "row has no identifier to verify"})
                continue
            if id_val.lower() not in typed_blob:
                missing.append({"row": idx, "label": label, "reason": "employee identifier never typed — row not entered"})
    return missing


def _check_required_contract_items_verified(contract: dict, typed_log, dom_review: dict | None) -> list[dict]:
    dom_entries = (dom_review or {}).get("fields") or []
    missing: list[dict] = []
    for item in _required_contract_items(contract):
        key = str(item.get("key") or "")
        dom_match = next((entry for entry in dom_entries if str(entry.get("key") or "") == key), None)
        if dom_match and bool(dom_match.get("matched")):
            continue
        missing.append({
            "key": key,
            "display_value": item.get("display_value"),
            "reason": (
                "required spreadsheet item was not DOM-verified"
                if dom_match is not None
                else "required spreadsheet item has no DOM review entry"
            ),
        })
    return missing


def _default_completion_state(safety_mode: str) -> str:
    return COMPLETION_SUBMITTED if safety_mode == "auto" else COMPLETION_READY_FOR_REVIEW


def _safe_session_key(key: object = DEFAULT_SESSION_KEY) -> str:
    raw = str(key or DEFAULT_SESSION_KEY).strip().lower()
    cleaned = re.sub(r"[^a-z0-9_.-]+", "_", raw)[:80].strip("._-")
    return cleaned or DEFAULT_SESSION_KEY


async def _call_maybe_async(value):
    if inspect.iscoroutinefunction(value):
        return await value()
    if callable(value):
        result = value()
        if inspect.isawaitable(result):
            return await result
        return result
    return value


def _page_supports_playwright_locators(page) -> bool:
    return page is not None and (hasattr(page, "locator") or hasattr(page, "keyboard"))


def _page_supports_actor_press(page) -> bool:
    return page is not None and hasattr(page, "evaluate") and hasattr(page, "press") and not _page_supports_playwright_locators(page)


def _page_kind(page) -> str:
    if page is None:
        return "none"
    if _page_supports_playwright_locators(page):
        return "playwright_page"
    if _page_supports_actor_press(page):
        return "actor_page"
    return "unknown_page"


async def _safe_page_eval(page, js: str, *args):
    if page is None or not hasattr(page, "evaluate"):
        return None
    result = await page.evaluate(js, *args)
    if isinstance(result, str):
        stripped = result.strip()
        if stripped and stripped[0] in "[{" and stripped[-1] in "]}":
            try:
                return json.loads(stripped)
            except Exception:
                return result
    return result


async def _resolve_session_page(session_or_agent):
    if session_or_agent is None:
        return None
    session = getattr(session_or_agent, "browser_session", session_or_agent)
    for attr_name in ("get_current_page", "current_page", "page"):
        attr = getattr(session, attr_name, None)
        if attr is None:
            continue
        try:
            page = await _call_maybe_async(attr)
            if page is not None:
                return page
        except Exception:
            continue
    return None


def _session_root_ready(session) -> bool:
    if session is None:
        return False
    return getattr(session, "_cdp_client_root", None) is not None


def _can_export_storage_state(session) -> tuple[bool, str]:
    if session is None:
        return False, "session missing"
    if not hasattr(session, "export_storage_state"):
        return False, "storage export unsupported"
    if not _session_root_ready(session):
        return False, "root CDP client unavailable"
    if getattr(session, "is_closed", False):
        return False, "browser session already closed"
    if getattr(session, "agent_focus_target_id", None) is None:
        return False, "agent focus target unavailable"
    return True, ""


async def _export_storage_state_if_possible(session, state_path: Path | str | None, *, context: str = "") -> bool:
    ok, reason = _can_export_storage_state(session)
    if not ok:
        _emit(
            "info",
            f"storage_state export skipped{f' ({context})' if context else ''}: {reason}",
            export_context=context or "default",
            export_skipped_reason=reason,
        )
        return False
    if not state_path:
        return False
    try:
        await session.export_storage_state(state_path)
        _emit("info", f"Saved session storage_state to {state_path}")
        return True
    except Exception as exc:
        msg = str(exc)
        if "CDP client not initialized" in msg or "Root CDP client not initialized" in msg:
            _emit(
                "info",
                f"storage_state export skipped{f' ({context})' if context else ''}: {msg}",
                export_context=context or "default",
                export_skipped_reason=msg,
            )
            return False
        raise


def _coerce_allowed_domains(value: object = None) -> list[str]:
    if value is None or value == "":
        items: list[object] = DEFAULT_ALLOWED_DOMAINS
    elif isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = DEFAULT_ALLOWED_DOMAINS

    domains: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        domain = item.strip().lower()
        if not domain:
            continue
        domain = re.sub(r"^https?://", "", domain)
        domain = domain.split("/")[0].split(":")[0]
        if domain and domain not in domains:
            domains.append(domain)

    if not domains:
        domains = list(DEFAULT_ALLOWED_DOMAINS)

    expanded: list[str] = []
    def add(domain: str) -> None:
        if domain and domain not in expanded:
            expanded.append(domain)

    for domain in domains:
        add(domain)
        if not domain.startswith("*.") and "*" not in domain:
            add(f"*.{domain}")
        if domain in DEFAULT_ALLOWED_DOMAINS or domain.endswith(".rs.ge"):
            for auth_domain in AUTH_DOMAIN_EXCEPTIONS:
                add(auth_domain)
                add(f"*.{auth_domain}")

    return expanded


def _domains_for_prompt(allowed_domains: list[str]) -> str:
    visible = [d for d in allowed_domains if not d.startswith("*.")]
    return ", ".join(visible or allowed_domains or DEFAULT_ALLOWED_DOMAINS)


def _freeform_policy_block(safety_mode: str, allowed_domains: list[str]) -> str:
    domain_text = _domains_for_prompt(allowed_domains)
    final_policy = (
        "You may click final submit/send/pay/confirm/register/delete controls only after all validators pass."
        if safety_mode == "auto"
        else "Do NOT click final submit/send/pay/confirm/register/delete controls. Stop with the page ready for human review."
    )
    if safety_mode == "dry-run":
        final_policy = "Dry run: do not perform any irreversible action; stop before the final action and summarize what would happen."
    # In halt/dry-run modes the Worker historically called done(success=False)
    # because "the task isn't fully done — the human still has to submit". This
    # was wrong: the safe halt IS the success criterion in those modes. Spell
    # that out so K1/M1 don't see a self-declared failure on a perfectly good run.
    success_criteria = (
        "When the form is filled and you have stopped before the irreversible "
        "submit/send/confirm step, call done(success=True) with a summary like "
        "'Form filled and ready for human review.' Stopping cleanly IS success here. "
        "Do NOT call done(success=False) just because submit was not clicked."
        if safety_mode in ("halt-on-dangerous", "dry-run")
        else ""
    )
    return "\n".join(filter(None, [
        "=== FREE-MODE BROWSER POLICY ===",
        f"- Stay inside these allowed domains only: {domain_text}.",
        "- Do not use web search or search-engine escape routes.",
        "- If a link, redirect, or button would leave the allowed domains, stop and report it.",
        f"- {final_policy}",
        f"- {success_criteria}" if success_criteria else "",
        "=== END FREE-MODE BROWSER POLICY ===",
    ]))


def _node_policy_text(node) -> str:
    pieces: list[str] = []
    for attr_name in ("tag_name", "xpath"):
        value = getattr(node, attr_name, None)
        if value:
            pieces.append(str(value))
    attrs = getattr(node, "attributes", None)
    if isinstance(attrs, dict):
        for key in ("id", "name", "role", "type", "aria-label", "title", "value", "class", "placeholder"):
            value = attrs.get(key)
            if value:
                pieces.append(str(value))
    for method_name in ("get_all_children_text", "get_meaningful_text_for_llm"):
        method = getattr(node, method_name, None)
        if callable(method):
            try:
                value = method()
                if value:
                    pieces.append(str(value))
            except Exception:
                pass
    for attr_name in ("text", "inner_text"):
        value = getattr(node, attr_name, None)
        if value:
            pieces.append(str(value))
    return " ".join(pieces)[:2000]


def _is_dangerous_click_text(text: str) -> bool:
    return bool(DANGEROUS_TEXT_RE.search(text or ""))


def _visual_outcome_for_safety(safety_mode: str, visual_result: dict) -> tuple[bool, str, str | None]:
    zero_fields = visual_result.get("suspicious_zero_fields") or []
    if zero_fields:
        return False, COMPLETION_FAILED, f"Visual validator: fields show 0.00: {zero_fields[:5]}"

    if safety_mode in ("halt-on-dangerous", "dry-run") and visual_result.get("irreversible_action_executed"):
        explanation = str(visual_result.get("explanation", ""))[:160]
        return False, COMPLETION_FAILED, f"Visual validator: irreversible action executed in {safety_mode}. {explanation}"

    if visual_result.get("is_confirmation_page"):
        if safety_mode in ("halt-on-dangerous", "dry-run"):
            return False, COMPLETION_FAILED, f"Visual validator: confirmation page reached in {safety_mode}."
        return True, COMPLETION_SUBMITTED, None

    explanation = str(visual_result.get("explanation", ""))[:160]
    if safety_mode == "auto":
        return False, COMPLETION_FAILED, f"Visual validator: not a confirmation page. {explanation}"

    if safety_mode in ("halt-on-dangerous", "dry-run") and (
        visual_result.get("is_ready_for_review") or visual_result.get("final_action_visible")
    ):
        return True, COMPLETION_READY_FOR_REVIEW, None

    return False, COMPLETION_FAILED, f"Visual validator: not ready for review. {explanation}"


_CREDENTIAL_KEY_RE = re.compile(
    r"password|token|otp|secret|pin|username|^user$|email|login",
    re.I,
)
_OPTIONAL_META_KEY_RE = re.compile(
    r"^period$|^company(_name)?$|^notes$|^note$|^comment$|^description$",
    re.I,
)
_FINANCIAL_KEY_RE = re.compile(
    r"amount|sum|total|tax|vat|turnover|balance|fee|price|თანხ|დღგ|გადასახად|ბრუნვ|ჯამ",
    re.I,
)
_PLACEHOLDER_VALUE_SET = {"[]", "{}"}


def _decimal_from_text(value: object) -> Decimal | None:
    text = str("" if value is None else value).strip()
    if not text:
        return None
    cleaned = re.sub(r"[^\d,.\-]", "", text)
    if not cleaned or cleaned in {"-", ".", ","}:
        return None

    last_dot = cleaned.rfind(".")
    last_comma = cleaned.rfind(",")
    if last_dot >= 0 and last_comma >= 0:
        decimal_sep = "." if last_dot > last_comma else ","
        thousands_sep = "," if decimal_sep == "." else "."
        cleaned = cleaned.replace(thousands_sep, "")
        cleaned = cleaned.replace(decimal_sep, ".")
    elif last_comma >= 0:
        parts = cleaned.split(",")
        cleaned = "".join(parts[:-1]) + "." + parts[-1] if len(parts[-1]) in (1, 2) else "".join(parts)
    elif last_dot >= 0:
        parts = cleaned.split(".")
        cleaned = "".join(parts[:-1]) + "." + parts[-1] if len(parts[-1]) in (1, 2) else "".join(parts)

    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


def _is_zero_like(value: object) -> bool:
    dec = _decimal_from_text(value)
    return dec is not None and dec == 0


def _values_equivalent(expected: object, actual: object) -> bool:
    exp = str("" if expected is None else expected).strip()
    got = str("" if actual is None else actual).strip()
    if not exp and not got:
        return True
    exp_dec = _decimal_from_text(exp)
    got_dec = _decimal_from_text(got)
    if exp_dec is not None and got_dec is not None:
        return exp_dec == got_dec
    exp_digits = re.sub(r"\D", "", exp)
    got_digits = re.sub(r"\D", "", got)
    if len(exp_digits) >= 6 and exp_digits == got_digits:
        return True
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    return norm(exp) == norm(got)


def _numeric_write_candidates(value: object) -> list[str]:
    raw = str("" if value is None else value).strip()
    if not raw:
        return []
    dec = _decimal_from_text(raw)
    if dec is None:
        return [raw]
    normalized = format(dec, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".") or "0"
    candidates: list[str] = []
    for candidate in (
        raw,
        normalized,
        f"{dec:.2f}",
        f"{dec:.2f}".replace(".", ","),
    ):
        c = str(candidate).strip()
        if c and c not in candidates:
            candidates.append(c)
    return candidates


def _reviewable_data_items(user_data: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for key, value in (user_data or {}).items():
        if _CREDENTIAL_KEY_RE.search(str(key)):
            continue
        text = str("" if value is None else value).strip()
        if not text or text in {"[]", "{}"}:
            continue
        out.append((str(key), text))
    return out


def _build_authoritative_data_contract(
    user_data: dict,
    field_map: dict[str, str] | None = None,
    task_hint: str = "",
) -> dict:
    field_map = field_map or {}
    items: list[dict] = []
    # Tabular values (e.g. payroll `lines`: a list of per-employee dicts) are
    # NOT scalar form fields — they describe N rows to enter one-by-one. Pull
    # them out so they don't pollute the scalar contract, and surface them as
    # a dedicated "rows to enter" block the agent loops over.
    row_groups: list[dict] = []
    required_count = 0
    optional_count = 0
    zero_like_count = 0
    mapped_count = 0
    sensitive_count = 0
    task_lower = (task_hint or "").lower()
    form_like_task = any(token in task_lower for token in ("fill", "submit", "declaration", "tax", "rs.ge", "ფორმ"))
    for raw_key, raw_value in (user_data or {}).items():
        if isinstance(raw_value, list) and raw_value and all(isinstance(r, dict) for r in raw_value):
            row_groups.append({"key": str(raw_key), "rows": raw_value})
            continue
        key = str(raw_key)
        value = "" if raw_value is None else str(raw_value).strip()
        is_sensitive = bool(_CREDENTIAL_KEY_RE.search(key))
        is_empty = not value
        is_placeholder = value in _PLACEHOLDER_VALUE_SET
        is_zero_like = _is_zero_like(value)
        field_label = str(field_map.get(key) or "").strip()
        has_field_map_label = bool(field_label)
        required_for_fill = False
        reason_if_optional = ""

        if is_sensitive:
            required_for_fill = not is_empty
            if not required_for_fill:
                reason_if_optional = "Sensitive field is empty."
        elif is_empty:
            reason_if_optional = "Value is empty."
        elif is_placeholder:
            reason_if_optional = "Placeholder collection value is not fillable."
        elif has_field_map_label:
            required_for_fill = True
        elif _OPTIONAL_META_KEY_RE.search(key):
            reason_if_optional = "Metadata field is not typically filled into the portal."
        elif is_zero_like and not (_FINANCIAL_KEY_RE.search(key) or form_like_task):
            reason_if_optional = "Zero-like value without field map was classified as optional."
        else:
            required_for_fill = True

        item = {
            "key": key,
            "value": value,
            "display_value": "***" if is_sensitive and value else value,
            "is_sensitive": is_sensitive,
            "is_empty": is_empty,
            "is_placeholder": is_placeholder,
            "is_zero_like": is_zero_like,
            "has_field_map_label": has_field_map_label,
            "field_label": field_label,
            "required_for_fill": required_for_fill,
            "reason_if_optional": reason_if_optional,
        }
        items.append(item)
        if required_for_fill:
            required_count += 1
        else:
            optional_count += 1
        if is_zero_like:
            zero_like_count += 1
        if has_field_map_label:
            mapped_count += 1
        if is_sensitive:
            sensitive_count += 1

    return {
        "items": items,
        "row_groups": row_groups,
        "summary": {
            "total": len(items),
            "required": required_count,
            "optional": optional_count,
            "zero_like": zero_like_count,
            "mapped": mapped_count,
            "unmapped": max(0, len(items) - mapped_count),
            "sensitive": sensitive_count,
            "row_groups": len(row_groups),
            "rows": sum(len(g.get("rows") or []) for g in row_groups),
        },
    }


def _format_row_groups_block(contract: dict) -> str:
    """Render tabular row data (e.g. payroll employees) as an explicit
    one-by-one entry instruction. Empty when there are no row groups, so
    scalar-only (VAT) runs are unaffected."""
    row_groups = contract.get("row_groups") or []
    out: list[str] = []
    # Keys that are bookkeeping, not values to type per row.
    skip = {"employee_id", "pension_participant"}
    for group in row_groups:
        rows = group.get("rows") or []
        if not rows:
            continue
        out.append(f"=== ROWS TO ENTER ONE-BY-ONE — {len(rows)} ({group.get('key')}) ===")
        out.append(
            "For EACH row below, repeat the per-row sub-flow (add a new line / "
            "person and fill its fields). Enter EVERY row — do not skip or stop "
            "early. After ALL rows are entered, submit the declaration ONCE."
        )
        for idx, row in enumerate(rows, 1):
            parts = []
            for k, v in row.items():
                if k in skip or v is None or str(v).strip() == "":
                    continue
                parts.append(f"{k}={v}")
            out.append(f"  Row {idx}: " + ", ".join(parts))
        out.append("=== END ROWS ===")
    return "\n".join(out)


def _required_contract_items(contract: dict) -> list[dict]:
    return [item for item in (contract.get("items") or []) if item.get("required_for_fill")]


def _format_authoritative_data_block(contract: dict) -> str:
    items = contract.get("items") or []
    rows_block = _format_row_groups_block(contract)
    if not items:
        # Scalar-free but may still have rows to enter (rare).
        return rows_block

    required = [item for item in items if item.get("required_for_fill") and not item.get("is_sensitive")]
    sensitive_required = [item for item in items if item.get("required_for_fill") and item.get("is_sensitive")]
    optional = [item for item in items if not item.get("required_for_fill")]

    lines = [
        "=== AUTHORITATIVE SPREADSHEET DATA ===",
        "These spreadsheet values are the source of truth for this run.",
        "Never invent replacements. Never use remembered values instead.",
        "If a required spreadsheet field cannot be mapped confidently to a form field, STOP and report it.",
    ]
    if required:
        lines.append("-- REQUIRED VALUES TO ENTER --")
        for item in required:
            label = f' -> "{item["field_label"]}"' if item.get("has_field_map_label") else ""
            lines.append(f'  {item["key"]}{label} -> {item["display_value"]}')
    if sensitive_required:
        lines.append("-- SENSITIVE EXECUTION-ONLY VALUES --")
        for item in sensitive_required:
            label = f' -> "{item["field_label"]}"' if item.get("has_field_map_label") else ""
            lines.append(f'  {item["key"]}{label} -> {item["display_value"]}')
    if optional:
        lines.append("-- OPTIONAL / IGNORED VALUES --")
        for item in optional[:20]:
            reason = f' ({item["reason_if_optional"]})' if item.get("reason_if_optional") else ""
            label = f' -> "{item["field_label"]}"' if item.get("has_field_map_label") else ""
            lines.append(f'  {item["key"]}{label} -> {item["display_value"] or "∅"}{reason}')
        if len(optional) > 20:
            lines.append(f"  ... and {len(optional) - 20} more optional/ignored value(s)")
    lines.append("=== END AUTHORITATIVE SPREADSHEET DATA ===")
    if rows_block:
        lines.append(rows_block)
    return "\n".join(lines)


def _format_authoritative_data_summary(contract: dict, *, max_items: int = 8) -> str:
    summary = contract.get("summary") or {}
    required = _required_contract_items(contract)
    lines = [
        "=== AUTHORITATIVE DATA SUMMARY ===",
        (
            f"required={summary.get('required', 0)}, optional={summary.get('optional', 0)}, "
            f"mapped={summary.get('mapped', 0)}, zero_like={summary.get('zero_like', 0)}"
        ),
    ]
    for item in required[:max_items]:
        label = f' -> "{item["field_label"]}"' if item.get("has_field_map_label") else ""
        lines.append(f'  {item["key"]}{label} -> {item["display_value"]}')
    if len(required) > max_items:
        lines.append(f"  ... and {len(required) - max_items} more required value(s)")
    if summary.get("rows"):
        lines.append(
            f"PLUS {summary.get('rows')} row(s) to enter ONE-BY-ONE (see ROWS TO ENTER "
            f"block) — repeat the per-row sub-flow for each, then submit once."
        )
    lines.append("Spreadsheet contract outranks site memory and KB heuristics.")
    lines.append("If knowledge or memory conflicts with spreadsheet values, keep the spreadsheet value and report the conflict.")
    lines.append("=== END AUTHORITATIVE DATA SUMMARY ===")
    return "\n".join(lines)


def _emit_contract_summary(event_prefix: str, contract: dict) -> None:
    summary = contract.get("summary") or {}
    block_len = len(_format_authoritative_data_block(contract))
    _emit(
        "info",
        (
            f"{event_prefix}: contract total={summary.get('total', 0)} "
            f"required={summary.get('required', 0)} optional={summary.get('optional', 0)} "
            f"zero_like={summary.get('zero_like', 0)} mapped={summary.get('mapped', 0)} "
            f"unmapped={summary.get('unmapped', 0)} block_chars={block_len}"
        ),
        contract_total=summary.get("total", 0),
        contract_required=summary.get("required", 0),
        contract_optional=summary.get("optional", 0),
        contract_zero_like=summary.get("zero_like", 0),
        contract_mapped=summary.get("mapped", 0),
        contract_unmapped=summary.get("unmapped", 0),
        contract_block_chars=block_len,
    )


def _compact_label(field: dict) -> str:
    parts = [
        field.get("label"),
        field.get("ariaLabel"),
        field.get("placeholder"),
        field.get("name"),
        field.get("id"),
    ]
    return " ".join(str(p).strip() for p in parts if str(p or "").strip())


def _label_contains(haystack: str, needle: str) -> bool:
    h = re.sub(r"\s+", " ", haystack).strip().lower()
    n = re.sub(r"\s+", " ", needle).strip().lower()
    return bool(n) and (n in h or h in n)


async def _get_current_page_from_agent(agent):
    return await _resolve_session_page(agent)


async def _extract_dom_fields(page) -> dict:
    """Read current form controls directly from DOM for deterministic review."""
    if page is None:
        return {"url": "", "title": "", "fields": [], "error": "page unavailable"}
    try:
        result = await _safe_page_eval(
            page,
            """() => {
              const clean = (s) => (s || '').replace(/\\s+/g, ' ').trim();
              const labelFor = (el) => {
                const id = el.id;
                if (id) {
                  try {
                    const byFor = document.querySelector(`label[for="${CSS.escape(id)}"]`);
                    if (byFor) return clean(byFor.innerText || byFor.textContent);
                  } catch {}
                }
                const parentLabel = el.closest('label');
                if (parentLabel) return clean(parentLabel.innerText || parentLabel.textContent);
                const container = el.closest('tr, .form-group, .field, .row, div');
                if (container) {
                  const label = container.querySelector('label, .label, [class*="label"], [class*="Label"]');
                  if (label) return clean(label.innerText || label.textContent);
                }
                return '';
              };
              return {
                url: window.location.href,
                title: document.title || '',
                fields: Array.from(document.querySelectorAll('input, textarea, select')).map((el, idx) => {
                  const tag = el.tagName.toLowerCase();
                  const type = (el.getAttribute('type') || tag).toLowerCase();
                  const option = tag === 'select' ? el.options[el.selectedIndex] : null;
                  return {
                    index: idx,
                    tag,
                    type,
                    id: el.id || '',
                    name: el.getAttribute('name') || '',
                    label: labelFor(el),
                    placeholder: el.getAttribute('placeholder') || '',
                    ariaLabel: el.getAttribute('aria-label') || '',
                    value: tag === 'select' ? clean(option ? option.textContent : el.value) : String(el.value || ''),
                    checked: type === 'checkbox' || type === 'radio' ? !!el.checked : null,
                    disabled: !!el.disabled,
                    readonly: !!el.readOnly,
                  };
                }).filter((f) => f.type !== 'hidden' && f.type !== 'password')
              };
            }"""
        )
        return result if isinstance(result, dict) else {"url": "", "title": "", "fields": [], "error": "unexpected page eval result"}
    except Exception as exc:
        return {"url": "", "title": "", "fields": [], "error": str(exc)}


async def _validate_typed_log_against_dom(tools, page) -> tuple[list[dict], list[dict]]:
    """S8 trust gate: walk the input-action log and check the live DOM.

    Returns (matched, mismatched). A mismatch means Worker called the input
    action with text X for an xpath, but the DOM at that xpath now shows
    something different. This catches the ExtJS-style "value snaps back to
    0.00" failure mode that R1 misses when the selector_map index has gone
    stale by the time we read.
    """
    log = getattr(tools, "_declario_typed_log", None) or []
    if not log or page is None:
        return [], []
    # Deduplicate to the LAST call per xpath — Worker may have retyped, so
    # only the final write counts.
    last_per_xpath: dict[str, dict] = {}
    for entry in log:
        xp = (entry.get("xpath") or "").strip()
        if not xp:
            continue
        last_per_xpath[xp] = entry
    matched: list[dict] = []
    mismatched: list[dict] = []
    for xpath, entry in last_per_xpath.items():
        wanted = str(entry.get("text") or "").strip()
        if not wanted:
            continue
        try:
            actual = await _safe_page_eval(
                page,
                """(xp) => {
                    const r = document.evaluate(xp, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                    const root = r && r.singleNodeValue;
                    if (!root) return null;
                    const tag = (root.tagName || '').toLowerCase();
                    let target = root;
                    if (tag !== 'input' && tag !== 'textarea') {
                        const inner = root.querySelector(
                            'input:not([type="hidden"]):not([type="button"]):not([type="submit"]), textarea'
                        );
                        if (inner) target = inner;
                    }
                    if (target.tagName === 'SELECT') {
                        const opt = target.options[target.selectedIndex];
                        return opt ? (opt.textContent || target.value || '') : (target.value || '');
                    }
                    return target.value != null ? String(target.value) : '';
                }""",
                xpath,
            )
        except Exception:
            actual = None
        actual_str = "" if actual is None else str(actual).strip()
        record = {
            "index": entry.get("index"),
            "xpath": xpath,
            "expected": wanted,
            "actual": actual_str,
        }
        if actual is None:
            # Node disappeared from DOM — count as mismatch so the run is
            # flagged for human review rather than silently passing.
            mismatched.append({**record, "reason": "node not found in current DOM"})
        elif _values_equivalent(actual_str, wanted):
            matched.append(record)
        else:
            mismatched.append({**record, "reason": "DOM value differs from typed text"})
    return matched, mismatched


async def _extract_dom_fields_via_session(session, page) -> dict:
    """Enumerate form fields through browser-use's selector_map.

    Unlike `_extract_dom_fields`, this pierces shadow DOMs and iframes — the
    selector_map walks the full DOM tree the agent already used to click and
    type. Critical for SPA portals (rs.ge ExtJS) where top-level
    querySelectorAll returns nothing.
    """
    if session is None or page is None:
        return {"url": "", "title": "", "fields": [], "error": "session unavailable"}
    smap = getattr(session, "_cached_selector_map", None) or {}
    if not smap:
        return {"url": "", "title": "", "fields": [], "error": "no selector_map"}
    try:
        url = str(await _safe_page_eval(page, "() => window.location.href") or "")
    except Exception:
        url = ""
    try:
        title = str(await _safe_page_eval(page, "() => document.title || ''") or "")
    except Exception:
        title = ""
    fields: list[dict] = []
    for idx, node in smap.items():
        try:
            tag = (getattr(node, "node_name", "") or "").lower()
            if tag not in ("input", "textarea", "select"):
                continue
            attrs = getattr(node, "attributes", {}) or {}
            type_ = (attrs.get("type") or tag).lower()
            if type_ in ("hidden", "password"):
                continue
            xpath = ""
            try:
                xpath = node.xpath or ""
            except Exception:
                xpath = ""
            if not xpath:
                continue
            value: str = ""
            try:
                raw = await _safe_page_eval(
                    page,
                    """(xp) => {
                        const r = document.evaluate(xp, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                        const el = r && r.singleNodeValue;
                        if (!el) return null;
                        if (el.tagName === 'SELECT') {
                            const opt = el.options[el.selectedIndex];
                            return opt ? (opt.textContent || el.value || '') : (el.value || '');
                        }
                        return el.value != null ? String(el.value) : '';
                    }""",
                    xpath,
                )
                value = "" if raw is None else str(raw)
            except Exception:
                value = ""
            ax = getattr(node, "ax_node", None)
            ax_name = getattr(ax, "name", "") if ax is not None else ""
            label = (
                str(ax_name or "")
                or attrs.get("aria-label", "")
                or attrs.get("name", "")
                or attrs.get("placeholder", "")
                or ""
            )
            fields.append({
                "index": idx,
                "tag": tag,
                "type": type_,
                "id": attrs.get("id", ""),
                "name": attrs.get("name", ""),
                "label": label,
                "placeholder": attrs.get("placeholder", ""),
                "ariaLabel": attrs.get("aria-label", ""),
                "value": value,
                "checked": None,
                "disabled": False,
                "readonly": False,
            })
        except Exception:
            continue
    return {"url": url or "", "title": title or "", "fields": fields}


def _infer_expected_portal(task: str = "", steps: list[dict] | None = None) -> str | None:
    haystack = [task or ""]
    for step in steps or []:
        haystack.append(str(step.get("url") or ""))
        haystack.append(str(step.get("target_description") or ""))
    text = " ".join(haystack).lower()
    if "rs.ge" in text or "eservices.rs.ge" in text or "vat" in text or "დღგ" in text:
        return "rs.ge"
    match = re.search(r"https?://([^/\s)]+)", text)
    return match.group(1) if match else None


def _portal_guard(current_url: str, expected_portal: str | None) -> dict:
    url = (current_url or "").lower()
    signals: list[str] = []
    ok = True
    if expected_portal and expected_portal.lower() not in url:
        ok = False
        signals.append(f"Expected portal {expected_portal}, current URL is {current_url or '(unknown)'}")
    if LOGIN_URL_RE.search(url):
        ok = False
        signals.append("Current page still looks like login/authentication")
    if any(bad in url for bad in ("duckduckgo.", "google.", "bing.")):
        ok = False
        signals.append("Agent appears to have escaped to a search engine")
    return {"ok": ok, "signals": signals}


def _build_postcondition_review(
    dom_snapshot: dict,
    user_data: dict,
    field_map: dict[str, str] | None = None,
    expected_portal: str | None = None,
    contract: dict | None = None,
) -> dict:
    fields = dom_snapshot.get("fields") if isinstance(dom_snapshot, dict) else []
    fields = fields if isinstance(fields, list) else []
    field_map = field_map or {}
    contract = contract or _build_authoritative_data_contract(user_data, field_map)
    entries: list[dict] = []
    errors: list[str] = []
    warnings: list[str] = []

    for item in contract.get("items") or []:
        key = str(item.get("key") or "")
        expected = str(item.get("value") or "").strip()
        if not expected or item.get("is_placeholder"):
            continue
        mapped_label = field_map.get(key, "")
        candidates = []
        if mapped_label:
            candidates = [
                f for f in fields
                if _label_contains(_compact_label(f), mapped_label)
            ]
        if not candidates:
            candidates = [
                f for f in fields
                if _label_contains(_compact_label(f), key)
            ]
        actual_field = next(
            (f for f in candidates if _values_equivalent(expected, f.get("value", ""))),
            candidates[0] if candidates else None,
        )
        if actual_field is None and not _is_zero_like(expected):
            actual_field = next(
                (f for f in fields if _values_equivalent(expected, f.get("value", ""))),
                None,
            )

        actual = str((actual_field or {}).get("value", "")).strip()
        label = _compact_label(actual_field or {}) or mapped_label or key
        matched = actual_field is not None and _values_equivalent(expected, actual)
        severity = "ok"
        reason = ""
        required_for_fill = bool(item.get("required_for_fill"))
        if actual_field is None:
            if not required_for_fill:
                severity = "warn"
                reason = str(item.get("reason_if_optional") or "Optional field was not located in DOM.")
                warnings.append(f"{key}: optional field not located")
            else:
                severity = "error"
                reason = "Expected value was not found in any visible form control."
                errors.append(f"{key}: expected value not found")
        elif not matched:
            severity = "error" if required_for_fill else "warn"
            if not _is_zero_like(expected) and _is_zero_like(actual):
                reason = "DOM value is zero but expected value is non-zero."
            else:
                reason = "DOM value does not match expected value."
            if required_for_fill:
                errors.append(f"{key}: expected {expected!r}, got {actual!r}")
            else:
                warnings.append(f"{key}: expected {expected!r}, got {actual!r}")

        entries.append({
            "key": key,
            "expected": expected,
            "actual": actual,
            "label": label,
            "matched": matched,
            "severity": severity,
            "reason": reason,
            "source": "mapped_label" if mapped_label else "value_search",
            "required_for_fill": required_for_fill,
        })

    portal = _portal_guard(str(dom_snapshot.get("url", "")), expected_portal)
    if not portal["ok"]:
        errors.extend(portal["signals"])

    return {
        "url": dom_snapshot.get("url", ""),
        "title": dom_snapshot.get("title", ""),
        "portal": portal,
        "fields": entries,
        "summary": {
            "ok": len(errors) == 0,
            "errorCount": len(errors),
            "warningCount": len(warnings),
            "checkedFieldCount": len(entries),
        },
        "errors": errors,
        "warnings": warnings,
        "domError": dom_snapshot.get("error"),
    }


async def _run_dom_postcondition(
    page,
    user_data: dict,
    field_map: dict[str, str] | None,
    expected_portal: str | None,
    contract: dict | None = None,
    fallback_snapshot: dict | None = None,
) -> tuple[bool, dict, str | None]:
    """Run DOM postcondition. If the live page snapshot has no URL/fields (the
    page already closed or navigated away — common in halt-on-dangerous after
    Worker calls done()), fall back to the last snapshot captured during the
    run (T1.1). That keeps postcondition from rejecting otherwise-valid runs
    just because the read happened a beat late."""
    dom_snapshot = await _extract_dom_fields(page)
    if (
        fallback_snapshot
        and (not dom_snapshot.get("url") or not dom_snapshot.get("fields"))
    ):
        _emit(
            "info",
            "DOM postcondition: live page unreadable, using last in-run snapshot.",
        )
        dom_snapshot = fallback_snapshot
    review = _build_postcondition_review(dom_snapshot, user_data, field_map, expected_portal, contract)
    ok = bool(review.get("summary", {}).get("ok"))
    if not ok:
        errors = review.get("errors") or []
        return False, review, "; ".join(str(e) for e in errors[:5]) or "DOM postcondition failed"
    return True, review, None


async def _visual_validator(
    screenshot_data,
    user_data: dict,
    agent_final_message: str,
    safety_mode: str = "auto",
) -> dict:
    """One Gemini Vision call: verify the final page matches the requested safety mode.

    Returns a dict with:
      is_confirmation_page: bool
      is_ready_for_review: bool
      final_action_visible: bool
      irreversible_action_executed: bool
      suspicious_zero_fields: list[str]
      reg_number: str | None
      explanation: str
    """
    try:
        from google import genai as _genai
        from google.genai import types as _gtypes

        raw: bytes
        if isinstance(screenshot_data, str):
            raw = base64.b64decode(screenshot_data)
        elif isinstance(screenshot_data, bytes):
            raw = screenshot_data
        else:
            return {"is_confirmation_page": False, "is_ready_for_review": False}

        cred_re = re.compile(
            r"password|token|otp|secret|pin|username|^user$|email|login|^period$|^company",
            re.I,
        )
        expected = {
            k: str(v).strip()
            for k, v in user_data.items()
            if not cred_re.search(k) and str(v).strip() not in ("", "0", "0.0", "0.00", "[]", "{}")
        }
        summary = "\n".join(f"  {k}: {v}" for k, v in list(expected.items())[:20])

        mode_instruction = {
            "auto": "AUTO mode: the agent was allowed to click final submit/send. Require a real confirmation or receipt page.",
            "halt-on-dangerous": "HUMAN REVIEW mode: the agent should fill the form and stop before the final irreversible submit/send. A filled form with the final submit/send button visible is success.",
            "dry-run": "DRY RUN mode: the agent should reach the intended pre-submit stage without an irreversible action. A form/pre-submit stage is acceptable.",
        }.get(safety_mode, "AUTO mode: require a real confirmation or receipt page.")

        prompt = f"""You are a Georgian VAT declaration verifier for rs.ge.
A browser agent just finished a VAT declaration workflow on rs.ge.
Safety mode: {safety_mode}
Mode rule: {mode_instruction}
Agent final message: "{agent_final_message[:300]}"

Expected values that should have been entered:
{summary or "  (none provided)"}

Look at this screenshot and answer these questions:
1. is_confirmation_page: Does this look like a CONFIRMATION or RECEIPT page?
   A confirmation page shows a registration/confirmation number (e.g. "N12345", "01Q68..."),
   or Georgian text like "წარდგენილი" / "registered" / "submission successful".
   A FORM page that still has fillable input boxes is NOT a confirmation — it means the form was NOT submitted.
2. suspicious_zero_fields: List the LABEL TEXT of any visible numeric input fields that show "0.00" or "0"
   when a non-zero value from the expected list was supposed to be entered. Empty list if none.
3. reg_number: If a registration or confirmation number is clearly visible on the page, extract it exactly. Otherwise null.
4. is_ready_for_review: In halt-on-dangerous or dry-run mode, is this a filled/pre-submit page where a human can review before final submission?
5. final_action_visible: Is a final irreversible button visible, such as submit/send/confirm/register/pay?
6. irreversible_action_executed: Does the page show evidence that the final irreversible action was already executed?

Reply ONLY as JSON (no markdown fences):
{{"is_confirmation_page": false, "is_ready_for_review": true, "final_action_visible": true, "irreversible_action_executed": false, "suspicious_zero_fields": [], "reg_number": null, "explanation": "one sentence"}}"""

        client = _genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model=os.environ.get("VALIDATOR_MODEL", "gemini-3.1-flash-lite"),
            contents=[
                _gtypes.Part.from_bytes(data=raw, mime_type="image/png"),
                prompt,
            ],
            config=_gtypes.GenerateContentConfig(
                temperature=0,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )
        text = (response.text or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            return {"is_confirmation_page": False, "is_ready_for_review": False}
        return {
            "is_confirmation_page": bool(parsed.get("is_confirmation_page", False)),
            "is_ready_for_review": bool(parsed.get("is_ready_for_review", False)),
            "final_action_visible": bool(parsed.get("final_action_visible", False)),
            "irreversible_action_executed": bool(parsed.get("irreversible_action_executed", False)),
            "suspicious_zero_fields": parsed.get("suspicious_zero_fields") or [],
            "reg_number": parsed.get("reg_number") or None,
            "explanation": str(parsed.get("explanation", "")),
        }
    except Exception as exc:
        raise RuntimeError(f"_visual_validator: {exc}") from exc


# ── P1: Vision form auto-discovery ─────────────────────────────────────────────
# When the Worker lands on a new form page, it can call discover_form_fields()
# (registered below as a custom browser-use action) to get a JSON map of every
# input the page actually contains. This eliminates the "guess and check" loop
# that dominates free-form runs where no playbook exists.

_field_discovery_cache: dict[str, dict] = {}


def _field_cache_get(url_key: str) -> dict | None:
    cached = _field_discovery_cache.get(url_key)
    return cached if isinstance(cached, dict) else None


def _field_cache_put(url_key: str, fields: list[dict], source: str, status: str = "ok", reason: str = "") -> None:
    _field_discovery_cache[url_key] = {
        "status": status,
        "source": source,
        "reason": reason,
        "fields": fields,
    }


async def _vision_discover_fields(screenshot_data, current_url: str = "") -> list[dict]:
    """Single Gemini Vision call: list every form field visible on the page.

    Returns a list of dicts:
        {label_ge, label_en, type, css_hint, page_section}

    Returns [] on any error so the worker can keep going (vision is best-effort).
    """
    try:
        from google import genai as _genai
        from google.genai import types as _gtypes

        if isinstance(screenshot_data, str):
            raw = base64.b64decode(screenshot_data)
        elif isinstance(screenshot_data, bytes):
            raw = screenshot_data
        else:
            return []

        prompt = """You are looking at a Georgian-language web form (likely on rs.ge).
List every visible form input the user can fill in: <input>, <select>, <textarea>, custom-React-dropdown.
For each, return:
  - label_ge: the visible Georgian label/caption next to the field (exact text)
  - label_en: a short English meaning of the label
  - type: one of "number", "text", "date", "select", "checkbox", "radio"
  - css_hint: a CSS-ish hint that would help find the input in DOM — id, name, aria-label, or "near label '<text>'" if no attribute is visible
  - page_section: a short label for the section the field belongs to (e.g. "Annex A part 1", "Login", "Final calculations")

Skip read-only / disabled / display-only labels. Skip purely decorative buttons.
If no form is visible, return [].

Reply ONLY as a JSON array (no markdown fences):
[{"label_ge": "...", "label_en": "...", "type": "number", "css_hint": "...", "page_section": "..."}]
"""

        client = _genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model=os.environ.get("VALIDATOR_MODEL", "gemini-3.1-flash-lite"),
            contents=[
                _gtypes.Part.from_bytes(data=raw, mime_type="image/png"),
                prompt,
            ],
            config=_gtypes.GenerateContentConfig(
                temperature=0,
                max_output_tokens=2048,
                response_mime_type="application/json",
            ),
        )
        text = (response.text or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            return []
        # Light validation: keep only dicts with at least a label_ge
        out = []
        for item in parsed:
            if isinstance(item, dict) and item.get("label_ge"):
                out.append({
                    "label_ge": str(item.get("label_ge", ""))[:200],
                    "label_en": str(item.get("label_en", ""))[:120],
                    "type": str(item.get("type", "text"))[:20],
                    "css_hint": str(item.get("css_hint", ""))[:200],
                    "page_section": str(item.get("page_section", ""))[:80],
                })
        return out
    except Exception as exc:
        # Vision is best-effort — never block the run on discovery failure
        return []


# ── B: Vision site-scout (post-login navigation map) ──────────────────────────
# Per-domain cache so a single dashboard scout result is reused across rows
# in the same bulk run (and across re-runs within one process).
_nav_scout_cache: dict[str, list[dict]] = {}


async def _vision_scout_navigation(screenshot_data, task_hint: str = "") -> list[dict]:
    """Single Gemini Vision call: enumerate the site's primary navigation
    (sidebar, top-bar, tabs, dropdowns visible at the moment) so the Worker
    knows where to click before it gets lost in scroll loops.

    Returns a list of dicts:
        {label_ge, label_en, kind, parent_label, hint}
    where kind ∈ {"top_menu", "sidebar", "tab", "dropdown_item", "submenu"}
    and hint is a short description of where it likely leads.

    Returns [] on any error.
    """
    try:
        from google import genai as _genai
        from google.genai import types as _gtypes

        if isinstance(screenshot_data, str):
            raw = base64.b64decode(screenshot_data)
        elif isinstance(screenshot_data, bytes):
            raw = screenshot_data
        else:
            return []

        hint_clause = f'\nThe user wants to: "{task_hint[:300]}".\nWhen you can guess from labels, mark which menu item likely leads to that goal in the "hint" field.\n' if task_hint else ""

        prompt = f"""You are looking at a logged-in dashboard or portal page (likely rs.ge — Georgia's revenue service).
Build a navigation map: list every clickable menu item, sidebar link, top-bar tab, and visible dropdown that helps the user reach a feature.{hint_clause}
For each item, return:
  - label_ge: the visible Georgian label (exact text; for top-level menus the icon caption too)
  - label_en: short English meaning
  - kind: one of "top_menu", "sidebar", "tab", "dropdown_item", "submenu", "card_link"
  - parent_label: the label of the parent menu if this is a submenu/dropdown_item, else null
  - hint: a 1-line guess about what this leads to

Skip purely decorative buttons (notifications bell, language switch, profile dropdown) UNLESS they're the only way to reach a feature.
Prioritise menu items that look related to "declarations" / "VAT" / "დღგ" / "დეკლარაცია" / "შემოსავალი".

Reply ONLY as a JSON array (no markdown fences):
[{{"label_ge": "...", "label_en": "...", "kind": "sidebar", "parent_label": null, "hint": "..."}}]
"""

        client = _genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model=os.environ.get("VALIDATOR_MODEL", "gemini-3.1-flash-lite"),
            contents=[
                _gtypes.Part.from_bytes(data=raw, mime_type="image/png"),
                prompt,
            ],
            config=_gtypes.GenerateContentConfig(
                temperature=0,
                max_output_tokens=2048,
                response_mime_type="application/json",
            ),
        )
        text = (response.text or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            return []
        out = []
        for item in parsed:
            if isinstance(item, dict) and item.get("label_ge"):
                out.append({
                    "label_ge": str(item.get("label_ge", ""))[:200],
                    "label_en": str(item.get("label_en", ""))[:120],
                    "kind": str(item.get("kind", "sidebar"))[:20],
                    "parent_label": (str(item.get("parent_label"))[:120]
                                     if item.get("parent_label") else None),
                    "hint": str(item.get("hint", ""))[:200],
                })
        return out
    except Exception:
        return []


def _format_navigation_for_worker(items: list[dict], url_short: str) -> str:
    """Render the nav-scout result into the compact text block injected
    into the Worker's tool-result context."""
    if not items:
        return f"(no navigation items detected on {url_short})"
    lines = [f"=== SITE NAVIGATION MAP on {url_short} ({len(items)} items) ==="]
    by_kind: dict[str, list[dict]] = {}
    for it in items:
        by_kind.setdefault(it.get("kind") or "menu", []).append(it)
    for kind, group in by_kind.items():
        lines.append(f"-- {kind} --")
        for it in group[:50]:
            parent = f" (under '{it['parent_label']}')" if it.get("parent_label") else ""
            hint = f" — {it['hint']}" if it.get("hint") else ""
            lines.append(f"  • '{it['label_ge']}' ({it['label_en']}){parent}{hint}")
    lines.append("=== END NAVIGATION MAP ===")
    return "\n".join(lines)


def _format_discovery_for_worker(fields: list[dict], url_short: str) -> str:
    """Format a field-discovery result into a compact text block the Worker
    can read in its `extracted_content` channel."""
    if not fields:
        return f"(no fillable fields detected on {url_short})"
    lines = [f"=== DISCOVERED FIELDS on {url_short} ({len(fields)} fields) ==="]
    by_section: dict[str, list[dict]] = {}
    for f in fields:
        by_section.setdefault(f.get("page_section") or "Section", []).append(f)
    for section, group in by_section.items():
        lines.append(f"-- {section} --")
        for f in group[:60]:
            lines.append(
                f"  • [{f['type']}] '{f['label_ge']}' ({f['label_en']}) — {f['css_hint']}"
            )
    lines.append("=== END DISCOVERED FIELDS ===")
    return "\n".join(lines)


def _build_tools(
    task_hint: str = "",
    safety_mode: str = "halt-on-dangerous",
    allowed_domains: list[str] | None = None,
    mode: str = "free",
):
    """Construct the browser-use Tools instance with our custom actions registered.

    Registers:
      - discover_form_fields: P1 — vision-grounded form mapping (per-page)
      - discover_navigation: B — vision-grounded site nav map (post-login dashboard)

    `task_hint` is the user's natural-language goal; the nav scout passes it
    to Vision so the model can flag which menu likely leads to the goal.
    """
    from browser_use import Tools, ActionResult
    from browser_use.tools.views import ClickElementActionIndexOnly, InputTextAction
    from browser_use.browser.events import TypeTextEvent, ClickElementEvent

    tools = Tools()
    policy_domains = _coerce_allowed_domains(allowed_domains)
    recent_tool_action: dict = {"signature": None, "count": 0}

    def _tool_repeat_guard(action: str, key: object, limit: int = 3) -> str | None:
        """Block obvious no-progress loops before they spend more agent steps."""
        signature = (action, repr(key))
        if recent_tool_action["signature"] == signature:
            recent_tool_action["count"] += 1
        else:
            recent_tool_action["signature"] = signature
            recent_tool_action["count"] = 1

        count = int(recent_tool_action["count"])
        if count == limit - 1:
            _emit(
                "warn",
                f"S4 repeat-action warning: {action} {key!r} repeated {count}x with no intervening progress.",
                action=action,
                repeat_count=count,
            )
            return None
        if count >= limit:
            msg = (
                f"S4 action-dedup: blocked repeat {action} {key!r} after {count} consecutive attempts. "
                "The page is not progressing. Take a fresh screenshot/page scan, choose a different element, "
                "or stop and report the blocker instead of repeating the same tool call."
            )
            _emit("warn", msg, action=action, repeat_count=count)
            return msg
        return None

    if policy_domains:
        tools.exclude_action("search")

    # S3: Worker has been observed hallucinating file-system + JS-evaluate
    # actions on the first few steps (gemini schema confusion), wasting step
    # budget before navigation begins. We never need these for a browser-only
    # agent — drop them from the registry so they aren't even valid outputs.
    for _excluded in ("write_file", "read_file", "replace_file", "evaluate"):
        try:
            tools.exclude_action(_excluded)
        except Exception:
            pass

    @tools.action(
        "Click element by index. This guarded click refuses final irreversible controls "
        "in halt-on-dangerous or dry-run safety modes.",
        param_model=ClickElementActionIndexOnly,
    )
    async def click(params: "ClickElementActionIndexOnly", browser_session) -> "ActionResult":
        try:
            node = await browser_session.get_element_by_index(params.index)
        except Exception as exc:
            msg = (
                f"Click target #{params.index} is stale or unreadable ({exc}). "
                "The page likely re-rendered. Re-scan the current page state before clicking again."
            )
            return ActionResult(error=msg, extracted_content=msg)

        if node is None:
            msg = (
                f"Click target #{params.index} is no longer available. "
                "The page changed. Re-scan the page and use a fresh element index."
            )
            return ActionResult(error=msg, extracted_content=msg)

        text = _node_policy_text(node)
        if safety_mode in ("halt-on-dangerous", "dry-run") and _is_dangerous_click_text(text):
            preview = re.sub(r"\s+", " ", text).strip()[:160]
            message = (
                f"Blocked dangerous click in {safety_mode}: element {params.index} looks like "
                f"a final irreversible action ({preview}). Stop and leave this for human review."
            )
            _emit("warn", message)
            return ActionResult(error=message, extracted_content=message, include_extracted_content_only_once=False)

        repeat_msg = _tool_repeat_guard("click", (int(params.index), re.sub(r"\s+", " ", text).strip()[:80]))
        if repeat_msg:
            return ActionResult(error=repeat_msg, extracted_content=repeat_msg, include_extracted_content_only_once=False)

        return await tools._click_by_index(params, browser_session)

    # S1 + S4: deterministic input wrapper. Forces a hard pre-clear via the DOM
    # (set value="" + fire input/change events) so SPA frameworks like ExtJS
    # actually drop the previous content, dispatches the underlying TypeText
    # event for the keystroke fidelity browser-use already implements, then
    # forces a blur so the framework commits the value to its backing model.
    # Also blocks repeated identical (index, text) calls — Worker tends to
    # retype the same value when Vision misreads the screenshot, which wastes
    # step budget without changing DOM state.
    # S8: log every TYPE attempt with its xpath so the post-run gate can
    # verify the value still appears in the DOM regardless of whether
    # browser-use's selector_map index has gone stale. Exposed on the
    # tools object so run_agent can iterate it after done().
    _typed_log: list[dict] = []
    setattr(tools, "_declario_typed_log", _typed_log)

    @tools.action(
        "Input text into element by index. Performs a hard pre-clear, types the "
        "text, then commits via blur — works on SPA portals (rs.ge ExtJS) where "
        "the default clear flag does not actually empty the widget. Dedupes "
        "consecutive identical typed values automatically.",
        param_model=InputTextAction,
    )
    async def input(
        params: "InputTextAction",
        browser_session,
        has_sensitive_data: bool = False,
        sensitive_data: dict | None = None,
    ) -> "ActionResult":
        # S4: same (index, text) three times in a row means the last attempts
        # did not create progress. Let one retry through, then force a fresh
        # strategy on the third identical tool call.
        dedup_key = (int(params.index), str(params.text))
        repeat_msg = _tool_repeat_guard("input", dedup_key)
        if repeat_msg:
            return ActionResult(error=repeat_msg, extracted_content=repeat_msg, include_extracted_content_only_once=False)

        node = None
        try:
            node = await browser_session.get_element_by_index(params.index)
        except Exception as exc:
            msg = (
                f"Input target #{params.index} is stale or unreadable ({exc}). "
                "The page likely re-rendered. Re-run field discovery or re-scan the page before typing again."
            )
            return ActionResult(extracted_content=msg, error=msg)
        if node is None:
            msg = (
                f"Input target #{params.index} is no longer available. "
                "The page changed. Re-run field discovery or re-scan the page before typing again."
            )
            return ActionResult(extracted_content=msg, error=msg)

        page = await _resolve_session_page(browser_session)
        page_kind = _page_kind(page)
        _emit("info", f"S1 page resolved for input #{params.index}: {page_kind}", page_kind=page_kind, field_index=int(params.index))

        xpath = ""
        try:
            xpath = node.xpath or ""
        except Exception:
            xpath = ""

        async def _target_info() -> dict | None:
            if page is None or not xpath:
                return None
            raw = await _safe_page_eval(
                page,
                """(xp) => {
                    const r = document.evaluate(xp, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                    const root = r && r.singleNodeValue;
                    if (!root) return null;
                    const tag = (root.tagName || '').toLowerCase();
                    let target = root;
                    let isInner = false;
                    if (tag !== 'input' && tag !== 'textarea' && tag !== 'select') {
                        const inner = root.querySelector(
                            'input:not([type="hidden"]):not([type="button"]):not([type="submit"]), textarea, select'
                        );
                        if (inner) { target = inner; isInner = true; }
                    }
                    try { target.scrollIntoView({block: 'center'}); } catch (_) {}
                    try { target.focus(); } catch (_) {}
                    return {
                        isInner,
                        tag: (target.tagName || '').toLowerCase(),
                        type: String(target.getAttribute('type') || ''),
                        className: String(target.className || ''),
                        inputMode: String(target.getAttribute('inputmode') || ''),
                        xName: String(target.getAttribute('x_name') || ''),
                        id: String(target.id || ''),
                        disabled: !!target.disabled,
                        readOnly: !!target.readOnly,
                        editable: !target.disabled && !target.readOnly,
                        value: target.value != null ? String(target.value) : ''
                    };
                }""",
                xpath,
            )
            return raw if isinstance(raw, dict) else None

        async def _read_back() -> str | None:
            if page is None or not xpath:
                return None
            raw = await _safe_page_eval(
                page,
                """(xp) => {
                    const r = document.evaluate(xp, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                    const root = r && r.singleNodeValue;
                    if (!root) return null;
                    const tag = (root.tagName || '').toLowerCase();
                    let target = root;
                    if (tag !== 'input' && tag !== 'textarea' && tag !== 'select') {
                        const inner = root.querySelector(
                            'input:not([type="hidden"]):not([type="button"]):not([type="submit"]), textarea, select'
                        );
                        if (inner) target = inner;
                    }
                    if (target.tagName === 'SELECT') {
                        const opt = target.options[target.selectedIndex];
                        return opt ? (opt.textContent || target.value || '') : (target.value || '');
                    }
                    return target.value != null ? String(target.value) : '';
                }""",
                xpath,
            )
            return None if raw is None else str(raw)

        async def _dom_set_value(value: str) -> bool:
            if page is None or not xpath:
                return False
            raw = await _safe_page_eval(
                page,
                """(xp, value) => {
                    const r = document.evaluate(xp, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                    const root = r && r.singleNodeValue;
                    if (!root) return false;
                    const tag = (root.tagName || '').toLowerCase();
                    let target = root;
                    if (tag !== 'input' && tag !== 'textarea' && tag !== 'select') {
                        const inner = root.querySelector(
                            'input:not([type="hidden"]):not([type="button"]):not([type="submit"]), textarea, select'
                        );
                        if (inner) target = inner;
                    }
                    try { target.scrollIntoView({block: 'center'}); } catch (_) {}
                    try { target.focus(); } catch (_) {}

                    const proto =
                        target.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype :
                        target.tagName === 'SELECT' ? HTMLSelectElement.prototype :
                        HTMLInputElement.prototype;
                    const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
                    if (setter) setter.call(target, value);
                    else target.value = value;
                    try { target.setAttribute('value', value); } catch (_) {}
                    try {
                        target.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true }));
                        target.dispatchEvent(new KeyboardEvent('keyup', { key: 'End', bubbles: true }));
                    } catch (_) {}
                    try {
                        target.dispatchEvent(new InputEvent('input', {
                            bubbles: true,
                            inputType: 'insertText',
                            data: String(value),
                        }));
                    } catch (_) {
                        try { target.dispatchEvent(new Event('input', { bubbles: true })); } catch (_) {}
                    }
                    try { target.dispatchEvent(new Event('change', { bubbles: true })); } catch (_) {}
                    try { if (typeof target.onchange === 'function') target.onchange.call(target); } catch (_) {}
                    try { target.blur(); } catch (_) {}
                    try { target.dispatchEvent(new Event('blur', { bubbles: true })); } catch (_) {}
                    return target.value != null ? String(target.value) : '';
                }""",
                xpath,
                value,
            )
            return False if raw is False or raw is None else True

        async def _extjs_set_value(value: str) -> dict | str | None:
            """Try Ext.getCmp().setValue() for ExtJS-managed fields.
            Returns the DOM readback value on success, 'no_ext_cmp' if no
            ExtJS component found, or None on error."""
            if page is None or not xpath:
                return None
            return await _safe_page_eval(
                page,
                """(xp, value) => {
                    const r = document.evaluate(xp, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                    const root = r && r.singleNodeValue;
                    if (!root) return null;
                    const tag = (root.tagName || '').toLowerCase();
                    let target = root;
                    if (tag !== 'input' && tag !== 'textarea' && tag !== 'select') {
                        const inner = root.querySelector(
                            'input:not([type="hidden"]):not([type="button"]):not([type="submit"]), textarea, select'
                        );
                        if (inner) target = inner;
                    }
                    if (typeof Ext === 'undefined') return 'no_ext_cmp';
                    const domValue = () => target && target.value != null ? String(target.value) : '';
                    const cmpCandidates = [];
                    const addCmp = (cmp) => {
                        if (cmp && cmpCandidates.indexOf(cmp) < 0 && typeof cmp.setValue === 'function') {
                            cmpCandidates.push(cmp);
                        }
                    };

                    // Walk ancestors because ExtJS usually owns a wrapper div,
                    // while browser-use often points at the inner input.
                    let el = target;
                    for (let i = 0; i < 8 && el; i++) {
                        if (el.id) {
                            try {
                                addCmp(Ext.getCmp(el.id));
                                if (Ext.ComponentQuery && Ext.ComponentQuery.query) {
                                    const escaped = String(el.id).replace(/([:.\\[\\],=>~])/g, '\\\\$1');
                                    for (const found of Ext.ComponentQuery.query('#' + escaped) || []) addCmp(found);
                                }
                            } catch (_) {}
                        }
                        el = el.parentElement;
                    }
                    try {
                        if (!cmpCandidates.length && target.name && Ext.ComponentQuery && Ext.ComponentQuery.query) {
                            for (const found of Ext.ComponentQuery.query('[name=' + JSON.stringify(String(target.name)) + ']') || []) {
                                addCmp(found);
                            }
                        }
                    } catch (_) {}

                    if (!cmpCandidates.length) return 'no_ext_cmp';

                    const cmp = cmpCandidates[0];
                    const oldValue = (typeof cmp.getValue === 'function') ? cmp.getValue() : undefined;
                    try { if (typeof cmp.focus === 'function') cmp.focus(false, 50); } catch (_) {}
                    try { if (typeof cmp.setRawValue === 'function') cmp.setRawValue(value); } catch (_) {}
                    try { cmp.setValue(value); } catch (_) {}
                    try {
                        const name =
                            (typeof cmp.getName === 'function' && cmp.getName())
                            || cmp.name
                            || target.name
                            || target.getAttribute('x_name')
                            || target.id
                            || '';
                        let record = null;
                        try { record = cmp.getRecord && cmp.getRecord(); } catch (_) {}
                        try { if (!record && cmp.ownerCt) record = cmp.ownerCt.getRecord && cmp.ownerCt.getRecord(); } catch (_) {}
                        try { if (!record && cmp.up) record = cmp.up('form')?.getRecord?.(); } catch (_) {}
                        if (record && name && typeof record.set === 'function') record.set(name, value);
                    } catch (_) {}
                    try { if (typeof cmp.fireEvent === 'function') cmp.fireEvent('change', cmp, value, oldValue); } catch (_) {}
                    try { if (typeof cmp.validate === 'function') cmp.validate(); } catch (_) {}
                    try { if (typeof cmp.blur === 'function') cmp.blur(); } catch (_) {}
                    try { target.dispatchEvent(new Event('input', { bubbles: true })); } catch (_) {}
                    try { target.dispatchEvent(new Event('change', { bubbles: true })); } catch (_) {}
                    try { target.blur(); } catch (_) {}
                    return {
                        status: 'ok',
                        componentId: String(cmp.id || cmp.itemId || ''),
                        domValue: domValue(),
                        rawValue: (typeof cmp.getRawValue === 'function') ? String(cmp.getRawValue()) : '',
                        value: (typeof cmp.getValue === 'function') ? String(cmp.getValue()) : '',
                    };
                }""",
                xpath,
                value,
            )

        used_inner = False
        strategy = "none"
        readback = None
        actor_element = None

        if page is not None and hasattr(page, "get_element"):
            try:
                backend_node_id = getattr(node, "backend_node_id", None)
                if backend_node_id is not None:
                    actor_element = await page.get_element(int(backend_node_id))
            except Exception:
                actor_element = None

        target_meta = None
        if page is not None and xpath:
            try:
                target_meta = await _target_info()
            except Exception:
                target_meta = None
        is_numeric_target = False
        if isinstance(target_meta, dict):
            used_inner = bool(target_meta.get("isInner"))
            class_name = str(target_meta.get("className") or "").lower()
            input_mode = str(target_meta.get("inputMode") or "").lower()
            input_type = str(target_meta.get("type") or "").lower()
            initial_value = str(target_meta.get("value") or "").strip()
            is_numeric_target = (
                "number" in class_name
                or input_type == "number"
                or input_mode in ("numeric", "decimal")
                or _decimal_from_text(initial_value) is not None
            )
            if target_meta.get("editable") is False:
                msg = (
                    f"S1 target #{params.index} is present but not editable "
                    f"(disabled={bool(target_meta.get('disabled'))}, readonly={bool(target_meta.get('readOnly'))}). "
                    "Do not retry the same field blindly. Re-check the field mapping or page state."
                )
                _emit("warn", msg, field_index=int(params.index), page_kind=page_kind)
                return ActionResult(extracted_content=msg, error=msg)

        if page is not None:
            try:
                click_event = browser_session.event_bus.dispatch(ClickElementEvent(node=node))
                await click_event
                await click_event.event_result(raise_if_any=False, raise_if_none=False)
            except Exception:
                pass

        # Primary path for current browser-use runtime: fill via the actor/CDP element.
        try:
            if actor_element is None:
                raise AttributeError("actor element unavailable")
            await actor_element.fill(params.text, clear=bool(params.clear))
            strategy = "element_fill"
        except Exception as exc:
            _emit(
                "info",
                f"S1 element.fill fallback on #{params.index}: {exc}",
                field_index=int(params.index),
                page_kind=page_kind,
            )
            try:
                dom_set_ok = await _dom_set_value(params.text)
                if not dom_set_ok:
                    raise RuntimeError("DOM setter could not resolve target")
                strategy = "dom_set"
            except Exception as dom_exc:
                try:
                    event = browser_session.event_bus.dispatch(
                        TypeTextEvent(
                            node=node,
                            text=params.text,
                            clear=bool(params.clear),
                            is_sensitive=has_sensitive_data,
                            sensitive_key_name=None,
                        )
                    )
                    await event
                    await event.event_result(raise_if_any=True, raise_if_none=False)
                    strategy = "typetext"
                except Exception as type_exc:
                    # Mixed path: use actor-page keypresses to clear/commit around TypeTextEvent when available.
                    if page is not None and _page_supports_actor_press(page):
                        try:
                            if params.clear:
                                await page.press("Control+A")
                                await page.press("Delete")
                            event = browser_session.event_bus.dispatch(
                                TypeTextEvent(
                                    node=node,
                                    text=params.text,
                                    clear=False,
                                    is_sensitive=has_sensitive_data,
                                    sensitive_key_name=None,
                                )
                            )
                            await event
                            await event.event_result(raise_if_any=True, raise_if_none=False)
                            strategy = "mixed"
                        except Exception as mixed_exc:
                            msg = (
                                f"S1 typing failed on #{params.index}. "
                                f"element.fill error={exc}; DOM setter error={dom_exc}; "
                                f"TypeText error={type_exc}; mixed recovery error={mixed_exc}. "
                                "Treat this field as rejected or detached. Re-scan before retrying."
                            )
                            return ActionResult(error=msg, extracted_content=msg)
                    else:
                        msg = (
                            f"S1 typing failed on #{params.index}. "
                            f"element.fill error={exc}; DOM setter error={dom_exc}; TypeText error={type_exc}. "
                            "Treat this field as rejected or detached. Re-scan before retrying."
                        )
                        return ActionResult(error=msg, extracted_content=msg)

        try:
            readback_settle_seconds = max(0.05, float(os.environ.get("INPUT_READBACK_SETTLE_SECONDS", "0.20")))
        except Exception:
            readback_settle_seconds = 0.20
        try:
            input_readback_retries = max(0, int(os.environ.get("INPUT_READBACK_RETRIES", "2")))
        except Exception:
            input_readback_retries = 2

        async def _commit_after_write() -> None:
            if page is None:
                return
            try:
                if _page_supports_actor_press(page):
                    await page.press("Tab")
                elif hasattr(page, "keyboard"):
                    await page.keyboard.press("Tab")
            except Exception:
                pass
            await asyncio.sleep(readback_settle_seconds)

        async def _settled_readback() -> str | None:
            await asyncio.sleep(readback_settle_seconds)
            try:
                return await _read_back()
            except Exception:
                return None

        # Commit the value after successful typing/fill. ExtJS needs a real
        # focus/blur cycle and a short settle window before readback.
        await _commit_after_write()
        readback = await _settled_readback()

        if readback is not None and not _values_equivalent(str(readback).strip(), str(params.text).strip()):
            for retry_idx in range(input_readback_retries):
                try:
                    dom_set_ok = await _dom_set_value(params.text)
                    if not dom_set_ok:
                        break
                    if "retry_dom_set" not in strategy:
                        strategy = f"{strategy}+retry_dom_set"
                    await _commit_after_write()
                    retry_readback = await _settled_readback()
                    if retry_readback is not None:
                        readback = retry_readback
                    if readback is not None and _values_equivalent(str(readback).strip(), str(params.text).strip()):
                        _emit(
                            "info",
                            f"S1 readback retry {retry_idx + 1} accepted for #{params.index}",
                            field_index=int(params.index),
                            strategy=strategy,
                            page_kind=page_kind,
                            readback=str(readback)[:120],
                        )
                        break
                except Exception:
                    continue

        if readback is not None and not _values_equivalent(str(readback).strip(), str(params.text).strip()) and is_numeric_target:
            numeric_candidates = _numeric_write_candidates(params.text)
            for candidate in numeric_candidates:
                if _values_equivalent(candidate, params.text) and candidate == str(params.text).strip():
                    continue
                try:
                    dom_set_ok = await _dom_set_value(candidate)
                    if not dom_set_ok:
                        continue
                    await _commit_after_write()
                    retry_readback = await _settled_readback()
                except Exception:
                    continue
                if retry_readback is not None and _values_equivalent(str(retry_readback).strip(), str(params.text).strip()):
                    readback = retry_readback
                    strategy = f"{strategy}+numeric_recovery"
                    _emit(
                        "info",
                        f"S1 numeric recovery accepted candidate {candidate!r} for #{params.index}",
                        field_index=int(params.index),
                        strategy=strategy,
                        page_kind=page_kind,
                        numeric_candidate=candidate,
                        readback=str(readback)[:120],
                    )
                    break

        # ExtJS fallback: if readback still mismatches, try Ext.getCmp().setValue().
        if (
            readback is not None
            and not _values_equivalent(str(readback).strip(), str(params.text).strip())
            and page is not None
            and xpath
        ):
            try:
                for ext_attempt in range(input_readback_retries + 1):
                    ext_result = await _extjs_set_value(params.text)
                    if ext_result is None or ext_result == "no_ext_cmp":
                        break
                    await _commit_after_write()
                    ext_readback = await _settled_readback()
                    if ext_readback is not None:
                        readback = ext_readback
                    if readback is not None and _values_equivalent(str(readback).strip(), str(params.text).strip()):
                        strategy = f"{strategy}+extjs_setValue"
                        _emit(
                            "info",
                            f"S1 ExtJS setValue accepted for #{params.index} on attempt {ext_attempt + 1}",
                            field_index=int(params.index),
                            strategy=strategy,
                            page_kind=page_kind,
                            readback=str(readback)[:120],
                        )
                        break
            except Exception as _ext_exc:
                _emit("info", f"S1 ExtJS setValue skipped for #{params.index}: {_ext_exc}")

        _emit(
            "info",
            f"S1 typed #{params.index} via {strategy} on {page_kind}",
            field_index=int(params.index),
            strategy=strategy,
            page_kind=page_kind,
            readback="" if readback is None else str(readback)[:120],
        )

        if readback is None and page is not None and xpath:
            msg = (
                f"S11 read-back failed for #{params.index}. The DOM target disappeared after typing. "
                "Treat this as a stale node and re-scan before retrying."
            )
            _emit("warn", msg, field_index=int(params.index), strategy=strategy, page_kind=page_kind)
            return ActionResult(extracted_content=msg, error=msg)

        if readback is not None and not _values_equivalent(str(readback).strip(), str(params.text).strip()):
            msg = (
                f"S11 read-back mismatch for #{params.index}: expected {params.text!r}, DOM shows {str(readback)!r}. "
                "The widget rejected or reformatted the value. Do not retry the same index blindly; "
                "re-check field mapping or editability first."
            )
            _emit(
                "warn",
                msg,
                field_index=int(params.index),
                strategy=strategy,
                page_kind=page_kind,
                expected=str(params.text)[:80],
                actual=str(readback)[:80],
            )
            try:
                _typed_log.append({
                    "index": int(params.index),
                    "text": str(params.text),
                    "xpath": str(xpath or ""),
                    "used_inner": bool(used_inner),
                    "strategy": strategy,
                    "page_kind": page_kind,
                    "readback": "" if readback is None else str(readback),
                })
            except Exception:
                pass
            return ActionResult(extracted_content=msg, error=msg, include_extracted_content_only_once=False)

        try:
            _typed_log.append({
                "index": int(params.index),
                "text": str(params.text),
                "xpath": str(xpath or ""),
                "used_inner": bool(used_inner),
                "strategy": strategy,
                "page_kind": page_kind,
                "readback": "" if readback is None else str(readback),
            })
        except Exception:
            pass

        msg = f"Typed {params.text!r} into #{params.index}"
        return ActionResult(
            extracted_content=msg,
            long_term_memory=f"Typed '{params.text}' into element {params.index}",
            include_extracted_content_only_once=False,
        )

    @tools.action(
        "Discover the site's navigation menu (sidebar, top-bar, tabs, dropdowns) via vision. "
        "Call this ONCE right after login when you first see the dashboard, BEFORE clicking around. "
        "It returns a JSON list of every visible menu item with label_ge / label_en / kind / "
        "parent_label / hint, so you know exactly which menu to click to reach your goal "
        "(e.g. 'click \"დეკლარაციები\" sidebar item, then submenu \"დღგ\"'). "
        "Cached per-domain so repeated calls within a session are free."
    )
    async def discover_navigation(browser_session, intent: str = "") -> "ActionResult":
        # S6: derive URL from the live session — Gemini's structured output
        # mangles parameterless schemas (browser-use generates an empty
        # object schema; Gemini insists on putting `_placeholder` in it,
        # which then fails additionalProperties=False). Keep one optional
        # `intent` slot so the schema has a property to fill. We don't read
        # it. Current page is the only sensible URL target anyway.
        _ = intent
        page_url = ""
        try:
            _p = await _resolve_session_page(browser_session)
            if _p is not None:
                try:
                    page_url = str(await _safe_page_eval(_p, "() => window.location.href") or "")
                except Exception:
                    page_url = getattr(_p, "url", "") or ""
        except Exception:
            page_url = ""
        try:
            domain_key = (page_url or "").split("/")[2] if "://" in (page_url or "") else (page_url or "")
            domain_key = domain_key[:120] or "unknown"

            repeat_msg = _tool_repeat_guard("discover_navigation", domain_key)
            if repeat_msg:
                return ActionResult(error=repeat_msg, extracted_content=repeat_msg, include_extracted_content_only_once=False)

            cached = _nav_scout_cache.get(domain_key)
            if cached is not None:
                content = _format_navigation_for_worker(cached, domain_key + " [cached]")
                return ActionResult(
                    extracted_content=content,
                    long_term_memory=f"Nav map on {domain_key}: "
                                     + ", ".join(it["label_ge"] for it in cached[:10]),
                    include_extracted_content_only_once=False,
                )

            try:
                screenshot = await browser_session.take_screenshot()
            except Exception as exc:
                return ActionResult(
                    extracted_content=f"discover_navigation: screenshot failed ({exc})",
                    error=str(exc),
                )

            items = await _vision_scout_navigation(screenshot, task_hint)
            _nav_scout_cache[domain_key] = items
            _emit(
                "info",
                f"B: scouted {len(items)} nav item(s) on {domain_key}",
                nav_item_count=len(items),
                domain=domain_key,
            )
            content = _format_navigation_for_worker(items, domain_key)
            return ActionResult(
                extracted_content=content,
                long_term_memory=f"Nav map on {domain_key}: "
                                 + ", ".join(it["label_ge"] for it in items[:10]),
                include_extracted_content_only_once=False,
            )
        except Exception as exc:
            return ActionResult(
                extracted_content=f"discover_navigation error: {exc}",
                error=str(exc),
            )

    @tools.action(
        "Discover every fillable form field on the current page via vision. "
        "Use this ONCE when arriving at a new form page (not on every step) "
        "to learn which Georgian labels map to which inputs before you start typing. "
        "Returns a structured list of label_ge / label_en / type / css_hint / page_section. "
        "Cached per-URL so repeated calls on the same page are free."
    )
    async def discover_form_fields(browser_session, intent: str = "") -> "ActionResult":
        # S6: optional `intent` slot avoids the Gemini parameterless-schema
        # placeholder-injection bug. See discover_navigation comment.
        _ = intent
        page_url = ""
        try:
            _p = await _resolve_session_page(browser_session)
            if _p is not None:
                try:
                    page_url = str(await _safe_page_eval(_p, "() => window.location.href") or "")
                except Exception:
                    page_url = getattr(_p, "url", "") or ""
        except Exception:
            page_url = ""
        try:
            url_key = (page_url or "").split("#")[0].split("?")[0][:300]
            url_short = url_key.replace("https://", "").replace("http://", "")[:80]

            repeat_msg = _tool_repeat_guard("discover_form_fields", url_key or "unknown")
            if repeat_msg:
                return ActionResult(error=repeat_msg, extracted_content=repeat_msg, include_extracted_content_only_once=False)

            cached = _field_cache_get(url_key)
            if cached is not None:
                fields = cached.get("fields") or []
                status = str(cached.get("status") or "ok")
                if status == "empty":
                    reason = str(cached.get("reason") or "no fields discovered")
                    _emit(
                        "info",
                        f"P1: using negative field-discovery cache on {url_short}",
                        url=url_short,
                        source=str(cached.get("source") or "unknown"),
                        cache_status=status,
                    )
                    msg = (
                        f"discover_form_fields: no fields discovered on {url_short} "
                        f"(source={cached.get('source') or 'unknown'}; reason={reason}). "
                        "Do not call discover_form_fields again on this unchanged URL. "
                        "Switch strategy: inspect visible controls, refresh page state, or navigate to a different form section."
                    )
                    return ActionResult(
                        extracted_content=msg,
                        long_term_memory=msg,
                        include_extracted_content_only_once=False,
                    )
                content = _format_discovery_for_worker(fields, url_short + " [cached]")
                return ActionResult(
                    extracted_content=content,
                    long_term_memory=f"Form fields on {url_short}: "
                                     + ", ".join(f["label_ge"] for f in fields[:12]),
                    include_extracted_content_only_once=False,
                )

            # S2: try the cheap path first — browser-use's selector_map already
            # walked the full DOM (incl. shadow roots / iframes) for the agent's
            # interaction. Use that instead of querySelectorAll-from-scratch
            # when the page is an SPA portal whose inputs Vision can't see.
            page_for_session = await _resolve_session_page(browser_session)

            session_snapshot = await _extract_dom_fields_via_session(browser_session, page_for_session)
            session_fields_raw = session_snapshot.get("fields") or []
            _emit(
                "info",
                f"P1 selector_map on {url_short}: {len(session_fields_raw)} raw field(s)",
                discovered_field_count=len(session_fields_raw),
                url=url_short,
                source="selector_map",
            )
            if session_fields_raw:
                fields = []
                for f in session_fields_raw:
                    label = (f.get("label") or f.get("ariaLabel") or f.get("placeholder")
                             or f.get("name") or "").strip()
                    if not label:
                        continue
                    css_bits: list[str] = []
                    if f.get("id"):
                        css_bits.append(f"#{f['id']}")
                    elif f.get("name"):
                        css_bits.append(f"[name=\"{f['name']}\"]")
                    css_hint = (f.get("tag") or "input") + ("".join(css_bits) if css_bits else f"[index={f.get('index')}]")
                    fields.append({
                        "label_ge": label[:200],
                        "label_en": "",
                        "type": str(f.get("type") or f.get("tag") or "input")[:20],
                        "css_hint": css_hint[:200],
                        "page_section": "",
                    })
                if fields:
                    _field_cache_put(url_key, fields, source="selector_map")
                    _emit(
                        "info",
                        f"P1: discovered {len(fields)} field(s) via selector_map on {url_short}",
                        discovered_field_count=len(fields),
                        url=url_short,
                        source="selector_map",
                    )
                    content = _format_discovery_for_worker(fields, url_short)
                    return ActionResult(
                        extracted_content=content,
                        long_term_memory=f"Form fields on {url_short}: "
                                         + ", ".join(f["label_ge"] for f in fields[:12]),
                        include_extracted_content_only_once=False,
                    )

            # Fall back to Vision only if selector_map gave nothing.
            try:
                screenshot = await browser_session.take_screenshot()
            except Exception as exc:
                msg = (
                    f"discover_form_fields: no fields discovered on {url_short} via selector_map, "
                    f"and screenshot capture failed ({exc}). Do not retry this tool on the same unchanged URL. "
                    "Switch strategy: inspect visible controls or refresh the page state."
                )
                _field_cache_put(url_key, [], source="selector_map", status="empty", reason=f"screenshot failed: {exc}")
                return ActionResult(extracted_content=msg, error=str(exc))

            fields = await _vision_discover_fields(screenshot, page_url)
            if not fields:
                reason = "selector_map and vision both returned zero fields"
                _field_cache_put(url_key, [], source="vision", status="empty", reason=reason)
                msg = (
                    f"discover_form_fields: no fields discovered on {url_short} "
                    f"(source=vision; reason={reason}). Do not call discover_form_fields again on this unchanged URL. "
                    "Switch strategy: inspect visible controls, refresh page state, or move to a different page section."
                )
                _emit(
                    "warn",
                    f"P1: zero fields discovered on {url_short}",
                    url=url_short,
                    source="vision",
                    cache_status="empty",
                )
                return ActionResult(
                    extracted_content=msg,
                    long_term_memory=msg,
                    include_extracted_content_only_once=False,
                )
            _field_cache_put(url_key, fields, source="vision")
            _emit(
                "info",
                f"P1: discovered {len(fields)} field(s) on {url_short}",
                discovered_field_count=len(fields),
                url=url_short,
                source="vision",
            )
            content = _format_discovery_for_worker(fields, url_short)
            return ActionResult(
                extracted_content=content,
                long_term_memory=f"Form fields on {url_short}: "
                                 + ", ".join(f["label_ge"] for f in fields[:12]),
                include_extracted_content_only_once=False,
            )
        except Exception as exc:
            return ActionResult(
                extracted_content=f"discover_form_fields error: {exc}",
                error=str(exc),
            )

    # T1.3 — Read-back verification.
    # Call this immediately BEFORE done() in halt-on-dangerous mode. Pass a
    # JSON map of {label_substring: expected_value} for the values you typed.
    # Returns a per-field pass/fail report so the Worker can re-type any
    # field that didn't actually accept its value (the most common failure
    # mode: SPA reset to 0.00 after Tab).
    #
    # Hard cap: after 2 calls in a single run we tell the Worker to stop
    # retrying and proceed to done(). Without the cap, Worker LLMs get stuck
    # calling verify in a loop instead of fixing or yielding.
    verify_call_count = {"n": 0}

    @tools.action(
        "Verify the values currently shown by the form's inputs match what was typed. "
        "Call this RIGHT BEFORE done() — pass a JSON object mapping label substrings to "
        'their expected values, e.g. {"(1) ანაზღაურების": "45800.00", "(2) წინასწარ": "2400.00"}. '
        "Returns OK or a list of mismatches so you can re-type the bad fields. "
        "HARD LIMIT: maximum 2 calls per run — after that, proceed to done() even if mismatches remain."
    )
    async def verify_typed_values(expected: dict[str, str], browser_session) -> "ActionResult":
        verify_call_count["n"] += 1
        if verify_call_count["n"] > 2:
            msg = (
                f"verify_typed_values: HARD LIMIT REACHED ({verify_call_count['n']} calls). "
                "STOP calling verify and call done(success=True) NOW. Note any unresolved "
                "fields in the done summary so the human reviewer can correct them."
            )
            _emit("warn", f"T1.3: verify hard-limit hit at call #{verify_call_count['n']}")
            return ActionResult(extracted_content=msg, long_term_memory=msg)
        try:
            if not isinstance(expected, dict) or not expected:
                return ActionResult(
                    extracted_content="verify_typed_values: empty 'expected' map; nothing to verify.",
                    long_term_memory="verify_typed_values noop",
                )

            page = await _resolve_session_page(browser_session)
            if page is None:
                return ActionResult(
                    extracted_content="verify_typed_values: page not reachable; cannot verify.",
                    error="page unavailable",
                )

            snapshot = await _extract_dom_fields(page)
            fields = snapshot.get("fields") or []
            if not fields:
                return ActionResult(
                    extracted_content="verify_typed_values: no form fields visible to read back.",
                    error="no fields",
                )

            ok_lines: list[str] = []
            mismatch_lines: list[str] = []
            for label_query, raw_expected in list(expected.items())[:30]:
                exp = str(raw_expected).strip()
                if not exp:
                    continue
                # Find the best-matching field by label substring.
                match = None
                for f in fields:
                    haystack = _compact_label(f)
                    if _label_contains(haystack, label_query):
                        match = f
                        break
                if match is None:
                    mismatch_lines.append(f'"{label_query}": NO MATCHING FIELD')
                    continue
                actual = str(match.get("value") or match.get("text") or "").strip()
                if _values_equivalent(actual, exp):
                    ok_lines.append(f'"{label_query}" = {actual!r} ✓')
                else:
                    mismatch_lines.append(
                        f'"{label_query}": expected {exp!r}, page shows {actual!r}'
                    )

            if not mismatch_lines:
                summary = f"verify_typed_values: ALL OK ({len(ok_lines)} field(s) verified)."
                _emit("info", f"T1.3: read-back verification all OK ({len(ok_lines)})")
                return ActionResult(
                    extracted_content=summary,
                    long_term_memory=summary,
                )
            body = "\n".join(["verify_typed_values: MISMATCH"] + mismatch_lines + ["", "Re-type the mismatched field(s) using Ctrl+A to clear first."])
            _emit("warn", f"T1.3: read-back found {len(mismatch_lines)} mismatch(es)")
            return ActionResult(
                extracted_content=body,
                long_term_memory=f"verify mismatches: {len(mismatch_lines)}",
            )
        except Exception as exc:
            return ActionResult(
                extracted_content=f"verify_typed_values error: {exc}",
                error=str(exc),
            )

    return tools


def _values_equivalent(actual: str, expected: str) -> bool:
    """Numeric-aware string comparison. Treats '2400', '2400.00', '2 400.00' as equal."""
    a = (actual or "").strip()
    e = (expected or "").strip()
    if not a and not e:
        return True
    if a == e:
        return True
    actual_dec = _decimal_from_text(a)
    expected_dec = _decimal_from_text(e)
    if actual_dec is not None and expected_dec is not None:
        return actual_dec == expected_dec
    actual_digits = re.sub(r"\D", "", a)
    expected_digits = re.sub(r"\D", "", e)
    if len(actual_digits) >= 6 and actual_digits == expected_digits:
        return True
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    return norm(a) == norm(e)


# ── Stdout protocol ────────────────────────────────────────────────────────────
# Every line written to stdout is a JSON object the Node.js backend can parse.

def _emit(event_type: str, message: str, **extra):
    """Emit a JSON event line. All string fields run through _sanitize_for_user
    so library-internal class names / brand words never reach the end user.

    `extra` may contain non-string fields (numbers, lists, screenshot bytes) —
    those pass through untouched. `image` is reserved for screenshots and is
    skipped from sanitisation since it's not user-readable text."""
    safe_message = _sanitize_for_user(message) if isinstance(message, str) else message
    safe_extra: dict = {}
    for k, v in extra.items():
        if k in ("image",):  # screenshot bytes/base64 — leave alone
            safe_extra[k] = v
            continue
        if isinstance(v, str):
            safe_extra[k] = _sanitize_for_user(v)
        elif isinstance(v, list):
            safe_extra[k] = [
                _sanitize_for_user(item) if isinstance(item, str) else item
                for item in v
            ]
        else:
            safe_extra[k] = v
    payload = {"type": event_type, "message": safe_message, **safe_extra}
    print(json.dumps(payload, ensure_ascii=False), flush=True)


# ── Agent callbacks ────────────────────────────────────────────────────────────

_SENSITIVE_VAR_RE = re.compile(r"password|token|otp|secret|pin", re.I)


# Strip library-specific words from any text that gets emitted to the user. We
# keep this list narrow on purpose — heavy-handed substitution makes error
# messages incomprehensible. The goal is to hide the *brand* of automation
# library, not to obscure what went wrong.
_BRAND_LEAK_RE = re.compile(
    r"\b(?:"
    r"browser[_-]use(?:\.tools(?:\.service)?|\.agent|\.controller|\.browser)?"
    r"|browser[_-]use"
    r"|playwright"
    r")\b",
    re.I,
)
# Internal Pydantic action class names that leak via repr().
_INTERNAL_CLASS_RE = re.compile(
    r"\broot=\w*ActionModel\([^)]*\)|"
    r"\b\w+ActionModel\b|"
    r"\b\w+ActionIndexOnly\b|"
    r"\bClickElementActionIndexOnly\b|"
    r"\bInputTextAction\b|"
    r"\bNavigateAction\b",
)


def _sanitize_for_user(text) -> str:
    """Best-effort: strip library brand names + raw class names from any
    string that flows out to the user-visible event stream."""
    if text is None:
        return ""
    s = str(text)
    s = _BRAND_LEAK_RE.sub("agent", s)
    s = _INTERNAL_CLASS_RE.sub("step", s)
    return s


def _sanitize_action_for_display(action_obj) -> str:
    """Convert an internal action object into a clean, library-agnostic
    description suitable for end-user display. Hides upstream class names
    like NavigateActionModel / ClickElementActionIndexOnly / InputTextAction
    that would expose which automation library powers us."""
    try:
        if hasattr(action_obj, "model_dump"):
            data = action_obj.model_dump(exclude_none=True)
        elif isinstance(action_obj, dict):
            data = action_obj
        else:
            return "action"
        if not isinstance(data, dict) or not data:
            return "action"

        # Each step's ActionModel has exactly one non-None field — the action
        # the agent decided to take. Find it and format.
        for action_name, params in data.items():
            if params is None:
                continue
            return _format_action_friendly(action_name, params)
        return "thinking..."
    except Exception:
        return "action"


def _format_action_friendly(action_name: str, params) -> str:
    """Map an internal action name + params dict to a clean human description."""
    if not isinstance(params, dict):
        try:
            params = params.model_dump(exclude_none=True) if hasattr(params, "model_dump") else {}
        except Exception:
            params = {}

    name = (action_name or "").lower()

    if name in ("navigate", "go_to_url", "open_url"):
        url = str(params.get("url") or "")
        return f"Navigate to {url}" if url else "Navigate"

    if name in ("click", "click_element", "click_element_by_index"):
        idx = params.get("index")
        return f"Click element #{idx}" if idx is not None else "Click"

    if name in ("input", "input_text", "type"):
        text = str(params.get("text") or "")
        idx = params.get("index")
        # Don't display credential-shaped text in plaintext.
        looks_sensitive = _SENSITIVE_VAR_RE.search(text) or len(text) >= 32
        preview = "***" if looks_sensitive else (text[:40] + "…" if len(text) > 40 else text)
        return f'Type "{preview}" into field #{idx}' if idx is not None else f'Type "{preview}"'

    if name in ("scroll", "scroll_down", "scroll_up"):
        down = bool(params.get("down", True))
        pages = float(params.get("pages") or 1.0)
        return f"Scroll {'down' if down else 'up'} {pages:.1f} page(s)"

    if name in ("wait", "wait_for"):
        secs = params.get("seconds") or params.get("ms") or 0
        return f"Wait {secs}s" if isinstance(secs, (int, float)) and secs else "Wait"

    if name in ("send_keys", "press_key", "press"):
        keys = params.get("keys") or params.get("key") or "?"
        return f'Press "{keys}"'

    if name in ("done", "finish"):
        return "Mark task done"

    if name in ("search_page",):
        pat = str(params.get("pattern") or "")[:80]
        return f'Search page for "{pat}"' if pat else "Search page"

    if name in ("find_elements",):
        sel = str(params.get("selector") or "")[:80]
        return f'Find elements "{sel}"' if sel else "Find elements"

    if name in ("evaluate", "evaluate_javascript"):
        return "Run page script"

    if name in ("discover_form_fields",):
        return "Inspect form fields on page"
    if name in ("discover_navigation",):
        return "Inspect site navigation"

    if name in ("upload", "upload_file"):
        return "Upload file"

    if name in ("select", "select_option"):
        val = str(params.get("value") or params.get("text") or "")[:60]
        idx = params.get("index")
        if val:
            return f'Select "{val}"' + (f" in field #{idx}" if idx is not None else "")
        return "Select"

    # Generic clean fallback — never leak underscores or "_action" suffixes.
    label = re.sub(r"_(action|model)$", "", name).replace("_", " ").strip().title()
    return label or "Action"


# R4: track which step numbers we've already emitted via on_step. Pro/
# thinking-mode runs occasionally drop callbacks (when an LLM call times
# out the post-LLM hook is skipped — see browser_use/agent/service.py
# `_handle_post_llm_processing` requires `last_model_output`). After
# agent.run() returns we walk `history.history` and emit any items whose
# step_number didn't already appear. Reset per-run by `_reset_emitted_steps()`.
_emitted_step_numbers: set[int] = set()


def _reset_emitted_steps() -> None:
    _emitted_step_numbers.clear()


def on_step(browser_state_summary, model_output, step_number: int):
    """Called after each agent step. Emits a sanitised, library-agnostic
    summary for end users."""
    action_summaries: list[str] = []
    if model_output and hasattr(model_output, "action"):
        actions = model_output.action or []
        for a in actions:
            try:
                action_summaries.append(_sanitize_action_for_display(a))
            except Exception:
                continue

    if not action_summaries:
        action_summaries = ["thinking..."]

    # browser-use can run several actions in one step — log every one of them
    # so the SSE stream reflects the actual filled values, not just the first.
    summary = (
        action_summaries[0]
        if len(action_summaries) == 1
        else f"[{len(action_summaries)} actions] " + " | ".join(action_summaries)
    )
    _emitted_step_numbers.add(step_number)
    _emit(
        "step",
        f"Step {step_number}: {summary}",
        step=step_number,
        action=summary,
        actions=action_summaries,  # keep full list for forensic checks
    )

    # Live screenshot streaming — emit a compressed preview the UI can show.
    # Disabled with STREAM_SCREENSHOTS=0 (e.g. for high-concurrency bulk runs).
    if os.environ.get("STREAM_SCREENSHOTS", "1") != "0":
        try:
            ss = getattr(browser_state_summary, "screenshot", None)
            if ss and isinstance(ss, str):
                # Cap at ~250KB to keep SSE frames responsive.
                trimmed = ss[:250_000]
                _emit(
                    "frame",
                    f"step {step_number} screenshot",
                    step=step_number,
                    image=trimmed,
                    truncated=len(ss) > len(trimmed),
                )
        except Exception:
            pass  # streaming is best-effort


def on_done(history):
    """Called when the agent finishes. Emits any step events that the
    per-step callback dropped (Pro/thinking-mode timeouts), so the UI's
    step timeline matches what actually happened."""
    _emit("done_callback", "Agent finished, collecting result...")
    try:
        items = getattr(history, "history", None) or []
        for idx, item in enumerate(items, start=1):
            mo = getattr(item, "model_output", None)
            metadata = getattr(item, "metadata", None)
            step_no = (
                getattr(metadata, "step_number", None)
                if metadata is not None
                else None
            )
            if step_no is None:
                step_no = idx
            if step_no in _emitted_step_numbers:
                continue
            actions = getattr(mo, "action", None) if mo else None
            summaries: list[str] = []
            for a in actions or []:
                try:
                    summaries.append(_sanitize_action_for_display(a))
                except Exception:
                    continue
            if not summaries:
                summaries = ["thinking..."]
            summary = (
                summaries[0]
                if len(summaries) == 1
                else f"[{len(summaries)} actions] " + " | ".join(summaries)
            )
            _emit(
                "step",
                f"Step {step_no}: {summary}",
                step=step_no,
                action=summary,
                actions=summaries,
                source="r4_replay",
            )
            _emitted_step_numbers.add(step_no)
    except Exception as exc:
        _emit("warn", f"R4 step replay failed (non-fatal): {exc}")


# ── Main runner ────────────────────────────────────────────────────────────────

async def fetch_playbook_steps(playbook_id: str) -> list[dict]:
    """Fetch playbook steps from the backend."""
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:3001")
    try:
        client = _get_http()
        resp = await client.get(f"{backend_url}/playbooks/{playbook_id}")
        if resp.status_code == 200:
            return resp.json().get("playbook", {}).get("steps", [])
        _emit("error", f"Playbook fetch failed: HTTP {resp.status_code}")
    except Exception as e:
        _emit("error", f"Could not fetch playbook: {e}")
    return []


_PLACEHOLDER_RE = re.compile(r"\$\{(\w+)\}")
_CRED_VAR_RE = re.compile(r"password|token|otp|secret|pin|username|^user$|email|login", re.I)


def _substitute(text: str, user_data: dict) -> str:
    """Replace ${variable} placeholders with user_data values."""
    return _PLACEHOLDER_RE.sub(lambda m: str(user_data.get(m.group(1), m.group(0))), text)


def _extract_field_map(steps: list[dict]) -> dict[str, str]:
    """Extract a {excel_key → form_label} map from playbook steps.

    For each TYPE/SELECT/INPUT step that types a `${var}` substitution, record
    var → step.target_text or target_description. Used to give the agent
    explicit knowledge of which Excel column maps to which Georgian form label
    on rs.ge — even in free-form mode where there's no playbook to follow.
    """
    field_map: dict[str, str] = {}
    for s in steps or []:
        action = (s.get("action") or "").upper()
        if action not in ("TYPE", "SELECT", "INPUT"):
            continue
        value = s.get("value") or ""
        label = (s.get("target_text") or s.get("target_description") or "").strip()
        if not label:
            continue
        for m in _PLACEHOLDER_RE.finditer(value):
            var = m.group(1)
            if _CRED_VAR_RE.search(var):
                continue
            # First occurrence wins (later playbooks don't overwrite earlier).
            if var not in field_map:
                field_map[var] = label
    return field_map


def _format_field_map(field_map: dict[str, str], user_data: dict | None = None) -> str:
    """Render the field map as an authoritative ground-truth block for prompts."""
    if not field_map:
        return ""
    lines = ["=== AUTHORITATIVE FIELD MAP (Excel column → form label on rs.ge) ==="]
    for var in sorted(field_map.keys()):
        label = field_map[var]
        v_display = ""
        if user_data and var in user_data:
            v = str(user_data[var]).strip()
            if v and v not in ("0", "0.0", "0.00", "[]", "{}"):
                v_display = f"  →  ENTER: {v}"
        lines.append(f"  {var}  →  \"{label}\"{v_display}")
    lines.append("=== END FIELD MAP ===")
    lines.append(
        "When filling the form, locate inputs by the EXACT Georgian label above. "
        "Do NOT guess label translations — use the verbatim string in quotes."
    )
    return "\n".join(lines)


# ── Phase Q: site memory + failure-pattern client ─────────────────────────────
# Q1: pull aggregated SiteKnowledge for a domain (distilled by backend from
# every reviewed playbook on that domain).
# Q2: read/write failure patterns so a K1/M1/K4 mistake on day-1 shows up as
# a "WATCH OUT FOR ..." hint on day-2.

def _domain_from_url(url: str) -> str:
    """Extract bare hostname from a URL or domain string. Empty if unparseable."""
    if not url:
        return ""
    raw = str(url).strip().lower()
    if raw.startswith("http://") or raw.startswith("https://"):
        try:
            from urllib.parse import urlparse
            return (urlparse(raw).hostname or "").strip()
        except Exception:
            return ""
    return raw.split("/")[0].split(":")[0].strip()


async def fetch_site_memory(domain: str) -> dict:
    """Returns {pages, transitions, dialogs, field_map} or {} if no memory yet."""
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:3001")
    norm = _domain_from_url(domain)
    if not norm:
        return {}
    try:
        client = _get_http()
        resp = await client.get(f"{backend_url}/agent/site-memory/{norm}", timeout=10)
        if resp.status_code != 200:
            return {}
        body = resp.json()
        return body.get("knowledge") or {}
    except Exception:
        return {}


async def fetch_failure_patterns(domain: str, limit: int = 20) -> list[dict]:
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:3001")
    norm = _domain_from_url(domain)
    if not norm:
        return []
    try:
        client = _get_http()
        resp = await client.get(
            f"{backend_url}/agent/failure-patterns",
            params={"domain": norm, "limit": limit},
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        body = resp.json()
        return body.get("patterns") or []
    except Exception:
        return []


async def report_failure_pattern(
    *,
    domain: str,
    failure_type: str,
    symptom: str,
    url_pattern: str | None = None,
    field_label: str | None = None,
    workaround: str | None = None,
) -> None:
    """Fire-and-forget POST to /agent/failure-patterns. Non-fatal on error."""
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:3001")
    norm = _domain_from_url(domain)
    if not norm or not symptom:
        return
    try:
        client = _get_http()
        await client.post(
            f"{backend_url}/agent/failure-patterns",
            json={
                "domain": norm,
                "url_pattern": url_pattern,
                "field_label": field_label,
                "failure_type": failure_type,
                "symptom": symptom[:1000],
                "workaround": workaround,
            },
            timeout=10,
        )
    except Exception:
        pass  # learning is best-effort; never break a run on telemetry


def _format_site_memory_for_worker(knowledge: dict, domain: str) -> str:
    """Render the distilled SiteKnowledge as a compact prompt block.

    Keep it tight — the Worker has limited context and we already inject
    field_map + nav_prefix elsewhere. This block adds *page-level* hints
    (transitions, common clicks, dialogs) that aren't in those.
    """
    if not knowledge:
        return ""
    pages = knowledge.get("pages") or []
    transitions = knowledge.get("transitions") or []
    dialogs = knowledge.get("dialogs") or []
    if not pages and not transitions and not dialogs:
        return ""

    lines = [f"=== SITE MEMORY for {domain} (distilled from past reviewed playbooks) ==="]

    if transitions:
        lines.append("-- KNOWN PAGE TRANSITIONS --")
        for t in transitions[:15]:
            lines.append(
                f"  • on {t.get('from_url_pattern','?')} → click \"{t.get('click_target','?')}\" → arrives at {t.get('to_url_pattern','?')}"
                + (f" [seen {t.get('seen')}×]" if t.get("seen", 0) > 1 else "")
            )

    if pages:
        lines.append("-- KNOWN PAGES --")
        for p in pages[:8]:
            role = p.get("role", "other")
            url = p.get("url_pattern", "?")
            lines.append(f"  • {url} ({role})")
            for c in (p.get("common_clicks") or [])[:6]:
                target = c.get("target", "?")
                seen = c.get("seen", 1)
                leads = c.get("leads_to_url_pattern")
                suffix = f" → {leads}" if leads else ""
                lines.append(f'      click "{target}"{suffix}' + (f" [seen {seen}×]" if seen > 1 else ""))
            field_labels = p.get("field_labels") or []
            if field_labels:
                preview = ", ".join(f'"{f}"' for f in field_labels[:6])
                more = f" (+{len(field_labels)-6} more)" if len(field_labels) > 6 else ""
                lines.append(f"      form fields: {preview}{more}")

    if dialogs:
        lines.append("-- KNOWN DIALOGS / POPUPS --")
        for d in dialogs[:8]:
            lines.append(f"  • on {d.get('trigger_url','?')}: dismiss with \"{d.get('button_label','?')}\"")

    lines.append("=== END SITE MEMORY ===")
    lines.append(
        "Use these transitions as a navigation recipe. If a known click target is "
        "visible on the page, prefer it over scrolling/searching."
    )
    return "\n".join(lines)


def _format_failure_patterns_for_worker(patterns: list[dict]) -> str:
    """Render the top failure patterns as a 'WATCH OUT' block."""
    if not patterns:
        return ""
    lines = ["=== KNOWN FAILURE MODES on this site (from past runs) ==="]
    for p in patterns[:10]:
        ftype = p.get("failure_type", "?")
        url = p.get("url_pattern") or ""
        label = p.get("field_label") or ""
        seen = p.get("occurrence_count", 1)
        symptom = (p.get("symptom") or "").strip()
        workaround = (p.get("workaround") or "").strip()
        loc = []
        if url: loc.append(f"on {url}")
        if label: loc.append(f"field \"{label}\"")
        loc_text = " ".join(loc)
        head = f"  • [{ftype}{f' × {seen}' if seen > 1 else ''}]"
        if loc_text: head += f" {loc_text}"
        lines.append(head)
        if symptom: lines.append(f"      symptom: {symptom[:300]}")
        if workaround: lines.append(f"      workaround: {workaround[:300]}")
    lines.append("=== END FAILURE MODES ===")
    return "\n".join(lines)


async def fetch_all_task_playbook_steps() -> list[list[dict]]:
    """Fetch steps from every 'task'-kind playbook for free-form field-map use."""
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:3001")
    try:
        client = _get_http()
        resp = await client.get(f"{backend_url}/playbooks")
        if resp.status_code != 200:
            return []
        playbooks = resp.json().get("playbooks", []) or []
        out: list[list[dict]] = []
        for pb in playbooks:
            kind = pb.get("kind")
            status = pb.get("status")
            if kind not in (None, "task"):
                continue
            if status and status != "ready":
                continue
            if pb.get("review_status") not in (None, "reviewed"):
                continue
            steps = pb.get("steps") or []
            if steps:
                out.append(steps)
        return out
    except Exception:
        return []


def _merge_field_maps(maps: list[dict[str, str]]) -> dict[str, str]:
    """Merge multiple field maps; first occurrence of each var wins."""
    merged: dict[str, str] = {}
    for m in maps:
        for k, v in m.items():
            if k not in merged:
                merged[k] = v
    return merged


# ── C: Navigation-prefix extraction (free-form gets a recipe from past playbooks)
# We scan a recorded playbook and collect every CLICK / SELECT step that
# occurs BEFORE the first TYPE step. Rationale: TYPE = entering form data,
# so every step before that is the navigation flow that gets you to the form.
# These descriptions are then formatted as a "recommended navigation path"
# block injected into free-form prompts so the Worker doesn't have to
# reverse-engineer the menu from scratch on every run.

def _extract_nav_prefix(steps: list[dict]) -> list[str]:
    """Return click/select target labels in order, up to (but not including)
    the first TYPE action. Each entry is a human-readable instruction line.
    Returns [] if no navigation steps found before the first TYPE."""
    if not steps:
        return []
    out: list[str] = []
    for s in steps:
        action = (s.get("action") or "").lower()
        if action in ("type", "input"):
            break  # form-fill begins here — stop collecting
        if action in ("click", "select"):
            label = (s.get("target_text") or s.get("target_description") or "").strip()
            if not label:
                continue
            label = label[:160]
            if action == "select":
                value = (s.get("value") or "").strip()
                if value:
                    out.append(f'SELECT "{value}" from "{label}"')
                    continue
            out.append(f'CLICK "{label}"')
        elif action == "navigate":
            url = (s.get("url") or "").strip()
            if url:
                out.append(f"NAVIGATE to {url[:160]}")
        elif action == "wait":
            ms = s.get("wait_ms") or 0
            if ms > 1500:
                out.append(f"WAIT {ms}ms")
    # Don't return excessively long prefixes — anything past 15 nav steps is
    # probably already mid-form, not "how to find the form."
    return out[:15]


def _format_nav_prefix(steps_list: list[list[str]]) -> str:
    """Render the longest distinct nav prefix from past playbooks as a
    recommended menu-click recipe. If multiple distinct prefixes exist,
    pick the longest (most informative)."""
    candidates = [s for s in steps_list if s]
    if not candidates:
        return ""
    canonical = max(candidates, key=len)
    lines = ["=== RECOMMENDED NAVIGATION PATH (from past task recordings) ==="]
    for i, step in enumerate(canonical, 1):
        lines.append(f"  {i}. {step}")
    lines.append("=== END NAVIGATION PATH ===")
    lines.append("Treat this as a known-good recipe for reaching the form. "
                 "Try these clicks in order before improvising. If a step doesn't "
                 "match what you see, fall back to discover_navigation for re-mapping.")
    return "\n".join(lines)


def format_playbook_for_prompt(steps: list[dict], user_data: dict, safety_mode: str = "halt-on-dangerous") -> str:
    """Convert playbook steps into numbered task instructions.

    safety_mode:
      - "auto": execute all steps including dangerous ones
      - "halt-on-dangerous": stop right before any dangerous step (default)
      - "dry-run": skip dangerous steps entirely with a warning
    """
    lines = ["Follow these steps exactly, one by one:"]
    for s in steps:
        idx = s.get("index", 0) + 1
        action = s.get("action", "click").upper()
        desc = s.get("target_description", "")
        value = _substitute(s.get("value", "") or "", user_data)
        url = _substitute(s.get("url", "") or "", user_data)
        is_dangerous = bool(s.get("dangerous"))
        danger_reason = s.get("danger_reason", "irreversible action")

        # Format the step line
        if action == "NAVIGATE" and url:
            line = f"{idx}. NAVIGATE to {url}"
        elif action == "TYPE" and value:
            label = s.get("target_text") or desc
            line = f"{idx}. TYPE '{value}' into the field: {label}"
        elif action == "SELECT" and value:
            label = s.get("target_text") or desc
            line = f"{idx}. SELECT '{value}' from: {label}"
        elif action == "WAIT":
            ms = s.get("wait_ms", 2000)
            line = f"{idx}. WAIT for {ms}ms — {desc}"
        elif action == "PRESS" and value:
            line = f"{idx}. PRESS key '{value}'"
        elif action == "UPLOAD" and value:
            line = f"{idx}. UPLOAD file '{value}' — {desc}"
        else:
            label = s.get("target_text") or desc
            line = f"{idx}. {action} — {label or desc}"

        # Apply safety mode markers
        if is_dangerous:
            if safety_mode == "halt-on-dangerous":
                lines.append(f"⚠️  STOP HERE — DO NOT EXECUTE STEPS {idx} AND BEYOND. ⚠️")
                lines.append(f"   Reason: {danger_reason}")
                lines.append(f"   The user will manually verify and complete the remaining steps.")
                lines.append(f"   (For your reference, the next step would be: {line})")
                break  # don't include further steps
            elif safety_mode == "dry-run":
                lines.append(f"{idx}. ⚠️  SKIP — would have done: {line[len(str(idx))+2:]}  (DRY RUN — irreversible: {danger_reason})")
                continue
            else:  # auto
                lines.append(f"{idx}. ⚠️  DANGEROUS — {line[len(str(idx))+2:]}  ({danger_reason})")
        else:
            lines.append(line)

    return "\n".join(lines)


# ── Bulk run mode ──────────────────────────────────────────────────────────────
# A single Python process iterates all rows of a bulk_run, reusing one browser
# session via browser-use's `add_new_task`. Coordinates with the backend via
# HTTP polling (next-row, result, heartbeat).

SESSIONS_DIR = Path.home() / ".declario" / "sessions"
BROWSER_RECYCLE_EVERY = 30   # recycle browser every N rows to bound memory growth
MAX_RELOGIN_ATTEMPTS = 3     # max re-login retries per row before marking failed
LOGIN_URL_RE = re.compile(r"/login|/auth|/signin|id\.gov\.ge|oauth", re.I)


def _session_state_path(key: str = DEFAULT_SESSION_KEY) -> Path:
    # Same tenant isolation as _user_data_dir — auth state must not leak
    # between companies.
    company_id = os.environ.get("AGENT_COMPANY_ID", "").strip() or "__shared__"
    state_dir = SESSIONS_DIR / company_id
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"{key}.json"


def _user_data_dir(key: str = DEFAULT_SESSION_KEY) -> Path:
    # Multi-tenant isolation: scope the Chromium profile by AGENT_COMPANY_ID so
    # two tenants on the same host never share rs.ge cookies/auth/local
    # storage. The backend's spawnBulkWorker injects AGENT_COMPANY_ID from the
    # bulk_runs.company_id when launching the worker.
    company_id = os.environ.get("AGENT_COMPANY_ID", "").strip() or "__shared__"
    base = SESSIONS_DIR / company_id / f"{key}_profile"
    base.mkdir(parents=True, exist_ok=True)
    return base


async def _backend_get(session: httpx.AsyncClient, path: str, **params) -> dict:
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:3001")
    resp = await session.get(f"{backend_url}{path}", params=params)
    resp.raise_for_status()
    return resp.json()


async def _backend_post(session: httpx.AsyncClient, path: str, payload: dict | None = None) -> dict:
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:3001")
    resp = await session.post(f"{backend_url}{path}", json=payload or {})
    resp.raise_for_status()
    return resp.json()


async def _heartbeat_loop(http: httpx.AsyncClient, run_id: str, interval: float = 15.0):
    """Background task: post heartbeat to backend so stalled-scanner stays happy."""
    while True:
        try:
            await _backend_post(http, f"/agent/bulk-runs/{run_id}/heartbeat")
        except Exception as exc:
            _emit("warn", f"heartbeat failed: {exc}")
        await asyncio.sleep(interval)


# ── Shared worker instruction blocks (single source of truth) ────────────────

def _authority_preamble(mode: str = "playbook") -> str:
    """Mode-specific data-integrity preamble lines."""
    is_free = mode == "free"
    lines = [
        ("IMPORTANT: Complete the task step by step following the instructions above."
         if is_free else
         "IMPORTANT: Follow the steps in the exact order shown above."),
        "Treat the authoritative spreadsheet block as immutable truth.",
        ("Spreadsheet values outrank site memory, KB hints, and inferred defaults."
         if is_free else
         "Spreadsheet values outrank memory, KB hints, and inferred defaults."),
        "Never invent a value that is not in the authoritative spreadsheet block.",
        "Never replace spreadsheet values with remembered values from previous runs.",
        "If a visible field label conflicts with the spreadsheet field map, STOP and report the conflict.",
    ]
    if not is_free:
        lines.append("Use the provided data wherever you see placeholders like ${variable}.")
        lines.append(
            "If a step is ambiguous about which option/value to use, "
            "follow the tactical plan that was prepended by the Planner."
        )
    return "\n".join(lines)


def _worker_rules_block(mode: str = "playbook") -> str:
    """Shared worker rules block: visual verification, grounding, failure handling.

    `mode` is 'playbook' (also used for bulk) or 'free'.
    """
    is_free = mode == "free"
    lines = [
        "VISUAL VERIFICATION RULES — apply after EVERY action:",
        "- After every CLICK or NAVIGATE: look at the screenshot and confirm the page actually changed.",
        "  If you clicked a menu/link but the page content did NOT change, you clicked the wrong element — try again.",
        "- After typing a value: look at the field in the screenshot and confirm the correct value appears.",
        "  If the field shows a different value, it did not work — try again.",
        "- After a dropdown selection: confirm the selected option is now visible in the dropdown.",
        "- Never assume an action succeeded just because no error was thrown. Always verify visually.",
        "",
        "POST-LOGIN SITE GROUNDING (do this FIRST after successful login, BEFORE clicking around the dashboard):",
        "- Call `discover_navigation` ONCE on the post-login dashboard. It returns a JSON list of every visible menu item (sidebar / top-bar / tabs / submenus) with Georgian labels and a hint for what each leads to.",
        "- Use that map to plan your menu clicks. Do NOT scroll-and-guess. If you can't find your target via menu, then try scrolling.",
        "- Cached per-domain — only the first call costs a Vision query.",
        "",
        "PRE-DONE READBACK (call ONCE before calling done() in halt-on-dangerous / dry-run mode):",
        "- Right BEFORE done(), call `verify_typed_values` ONCE with a JSON map of the labels you targeted and the values you typed.",
        '  Example: verify_typed_values({"(1) ანაზღაურების": "45800.00", "(2) წინასწარ": "2400.00"}).',
        "- If any field reports MISMATCH: click that field, press Ctrl+A, type the correct value, then call verify_typed_values ONE more time (HARD LIMIT: 2 total verify calls).",
        "- After 2 verify calls, proceed to done(success=True) regardless — note any unresolved fields in the done summary for human review.",
        "- If verify reports NO MATCHING FIELD for a label, the label query was wrong. Do NOT keep retrying with the same query — proceed to done() and note the unverifiable label.",
        "",
        "FORM-PAGE GROUNDING (do this FIRST when you arrive at a NEW form page):",
        "- Call `discover_form_fields` ONCE before typing. It returns a JSON list of every input with Georgian labels + CSS hints.",
        "- If `discover_form_fields` returns zero fields or an error on this URL, do NOT call it again unless the page visibly changed.",
        "- Use that list to map Excel columns to inputs. Do NOT guess. Quote the discovered label_ge in your reasoning.",
        "- Repeated calls on the same URL are cached. Different page → fresh discovery.",
        "",
        "FAILURE HANDLING RULES (strictly follow these):",
        "1. If you get 'No node with given id found' or element not found: the page re-rendered.",
        "   WAIT 2 seconds, scroll to top, re-scan the page, then retry with the correct new element.",
        "2. POST-TYPE VERIFICATION — the system automatically reads the DOM after each TYPE and emits a warning if the field's <input>.value does not match what you typed.",
        "   You do NOT need to inspect the screenshot after every TYPE. Trust the auto-check.",
        "   If a `field-reset` warning fires for a field, do NOT retype the same index blindly. First confirm the field is still editable or refresh the field mapping.",
        "   Anti-aliasing, focus rings, and SPA re-renders make screenshots an unreliable signal here. The DOM read is the source of truth.",
    ]
    if is_free:
        lines.extend([
            "2b. Never repeat the same click index or discover_form_fields call more than once without new evidence that the page changed.",
            "3. If you fail the SAME action 2 times in a row: STOP immediately.",
            "   Output exactly: 'STOPPED: [action you tried] failed [N] times. Reason: [error]. Cannot continue.'",
            "   Do NOT retry a third time — stop and report.",
            "4. If you cannot find a required element after re-scanning: STOP and report what is missing.",
            "",
            "At the end, summarise what you did and whether the task was completed successfully.",
        ])
    else:
        lines.extend([
            "3. For SELECT / dropdown steps: the site uses CUSTOM dropdowns (not native <select>).",
            "   To interact: CLICK the dropdown label/button to open it, then CLICK the desired option from the list.",
            "   Do NOT use select_dropdown — use click actions instead.",
            "3b. Never repeat the same click index or discover_form_fields call more than once without new evidence that the page changed.",
            "4. If you fail the SAME action 2 times in a row on a NON-CRITICAL step (filter, dropdown, search):",
            "   Skip it, note 'SKIPPED step N: [reason]', and continue to the next step.",
            "   If it is a CRITICAL step (login, form fill, submit): STOP and report.",
            "5. If you cannot find a required element after re-scanning: use your own judgment.",
            "   Look at the screenshot — find the visually closest matching element and try that instead.",
            "   Do not loop on the same failed approach more than 2 times.",
            "",
            "At the end, summarise what you did and whether all steps completed successfully.",
        ])
    return "\n".join(lines)


def _build_task_prompt(
    playbook_steps: list[dict],
    merged_data: dict,
    safety_mode: str,
    task_label: str | None,
    knowledge_block: str = "",
) -> str:
    """Build a single-row task prompt reusing the existing helpers.

    `knowledge_block` is the formatted output of `format_context_for_prompt()`;
    bulk callers fetch it once per playbook (not per row) and pass it in.
    """
    steps_block = format_playbook_for_prompt(playbook_steps, merged_data, safety_mode)
    field_map = _extract_field_map(playbook_steps)
    field_map_block = _format_field_map(field_map, merged_data)
    data_contract = _build_authoritative_data_contract(merged_data, field_map, task_label or "")
    authority_block = _format_authoritative_data_block(data_contract)

    safety_instructions = {
        "halt-on-dangerous": (
            "SAFETY MODE: HALT-ON-DANGEROUS. "
            "When you reach the marked '⚠️ STOP HERE' line, IMMEDIATELY stop. "
            "Do NOT click submit/send/confirm/pay buttons. "
            "When the form is filled and you've stopped before the irreversible "
            "step, the task is COMPLETE — call done(success=True) with a summary "
            "describing what's ready for human review. Stopping cleanly is success in this mode."
        ),
        "dry-run": (
            "SAFETY MODE: DRY-RUN. Skip every step marked '⚠️ SKIP'. "
            "Do NOT click submit/send/confirm/pay/delete buttons. "
            "When you've reached the pre-submit stage, call done(success=True)."
        ),
        "auto": "SAFETY MODE: AUTO. All steps will be executed including dangerous ones.",
    }.get(safety_mode, "")

    return "\n".join(filter(None, [
        "Complete the following task on the web portal step by step.",
        task_label or "",
        "",
        field_map_block,
        "",
        steps_block,
        "",
        authority_block,
        "",
        safety_instructions,
        "",
        _authority_preamble("playbook"),
        "",
        _worker_rules_block("playbook"),
    ]))


# ── Per-row audit helpers ──────────────────────────────────────────────────────

def _compress_to_jpeg(data: bytes, quality: int = 70) -> bytes:
    """Compress raw image bytes to JPEG q70. Returns original bytes on any error."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except Exception:
        return data


def _save_screenshot_frame(
    data: bytes,
    run_id: str,
    row_index: int,
    step_index: int,
    frame_type: str,
    recordings_base: Path,
) -> str | None:
    """Compress and persist one frame; returns a URL/path string or None."""
    jpeg = _compress_to_jpeg(data)
    fname = f"step_{step_index:04d}_{frame_type}.jpg"
    s3_bucket = os.environ.get("S3_BUCKET", "")

    if s3_bucket:
        try:
            import boto3
            s3_endpoint = os.environ.get("S3_ENDPOINT", "")
            s3_region = os.environ.get("S3_REGION", "auto")
            s3_key_id = os.environ.get("S3_ACCESS_KEY_ID", "")
            s3_secret = os.environ.get("S3_SECRET_ACCESS_KEY", "")
            s3_client = boto3.client(
                "s3",
                endpoint_url=s3_endpoint or None,
                aws_access_key_id=s3_key_id or None,
                aws_secret_access_key=s3_secret or None,
                region_name=s3_region,
            )
            key = f"bulk-runs/{run_id}/row_{row_index:04d}/{fname}"
            s3_client.put_object(Bucket=s3_bucket, Key=key, Body=jpeg, ContentType="image/jpeg")
            base = s3_endpoint.rstrip("/") if s3_endpoint else f"https://s3.{s3_region}.amazonaws.com"
            return f"{base}/{s3_bucket}/{key}"
        except Exception as exc:
            _emit("warn", f"S3 upload failed ({fname}): {exc}")

    row_dir = recordings_base / f"row_{row_index:04d}"
    row_dir.mkdir(parents=True, exist_ok=True)
    local = row_dir / fname
    try:
        local.write_bytes(jpeg)
        return str(local)
    except Exception as exc:
        _emit("warn", f"local screenshot save failed: {exc}")
        return None


def _build_row_result(
    history,
    run_id: str,
    row_index: int,
    recordings_base: Path,
    safety_mode: str = "halt-on-dangerous",
) -> dict:
    """Translate browser-use AgentHistoryList into a backend payload with screenshot audit."""
    succeeded = bool(history.is_successful()) and not history.has_errors()
    final = history.final_result()

    if not succeeded and history.has_errors():
        errs = history.errors()
        last_err = errs[-1] if errs else "unknown error"
        message = f"Agent stopped with errors. Last error: {last_err}"
    else:
        message = final or "Agent finished without explicit result"

    # ── Action log ─────────────────────────────────────────────────────────────
    action_log: list[dict] = []
    try:
        actions = history.model_actions() or []
        urls = history.urls() or []
        errors_list = history.errors() or []
        for i, act in enumerate(actions):
            action_log.append({
                "step": i + 1,
                "action": str(act)[:500] if act else "",
                "url": urls[i] if i < len(urls) else None,
                "error": str(errors_list[i])[:200] if i < len(errors_list) and errors_list[i] else None,
            })
    except Exception as exc:
        _emit("warn", f"action_log build failed: {exc}")

    # ── Screenshots: first + last + error frames only ──────────────────────────
    screenshots: list[dict] = []
    try:
        all_shots = history.screenshots() or []
        errors_list_s = history.errors() or []
        n = len(all_shots)
        if n > 0:
            keep: list[tuple[int, str]] = [(0, "first")]
            for i, e in enumerate(errors_list_s):
                if e and 0 < i < n - 1:
                    keep.append((i, "error"))
            if n > 1:
                keep.append((n - 1, "last"))

            for step_idx, frame_type in keep:
                raw = all_shots[step_idx]
                if raw is None:
                    continue
                img_bytes: bytes
                if isinstance(raw, str):
                    img_bytes = base64.b64decode(raw)
                elif isinstance(raw, bytes):
                    img_bytes = raw
                else:
                    continue
                url = _save_screenshot_frame(img_bytes, run_id, row_index, step_idx, frame_type, recordings_base)
                if url:
                    screenshots.append({"step": step_idx + 1, "url": url, "type": frame_type})
    except Exception as exc:
        _emit("warn", f"screenshot processing failed: {exc}")

    return {
        "status": "success" if succeeded else "failed",
        "completionState": _default_completion_state(safety_mode) if succeeded else COMPLETION_FAILED,
        "error": None if succeeded else message,
        "stepsTaken": history.number_of_steps(),
        "screenshots": screenshots,
        "actionLog": action_log,
    }


# ── Re-login helpers ───────────────────────────────────────────────────────────

async def _count_login_signals(agent, consecutive_failures: int) -> int:
    """
    Return the count of active login signals (0–3):
      A) Current URL matches a login pattern
      B) A password input is visible on the page
      C) 2+ consecutive action failures
    """
    count = 0

    # Signal A: URL
    try:
        session = getattr(agent, "browser_session", None)
        if session:
            url: str | None = None
            for method_name in ("get_current_url", "cdp_get_url", "current_url"):
                fn = getattr(session, method_name, None)
                if fn is None:
                    continue
                try:
                    url = await _call_maybe_async(fn)
                except Exception:
                    pass
                if url:
                    break
            if url and LOGIN_URL_RE.search(str(url)):
                count += 1
    except Exception:
        pass

    # Signal B: password input visible
    try:
        session = getattr(agent, "browser_session", None)
        if session:
            page = await _resolve_session_page(session)
            if page is not None:
                if _page_supports_playwright_locators(page):
                    n_pw = await page.locator('input[type="password"]').count()
                else:
                    raw = await _safe_page_eval(page, "() => document.querySelectorAll('input[type=\"password\"]').length")
                    n_pw = int(raw or 0)
                if n_pw > 0:
                    count += 1
    except Exception:
        pass

    # Signal C: consecutive failures
    if consecutive_failures >= 2:
        count += 1

    return count


async def _fetch_login_playbook(http: httpx.AsyncClient) -> dict | None:
    """Fetch the designated login playbook (kind='login') from the backend."""
    try:
        data = await _backend_get(http, "/playbooks/login")
        return data.get("playbook")
    except Exception:
        return None


async def _run_relogin(agent, http: httpx.AsyncClient, shared_data: dict) -> bool:
    """Run the login playbook to refresh the browser session. Returns True on success."""
    login_pb = await _fetch_login_playbook(http)
    if not login_pb or not login_pb.get("steps"):
        _emit("warn", "Re-login needed but no login playbook configured")
        return False

    task_str = _build_task_prompt(login_pb["steps"], shared_data, "auto", "Re-login: restore session")
    agent.add_new_task(task_str)
    try:
        _reset_emitted_steps()
        history = await agent.run(max_steps=20)
        if history.is_successful() or not history.has_errors():
            _emit("info", "Re-login succeeded")
            return True
    except Exception as exc:
        _emit("warn", f"Re-login exception: {exc}")
    _emit("warn", "Re-login attempt did not succeed")
    return False


async def run_bulk(run_id: str):
    """Single Python process executes all rows of a bulk run via one browser session."""
    from browser_use import Agent, BrowserProfile

    worker_id = str(uuid.uuid4())
    _emit("info", f"Bulk worker {worker_id[:8]} started for run {run_id[:8]}")

    # Per-company headless override: set AGENT_HEADLESS_COMPANIES="acme,zorba"
    # to keep those companies visible-mode even when the global default is
    # headless. Helps tenants that hit rs.ge captchas regularly and need a
    # human at the keyboard.
    global_headless = os.environ.get("AGENT_HEADLESS", "false").lower() == "true"
    company_id = os.environ.get("AGENT_COMPANY_ID", "").strip()
    override = {
        c.strip().lower()
        for c in os.environ.get("AGENT_HEADLESS_COMPANIES", "").split(",")
        if c.strip()
    }
    if company_id and company_id.lower() in override:
        headless = False
        _emit("info", f"company {company_id} forced to non-headless (captcha override)")
    else:
        headless = global_headless
    recordings_base = Path(__file__).parent / "recordings" / run_id

    state_path: Path | None = None
    profile_kwargs: dict | None = None

    def configure_profile(run_config: dict) -> None:
        nonlocal state_path, profile_kwargs
        if profile_kwargs is not None:
            return
        session_key = _safe_session_key(run_config.get("sessionKey") or DEFAULT_SESSION_KEY)
        allowed_domains = _coerce_allowed_domains(run_config.get("allowedDomains"))
        state_path = _session_state_path(session_key)
        profile_dir = _user_data_dir(session_key)
        profile_kwargs = {
            "headless": headless,
            "demo_mode": not headless,
            "user_data_dir": str(profile_dir),
            "allowed_domains": allowed_domains,
            "cookie_whitelist_domains": [d for d in allowed_domains if "*" not in d],
        }
        if state_path.exists():
            profile_kwargs["storage_state"] = str(state_path)
            _emit("info", f"Restored session storage_state from {state_path}")
        _emit("info", f"Bulk browser policy: domains={_domains_for_prompt(allowed_domains)}, session={session_key}")

    def make_profile() -> "BrowserProfile":
        assert profile_kwargs is not None
        kw = dict(profile_kwargs)
        if state_path and state_path.exists():
            kw["storage_state"] = str(state_path)
        return BrowserProfile(**kw)

    browser_profile = None
    llm = build_llm()

    agent: "Agent | None" = None
    agent_tools = None
    rows_completed_in_session = 0
    consecutive_failures = 0
    # Per-playbook RAG cache — fetched once on first encounter, reused across rows.
    knowledge_by_playbook: dict[str, str] = {}
    knowledge_disabled = os.environ.get("KNOWLEDGE_DISABLE") == "1"
    # Free-form field map + nav-prefix cache — fetched once across the bulk run.
    freeform_field_map: dict[str, str] | None = None
    freeform_nav_prefix_block: str | None = None
    # Phase Q: site-memory + failure-patterns cache, keyed by inferred portal.
    # Pulled once per portal then reused for every row touching that portal.
    freeform_site_memory_by_portal: dict[str, str] = {}
    freeform_failure_block_by_portal: dict[str, str] = {}

    async with httpx.AsyncClient(timeout=30, headers=_backend_auth_headers()) as http:
        heartbeat_task = asyncio.create_task(_heartbeat_loop(http, run_id))

        try:
            while True:
                try:
                    payload = await _backend_get(
                        http, f"/agent/bulk-runs/{run_id}/next-row", workerId=worker_id
                    )
                except httpx.HTTPError as exc:
                    _emit("error", f"next-row failed: {exc}")
                    break

                if not payload.get("row"):
                    _emit("info", f"No more rows (runStatus={payload.get('runStatus', '')})")
                    break

                row = payload["row"]
                playbook = payload.get("playbook")
                run_config = payload.get("config") or {}
                configure_profile(run_config)
                if browser_profile is None:
                    browser_profile = make_profile()

                merged = (row.get("data") or {}).get("merged") or {}
                safety_mode = run_config.get("safetyMode") or "halt-on-dangerous"
                safety_mode = safety_mode if safety_mode in SAFETY_MODES else "halt-on-dangerous"
                allowed_domains = _coerce_allowed_domains(run_config.get("allowedDomains"))
                max_steps = int(run_config.get("maxSteps") or os.environ.get("AGENT_MAX_STEPS", 50))
                task_label = run_config.get("task")
                _row_portal = _infer_expected_portal(task_label or "", (playbook or {}).get("steps") or [])

                # Require either a playbook with steps OR a free-form task description.
                if (not playbook or not playbook.get("steps")) and not task_label:
                    await _backend_post(
                        http,
                        f"/agent/bulk-runs/{run_id}/rows/{row['row_index']}/result",
                        {"status": "failed", "error": "playbook missing or empty and no task provided",
                         "stepsTaken": 0, "screenshots": [], "actionLog": []},
                    )
                    consecutive_failures += 1
                    continue

                # Per-playbook or per-task RAG fetch.
                # Playbook mode: keyed by playbook ID. Freeform: keyed by "__freeform__".
                pb_id = (playbook.get("id") or "") if playbook else ""
                cache_key = pb_id or "__freeform__"
                if not knowledge_disabled and cache_key not in knowledge_by_playbook:
                    if pb_id:
                        target_descriptions = " ".join(
                            (s.get("target_description") or "") for s in playbook["steps"]
                            if s.get("target_description")
                        )
                        rag_query = ((task_label or "") + " " + target_descriptions).strip() or "rs.ge tax declaration"
                        _emit("info", f"Fetching accounting knowledge for playbook {playbook.get('name', '?')}…")
                    else:
                        rag_query = task_label or "rs.ge tax declaration"
                        _emit("info", "Fetching accounting knowledge for free-form task…")
                    chunks = await fetch_context(rag_query, limit=8)
                    if chunks:
                        knowledge_by_playbook[cache_key] = format_context_for_prompt(chunks)
                        _emit("info", f"Cached {len(chunks)} knowledge-base chunk(s)")
                    else:
                        knowledge_by_playbook[cache_key] = ""

                knowledge_block = knowledge_by_playbook.get(cache_key, "")

                if playbook and playbook.get("steps"):
                    _row_field_map = _extract_field_map(playbook["steps"])
                    _row_contract = _build_authoritative_data_contract(merged, _row_field_map, task_label or "")
                    _emit_contract_summary(f"row {row['row_index']} playbook prompt", _row_contract)
                    task_str = _build_task_prompt(playbook["steps"], merged, safety_mode, task_label, knowledge_block)
                    _emit("info", f"Row {row['row_index']} starting (playbook={playbook.get('name', '?')})")
                else:
                    # Lazy-load the field map + nav prefix once for the bulk run.
                    if freeform_field_map is None or freeform_nav_prefix_block is None:
                        try:
                            _all_steps = await fetch_all_task_playbook_steps()
                            freeform_field_map = (
                                _merge_field_maps([_extract_field_map(s) for s in _all_steps])
                                if _all_steps else {}
                            )
                            freeform_nav_prefix_block = _format_nav_prefix(
                                [_extract_nav_prefix(s) for s in (_all_steps or [])]
                            )
                            if freeform_field_map:
                                _emit("info", f"Loaded field map ({len(freeform_field_map)} entries) for free-form rows")
                            if freeform_nav_prefix_block:
                                _emit("info", "Loaded nav prefix from existing playbooks for free-form rows")
                        except Exception:
                            freeform_field_map = {}
                            freeform_nav_prefix_block = ""
                    _ff_field_block = _format_field_map(freeform_field_map, merged)
                    _row_contract = _build_authoritative_data_contract(merged, freeform_field_map, task_label or "")
                    _emit_contract_summary(f"row {row['row_index']} free-form prompt", _row_contract)
                    _authority_block = _format_authoritative_data_block(_row_contract)

                    # Phase Q: site memory + failure-patterns per inferred portal.
                    _site_memory_block = ""
                    _failure_patterns_block = ""
                    if _row_portal:
                        if _row_portal not in freeform_site_memory_by_portal:
                            try:
                                _mem = await fetch_site_memory(_row_portal)
                                freeform_site_memory_by_portal[_row_portal] = (
                                    _format_site_memory_for_worker(_mem, _row_portal) if _mem else ""
                                )
                                if freeform_site_memory_by_portal[_row_portal]:
                                    _emit("info", f"Q1: site memory loaded for {_row_portal}")
                            except Exception:
                                freeform_site_memory_by_portal[_row_portal] = ""
                        if _row_portal not in freeform_failure_block_by_portal:
                            try:
                                _patts = await fetch_failure_patterns(_row_portal)
                                freeform_failure_block_by_portal[_row_portal] = (
                                    _format_failure_patterns_for_worker(_patts) if _patts else ""
                                )
                                if freeform_failure_block_by_portal[_row_portal]:
                                    _emit("info", f"Q2: {len(_patts)} failure pattern(s) loaded for {_row_portal}")
                            except Exception:
                                freeform_failure_block_by_portal[_row_portal] = ""
                        _site_memory_block = freeform_site_memory_by_portal.get(_row_portal, "")
                        _failure_patterns_block = freeform_failure_block_by_portal.get(_row_portal, "")

                    # R2: bulk free-form Worker drops the raw RAG chunks too;
                    # accounting reasoning belongs in the Planner pass that runs
                    # in playbook mode, not in every per-row Worker prompt.
                    task_str = "\n".join(filter(None, [
                        f"TASK: {task_label}",
                        _freeform_policy_block(safety_mode, allowed_domains),
                        _site_memory_block,
                        _failure_patterns_block,
                        freeform_nav_prefix_block or "",
                        _ff_field_block,
                        _authority_block,
                        "",
                        _authority_preamble("free"),
                        "",
                        _worker_rules_block("free"),
                    ]))
                    _emit("info", f"Row {row['row_index']} starting (free-form task)")

                # Inner retry loop: handles re-login + row retry
                relogin_attempts = 0
                while True:
                    try:
                        if agent is None:
                            agent_tools = _build_tools(
                                task_hint=task_label or task_str[:200],
                                safety_mode=safety_mode,
                                allowed_domains=allowed_domains,
                                mode="playbook" if playbook else "free",
                            )
                            agent = Agent(
                                task=task_str,
                                llm=llm,
                                browser_profile=browser_profile,
                                use_vision=True,
                                tools=agent_tools,
                                register_new_step_callback=on_step,
                                register_done_callback=on_done,
                                max_actions_per_step=1,
                            )
                        else:
                            agent.add_new_task(task_str)

                        _reset_emitted_steps()
                        history = await agent.run(max_steps=max_steps)
                        result = _build_row_result(
                            history,
                            run_id,
                            row["row_index"],
                            recordings_base,
                            safety_mode,
                        )

                        # Q5b — Halt-on-dangerous self-success guard for bulk rows.
                        # Worker's done(success=False) is wrong when the form is
                        # filled and we just stopped before submit. K1 (typed-
                        # values check) is the real success signal in halt mode;
                        # has_errors() is too noisy to gate on (recoverable
                        # element-misses still flip it).
                        if (
                            result["status"] != "success"
                            and safety_mode in ("halt-on-dangerous", "dry-run")
                        ):
                            _missing_for_guard = _check_authoritative_contract_coverage(history, _row_contract)
                            if not _missing_for_guard:
                                _emit(
                                    "info",
                                    f"Row {row['row_index']}: halt-mode self-success guard — "
                                    "Worker reported done(success=False) but every expected value "
                                    "was typed; treating clean stop as success.",
                                )
                                result["status"] = "success"
                                result["completionState"] = _default_completion_state(safety_mode)
                                if not result.get("message"):
                                    result["message"] = "Form filled and stopped before submit (auto-promoted)."

                        # K1: Post-run typed-value verification for bulk rows. Runs
                        # before M1 because it's deterministic + free, and a missing
                        # value means we don't even need to ask Vision.
                        if result["status"] == "success":
                            _k1_missing = _check_authoritative_contract_coverage(history, _row_contract)
                            if _k1_missing:
                                _preview = ", ".join(str(v.get("key")) for v in _k1_missing[:8])
                                _more = f" (+{len(_k1_missing) - 8} more)" if len(_k1_missing) > 8 else ""
                                _emit(
                                    "warn",
                                    f"Row {row['row_index']}: ⚠ Hallucinated success — never typed required spreadsheet fields: "
                                    f"[{_preview}{_more}]",
                                )
                                result["status"] = "failed"
                                result["completionState"] = COMPLETION_FAILED
                                result["error"] = (
                                    f"K1: agent never typed required spreadsheet field(s): [{_preview}{_more}]"
                                )
                                # Q3: feed failure into the learning loop.
                                if _row_portal:
                                    asyncio.create_task(report_failure_pattern(
                                        domain=_row_portal,
                                        failure_type="k1_hallucination",
                                        symptom=f"Agent claimed success but never typed required spreadsheet fields: [{_preview}{_more}]",
                                        workaround="Use the authoritative spreadsheet block and verify every required field before success.",
                                    ))

                        # K1-rows: every tabular row (e.g. each payroll employee)
                        # must have its UNIQUE identifier in the typed history.
                        # Catches "agent submitted but skipped/hallucinated some
                        # employees" — the exact hallucination we must exclude.
                        if result["status"] == "success":
                            _rows_missing = _check_row_coverage(history, _row_contract)
                            if _rows_missing:
                                _rpreview = ", ".join(str(v.get("label")) for v in _rows_missing[:8])
                                _rmore = f" (+{len(_rows_missing) - 8} more)" if len(_rows_missing) > 8 else ""
                                _emit(
                                    "warn",
                                    f"Row {row['row_index']}: ⚠ Incomplete — {len(_rows_missing)} employee row(s) "
                                    f"never entered: [{_rpreview}{_rmore}]",
                                )
                                result["status"] = "failed"
                                result["completionState"] = COMPLETION_FAILED
                                result["error"] = (
                                    f"K1-rows: {len(_rows_missing)} employee row(s) not entered/verified: "
                                    f"[{_rpreview}{_rmore}]"
                                )
                                if _row_portal:
                                    asyncio.create_task(report_failure_pattern(
                                        domain=_row_portal,
                                        failure_type="k1_rows_incomplete",
                                        symptom=f"Declaration submitted/attempted but {len(_rows_missing)} employee row(s) were never entered: [{_rpreview}{_rmore}]",
                                        workaround="Enter EVERY employee row from the ROWS TO ENTER block; verify each personal_id is typed before submitting.",
                                    ))

                        if result["status"] == "success":
                            _dom_ok, _review_payload, _dom_error = await _run_dom_postcondition(
                                await _get_current_page_from_agent(agent),
                                merged,
                                _row_field_map if playbook and playbook.get("steps") else freeform_field_map,
                                _row_portal,
                                _row_contract,
                                fallback_snapshot=None,
                            )
                            if not _dom_ok:
                                result["status"] = "failed"
                                result["completionState"] = COMPLETION_FAILED
                                result["error"] = f"DOM postcondition failed: {_dom_error}"
                            else:
                                _contract_missing = _check_required_contract_items_verified(
                                    _row_contract,
                                    getattr(agent_tools, "_declario_typed_log", None),
                                    _review_payload,
                                )
                                if _contract_missing:
                                    _preview = ", ".join(str(v.get("key")) for v in _contract_missing[:8])
                                    _more = f" (+{len(_contract_missing) - 8} more)" if len(_contract_missing) > 8 else ""
                                    result["status"] = "failed"
                                    result["completionState"] = COMPLETION_FAILED
                                    result["error"] = (
                                        f"Spreadsheet contract not fully verified: [{_preview}{_more}]"
                                    )

                        # M1: Visual Validator for bulk rows (same logic as single-run).
                        if result["status"] == "success":
                            _m1_ss = history.screenshots() or []
                            if _m1_ss:
                                try:
                                    _emit("info", f"Row {row['row_index']}: running visual validator…")
                                    _vr = await _visual_validator(_m1_ss[-1], merged, "", safety_mode)
                                    _zf = _vr.get("suspicious_zero_fields") or []
                                    _ok, _completion_state, _verr = _visual_outcome_for_safety(safety_mode, _vr)
                                    result["completionState"] = _completion_state
                                    if _vr.get("reg_number"):
                                        _emit("info", f"Row {row['row_index']}: ✓ reg number: {_vr['reg_number']}")
                                    if _zf:
                                        _emit("warn",
                                              f"Row {row['row_index']}: ⚠ Visual validator: "
                                              f"{len(_zf)} field(s) show 0.00: {_zf[:5]}")
                                        result["status"] = "failed"
                                        result["completionState"] = COMPLETION_FAILED
                                        result["error"] = f"Visual validator: fields show 0.00: {_zf[:5]}"
                                        # Q3: report each zero-field as a separate pattern
                                        # so the failure-list groups by label.
                                        if _row_portal:
                                            for _zlabel in _zf[:5]:
                                                asyncio.create_task(report_failure_pattern(
                                                    domain=_row_portal,
                                                    failure_type="m1_zero_field",
                                                    field_label=str(_zlabel)[:200],
                                                    symptom=f"Field '{_zlabel}' shows 0.00 after typing — likely got reset.",
                                                    workaround="Click the field, press Ctrl+A, then re-type.",
                                                ))
                                    elif not _ok:
                                        # The submit fired and the deterministic gates (K1 typed
                                        # values + DOM postcondition) already passed; only the
                                        # screenshot heuristic is unsure. In real-submission
                                        # ("auto") mode the data has almost certainly reached
                                        # rs.ge, so DON'T raise a false "failed" — record it as
                                        # submitted-but-unconfirmed and flag it for a quick human
                                        # check. (In halt/dry-run modes a non-confirmation screen
                                        # is still a real stop, so keep the failure there.)
                                        if safety_mode == "auto":
                                            _emit("warn",
                                                  f"Row {row['row_index']}: ⚠ couldn't visually confirm the "
                                                  f"confirmation page, but typed values + DOM checks passed — "
                                                  f"marking SUBMITTED (needs verification), not failed.")
                                            result["status"] = "success"
                                            result["completionState"] = COMPLETION_NEEDS_REVIEW
                                            result["needsVerification"] = True
                                            result["verificationNote"] = (
                                                "Submitted, but the confirmation page could not be visually "
                                                f"verified — please double-check on rs.ge. {_vr.get('explanation', '')[:120]}"
                                            )
                                        else:
                                            _emit("warn",
                                                  f"Row {row['row_index']}: ⚠ Visual validator: "
                                                  f"not a confirmation page. ({_vr.get('explanation', '')[:120]})")
                                            result["status"] = "failed"
                                            result["completionState"] = COMPLETION_FAILED
                                            result["error"] = (
                                                _verr or (
                                                    f"Visual validator: not a confirmation page. "
                                                    f"{_vr.get('explanation', '')[:120]}"
                                                )
                                            )
                                            if _row_portal:
                                                asyncio.create_task(report_failure_pattern(
                                                    domain=_row_portal,
                                                    failure_type="m1_not_confirmation",
                                                    symptom=f"Not a confirmation page after run. {_vr.get('explanation','')[:300]}",
                                                    workaround="Verify the submit/finalise step actually fired; check for stuck loading spinner or popup.",
                                                ))
                                    else:
                                        _emit("info", f"Row {row['row_index']}: visual validator accepted ({_completion_state}).")
                                except Exception as _ve:
                                    _emit("warn", f"Row {row['row_index']}: visual validator error (non-fatal): {_ve}")
                    except Exception as exc:
                        result = {
                            "status": "failed",
                            "completionState": COMPLETION_FAILED,
                            "error": str(exc),
                            "stepsTaken": 0,
                            "screenshots": [],
                            "actionLog": [],
                        }

                    if result["status"] == "failed":
                        consecutive_failures += 1
                    else:
                        consecutive_failures = 0

                    # 3-signal re-login detection
                    if agent is not None and relogin_attempts < MAX_RELOGIN_ATTEMPTS:
                        signals = await _count_login_signals(agent, consecutive_failures)
                        if signals >= 2:
                            relogin_attempts += 1
                            _emit(
                                "info",
                                f"Row {row['row_index']}: {signals} login signals, "
                                f"re-login attempt {relogin_attempts}/{MAX_RELOGIN_ATTEMPTS}",
                            )
                            ok = await _run_relogin(agent, http, merged)
                            if ok:
                                consecutive_failures = 0
                                continue  # retry original row task
                            result["status"] = "failed"
                            result["error"] = (
                                f"session expired; re-login failed after {relogin_attempts} attempt(s)"
                            )
                    break  # no re-login triggered, or max attempts reached

                # Tell backend the current attempt count so it can decide
                # whether to requeue this row for an automatic retry.
                result["attemptCount"] = int(row.get("attempt_count") or 0)

                requeued = False
                stopped_on_failure = False
                try:
                    resp_json = await _backend_post(
                        http,
                        f"/agent/bulk-runs/{run_id}/rows/{row['row_index']}/result",
                        result,
                    )
                    stopped_on_failure = bool(isinstance(resp_json, dict) and resp_json.get("stoppedOnFailure"))
                    if isinstance(resp_json, dict) and resp_json.get("requeued"):
                        requeued = True
                        next_attempt = resp_json.get("nextAttempt")
                        _emit(
                            "info",
                            f"Row {row['row_index']} requeued for retry "
                            f"(attempt {next_attempt}) due to transient error",
                        )
                except httpx.HTTPError as exc:
                    _emit("error", f"result post failed: {exc}")

                rows_completed_in_session += 1
                _emit(
                    "info",
                    f"Row {row['row_index']} → "
                    f"{('requeued' if requeued else result['status'])} "
                    f"(in-session: {rows_completed_in_session})",
                )

                # Browser lifecycle: recycle every BROWSER_RECYCLE_EVERY rows
                if rows_completed_in_session >= BROWSER_RECYCLE_EVERY and agent is not None:
                    _emit("info", f"Recycling browser after {rows_completed_in_session} rows…")
                    try:
                        if state_path:
                            await _export_storage_state_if_possible(
                                agent.browser_session,
                                state_path,
                                context="pre-recycle",
                            )
                    except Exception as exc:
                        _emit("warn", f"pre-recycle state export failed: {exc}")
                    try:
                        await agent.browser_session.close()
                    except Exception:
                        pass
                    browser_profile = make_profile()
                    agent = None
                    rows_completed_in_session = 0

                if stopped_on_failure:
                    _emit("warn", "Stopping bulk worker because stopOnFailure was triggered")
                    break

        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

            if agent is not None and getattr(agent, "browser_session", None) is not None:
                try:
                    if state_path:
                        await _export_storage_state_if_possible(
                            agent.browser_session,
                            state_path,
                            context="bulk-finally",
                        )
                except Exception as exc:
                    _emit("warn", f"storage_state export failed: {exc}")

            try:
                await _backend_post(http, f"/agent/bulk-runs/{run_id}/finalize")
            except httpx.HTTPError as exc:
                _emit("warn", f"finalize failed: {exc}")


async def run_agent(
    task: str,
    user_data: dict,
    max_steps: int,
    record: bool,
    playbook_id: str = "",
    safety_mode: str = "halt-on-dangerous",
    allowed_domains: list[str] | None = None,
    session_key: str = DEFAULT_SESSION_KEY,
    mode: str = "free",
):
    from browser_use import Agent, BrowserProfile

    safety_mode = safety_mode if safety_mode in SAFETY_MODES else "halt-on-dangerous"
    mode = mode if mode in AGENT_MODES else ("playbook" if playbook_id else "free")
    if playbook_id and mode == "free":
        mode = "playbook"
    allowed_domains = _coerce_allowed_domains(allowed_domains)
    session_key = _safe_session_key(session_key)

    # Defined here so the on_step_start closure (declared later in this fn)
    # always has a binding regardless of which branch builds the prompt.
    planner_disabled = os.environ.get("PLANNER_DISABLE") == "1"
    # plan_text is set in the playbook branch; free-form runs leave it empty.
    # Defined here so P2 plan-preview check (line ~2048) doesn't UnboundLocalError.
    plan_text = ""
    postcondition_field_map: dict[str, str] = {}
    expected_portal: str | None = None
    data_contract: dict = {"items": [], "summary": {}}

    if playbook_id:
        # Cache-first execution: replay saved steps deterministically.
        # Only fall through to the AI Planner+Worker path when reason=="no_cache"
        # (first ever run for this playbook). Any other failure (step blocked,
        # cancelled, postcondition) is surfaced as an error — we never let the
        # free AI agent deviate from a known-good playbook.
        cache_disabled = os.environ.get("CACHE_DISABLE") == "1"
        if not cache_disabled:
            cache_headless = os.environ.get("AGENT_HEADLESS", "false").lower() == "true"
            try:
                replayed = await run_from_cache(
                    playbook_id,
                    user_data,
                    headless=cache_headless,
                    safety_mode=safety_mode,
                )
            except Exception as exc:
                _emit("warn", f"Cache replay error: {exc}")
                replayed = {"success": False, "reason": "no_cache", "completion_state": COMPLETION_FAILED}

            if replayed.get("success"):
                replay_completion_state = replayed.get("completion_state") or _default_completion_state(safety_mode)
                _emit(
                    "result",
                    "Task completed via cached replay (no LLM calls).",
                    success=True,
                    recording=None,
                    steps_taken=0,
                    completion_state=replay_completion_state,
                )
                return True

            replay_reason = replayed.get("reason", "no_cache")

            if replay_reason == "cancelled":
                # Operator explicitly cancelled a blocked step — stop cleanly.
                _emit(
                    "result",
                    "Playbook replay cancelled by operator.",
                    success=False,
                    recording=None,
                    steps_taken=0,
                    completion_state=COMPLETION_FAILED,
                )
                return False

            if replay_reason == "postcondition_failed":
                # Replay ran all steps but final validation failed — report and stop.
                _emit(
                    "result",
                    "Playbook replay completed but postcondition validation failed.",
                    success=False,
                    recording=None,
                    steps_taken=0,
                    completion_state=COMPLETION_FAILED,
                )
                return False

            # replay_reason == "no_cache" → first run for this playbook, fall through
            # to the AI recording path below.
            if replay_reason != "no_cache":
                # Unknown non-success reason — do not start free AI.
                _emit(
                    "result",
                    f"Playbook replay failed ({replay_reason}). AI deviation is disabled in playbook mode.",
                    success=False,
                    recording=None,
                    steps_taken=0,
                    completion_state=COMPLETION_FAILED,
                )
                return False

        # Playbook mode: fetch steps and build structured prompt
        _emit("info", f"Fetching playbook {playbook_id}…")
        steps = await fetch_playbook_steps(playbook_id)
        if not steps:
            _emit("error", "Playbook has no steps or could not be fetched")
            return False

        _emit("info", f"Loaded {len(steps)} playbook steps")
        dangerous_count = sum(1 for s in steps if s.get("dangerous"))
        if dangerous_count:
            _emit("warn", f"Playbook contains {dangerous_count} dangerous step(s). Safety mode: {safety_mode}")

        # ── Auto-login prelude ────────────────────────────────────────────────
        # Many playbooks are recorded mid-flow and assume the user is already
        # logged in. If credentials are provided AND a 'login' playbook is
        # registered AND the target playbook does NOT navigate to the auth
        # entrypoint as its first step, prepend the login playbook's steps so
        # the agent always starts authenticated.
        login_prepended_count = 0
        if (
            os.environ.get("AUTO_LOGIN_DISABLE") != "1"
            and user_data
            and any(k in user_data for k in ("username", "password", "user", "email"))
        ):
            first = steps[0] if steps else {}
            first_action = (first.get("action") or "").lower()
            first_url = (first.get("url") or "").lower()
            starts_at_root = (
                first_action == "navigate"
                and ("rs.ge" in first_url or "eservices.rs.ge" in first_url)
            )
            if not starts_at_root:
                login_pb = await _fetch_login_playbook(_get_http())
                if login_pb and login_pb.get("steps"):
                    login_steps = login_pb["steps"]
                    if login_pb.get("id") != playbook_id:
                        steps = list(login_steps) + list(steps)
                        login_prepended_count = len(login_steps)
                        _emit(
                            "info",
                            f"🔐 Auto-prepended {login_prepended_count} login steps "
                            f"(playbook '{login_pb.get('name', 'login')}' starts the flow)",
                        )
                        # Recompute dangerous-step count over the combined list
                        dangerous_count = sum(1 for s in steps if s.get("dangerous"))
                        if dangerous_count:
                            _emit(
                                "warn",
                                f"Combined flow has {dangerous_count} dangerous step(s). Safety mode: {safety_mode}",
                            )

        postcondition_field_map = _extract_field_map(steps)
        expected_portal = _infer_expected_portal(task, steps)
        data_contract = _build_authoritative_data_contract(user_data, postcondition_field_map, task)
        _emit_contract_summary("playbook prompt", data_contract)
        steps_block = format_playbook_for_prompt(steps, user_data, safety_mode)

        # ── Knowledge-base RAG (chat brain → agent brain) ─────────────────────
        # Pull relevant accounting / rs.ge documentation chunks once per run.
        # The same vector store powers the chat — the agent now sees what the
        # chat sees. Disabled with KNOWLEDGE_DISABLE=1.
        knowledge_block = ""
        if os.environ.get("KNOWLEDGE_DISABLE") != "1":
            target_descriptions = " ".join(
                (s.get("target_description") or "") for s in steps if s.get("target_description")
            )
            rag_query = (task + " " + target_descriptions).strip() or "rs.ge tax declaration"
            _emit("info", "Fetching accounting knowledge for playbook…")
            chunks = await fetch_context(rag_query, limit=8)
            if chunks:
                knowledge_block = format_context_for_prompt(chunks)
                _emit("info", f"Found {len(chunks)} relevant knowledge-base chunk(s)")
            else:
                _emit("info", "No matching knowledge-base entries — proceeding without")

        # Pre-pass Planner (gemini-3.1-pro): produces a tactical plan that
        # gets prepended to the worker's task prompt. (planner_disabled is set
        # at the top of run_agent so the on_step_start closure can read it.)
        plan_text = ""
        if not planner_disabled:
            _emit("info", f"Running Planner ({os.environ.get('PLANNER_MODEL', 'gemini-3.1-pro-preview')})…")
            plan_text = await plan_with_planner(steps, user_data, task, knowledge_block, data_contract)
            if plan_text:
                _emit("info", f"Plan ready ({len(plan_text)} chars)")
        plan_block = (
            f"--- TACTICAL PLAN (read this first, follow throughout) ---\n{plan_text}\n--- END PLAN ---"
            if plan_text else ""
        )

        safety_instructions = {
            "halt-on-dangerous": (
                "SAFETY MODE: HALT-ON-DANGEROUS. "
                "When you reach the marked '⚠️ STOP HERE' line, IMMEDIATELY stop. "
                "Do NOT click submit/send/confirm/pay buttons. "
                "Take a final screenshot and end with a summary like: "
                "'Form is filled and ready for human review. The user must manually click the final submit button.'"
            ),
            "dry-run": (
                "SAFETY MODE: DRY-RUN. "
                "Skip every step marked '⚠️ SKIP'. Do NOT click submit/send/confirm/pay/delete buttons. "
                "Fill all non-dangerous fields, then end with a summary of what would have been submitted."
            ),
            "auto": (
                "SAFETY MODE: AUTO. "
                "All steps will be executed including dangerous ones. "
                "Be extra careful with steps marked '⚠️ DANGEROUS' — read the values twice before clicking."
            ),
        }.get(safety_mode, "")

        full_task = "\n".join(filter(None, [
            "Complete the following task on the web portal step by step.",
            task if task else "",
            "",
            plan_block,
            "",
            steps_block,
            _format_authoritative_data_block(data_contract),
            "",
            safety_instructions,
            "",
            _authority_preamble("playbook"),
            "",
            _worker_rules_block("playbook"),
        ]))
    else:
        # Original RAG mode
        _emit("info", "Fetching relevant instructions from knowledge base…")
        chunks = await fetch_context(task)
        if chunks:
            _emit("info", f"Found {len(chunks)} relevant knowledge-base chunks")
        else:
            _emit("warn", "No matching knowledge-base entries found — proceeding with task only")

        context_block = format_context_for_prompt(chunks)

        # Free-form mode has no playbook, but we can still inject the field-map
        # extracted from EVERY recorded task playbook. This gives the agent
        # explicit "Excel column → form label" knowledge even without a script.
        # We also extract the navigation prefix (steps before first TYPE) and
        # render it as a recommended menu-click recipe — closes the biggest
        # weakness of free-form mode where the Worker gets lost in menus.
        _ff_field_map: dict[str, str] = {}
        nav_prefix_block = ""
        try:
            _all_steps = await fetch_all_task_playbook_steps()
            if _all_steps:
                _ff_field_map = _merge_field_maps([_extract_field_map(s) for s in _all_steps])
                if _ff_field_map:
                    _emit("info", f"Loaded field map ({len(_ff_field_map)} entries) from existing playbooks")
                _all_prefixes = [_extract_nav_prefix(s) for s in _all_steps]
                nav_prefix_block = _format_nav_prefix(_all_prefixes)
                if nav_prefix_block:
                    _canonical_len = len(max((p for p in _all_prefixes if p), key=len, default=[]))
                    _emit("info", f"Loaded nav prefix ({_canonical_len} steps) from existing playbooks")
        except Exception as _e:
            _emit("warn", f"Field-map / nav-prefix fetch failed (non-fatal): {_e}")
        field_map_block = _format_field_map(_ff_field_map, user_data)
        postcondition_field_map = _ff_field_map
        expected_portal = _infer_expected_portal(task)
        data_contract = _build_authoritative_data_contract(user_data, _ff_field_map, task)
        _emit_contract_summary("free-form prompt", data_contract)

        # Phase Q3: pull aggregated site memory + failure patterns for the
        # inferred portal. This is the most impactful injection for free-form
        # mode — it gives the Worker page-level context (transitions, common
        # clicks, dialog patterns) and warns it about known-bad spots.
        site_memory_block = ""
        failure_patterns_block = ""
        if expected_portal:
            try:
                _site_mem = await fetch_site_memory(expected_portal)
                if _site_mem:
                    site_memory_block = _format_site_memory_for_worker(_site_mem, expected_portal)
                    _emit(
                        "info",
                        f"Q1: site memory loaded for {expected_portal} "
                        f"({len(_site_mem.get('pages') or [])} page(s), "
                        f"{len(_site_mem.get('transitions') or [])} transitions)",
                    )
            except Exception as _e:
                _emit("warn", f"Site memory fetch failed (non-fatal): {_e}")
            try:
                _patterns = await fetch_failure_patterns(expected_portal)
                if _patterns:
                    failure_patterns_block = _format_failure_patterns_for_worker(_patterns)
                    _emit("info", f"Q2: failure patterns loaded ({len(_patterns)})")
            except Exception as _e:
                _emit("warn", f"Failure patterns fetch failed (non-fatal): {_e}")

        # Synthesize a plan_text for free-form runs so the P2 plan-preview
        # banner has something to show. Free-form runs don't use the heavy
        # gemini-3.1-pro Planner (no recorded steps to ground a tactical plan
        # against), so we just digest what the Worker will see.
        _ff_plan_lines = [f"TASK: {task}"]
        if data_contract.get("items"):
            _ff_plan_lines.append(_format_authoritative_data_summary(data_contract, max_items=6))
        if _ff_field_map:
            _ff_plan_lines.append(f"FIELD MAP: {len(_ff_field_map)} entries from past playbooks")
            for _k, _v in list(_ff_field_map.items())[:6]:
                _ff_plan_lines.append(f'  • {_k} → "{_v}"')
            if len(_ff_field_map) > 6:
                _ff_plan_lines.append(f"  … and {len(_ff_field_map) - 6} more")
        else:
            _ff_plan_lines.append("FIELD MAP: (none — Worker will rely on discover_form_fields)")
        _ff_plan_lines.append(f"KNOWLEDGE CONTEXT: {len(chunks)} chunk(s) attached")
        _ff_plan_lines.append("MODE: free-form (no playbook); discover_form_fields will run on each new form page.")
        plan_text = "\n".join(_ff_plan_lines)
        _emit("info", f"Free-form plan digest ready ({len(plan_text)} chars)")

        freeform_policy = _freeform_policy_block(safety_mode, allowed_domains)

        # R2: free-form Worker no longer sees the raw RAG chunks (`context_block`).
        # In free-form mode there's no Planner pass to distil them; if the
        # caller actually needs accounting-rule reasoning mid-task they should
        # promote to playbook mode where the Planner can do that work once.
        full_task = "\n".join(filter(None, [
            f"TASK: {task}",
            freeform_policy,
            site_memory_block,
            failure_patterns_block,
            nav_prefix_block,
            field_map_block,
            _format_authoritative_data_block(data_contract),
            "",
            _authority_preamble("free"),
            "",
            _worker_rules_block("free"),
        ]))

    _emit("info", f"Task prompt built ({len(full_task)} chars)")
    _emit("task", full_task)  # send full prompt so the UI can display it

    # 3. Browser setup
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    recordings_dir = Path(__file__).parent / "recordings" / session_id
    # Same per-company override pattern as run_bulk — see comment there.
    _global_headless = os.environ.get("AGENT_HEADLESS", "false").lower() == "true"
    _company_id = os.environ.get("AGENT_COMPANY_ID", "").strip()
    _override = {
        c.strip().lower()
        for c in os.environ.get("AGENT_HEADLESS_COMPANIES", "").split(",")
        if c.strip()
    }
    headless = False if (_company_id and _company_id.lower() in _override) else _global_headless
    state_path = _session_state_path(session_key)
    profile_dir = _user_data_dir(session_key)

    profile_kwargs: dict = {
        "headless": headless,
        "demo_mode": not headless,  # show live side-panel when browser is visible
        "user_data_dir": str(profile_dir),
        "allowed_domains": allowed_domains,
        "cookie_whitelist_domains": [d for d in allowed_domains if "*" not in d],
    }
    if state_path.exists():
        profile_kwargs["storage_state"] = str(state_path)
        _emit("info", f"Restored session storage_state from {state_path}")
    if record:
        recordings_dir.mkdir(parents=True, exist_ok=True)
        profile_kwargs["record_video_dir"] = recordings_dir

    browser_profile = BrowserProfile(**profile_kwargs)

    # 4. LLM
    llm = build_llm()

    # 5. Agent
    agent_tools = _build_tools(
        task_hint=task[:300],
        safety_mode=safety_mode,
        allowed_domains=allowed_domains,
        mode=mode,
    )
    agent = Agent(
        task=full_task,
        llm=llm,
        browser_profile=browser_profile,
        use_vision=True,
        tools=agent_tools,
        register_new_step_callback=on_step,
        register_done_callback=on_done,
        max_actions_per_step=1,
    )

    _emit(
        "info",
        f"Browser agent starting... (headless={headless}, max_steps={max_steps}, "
        f"domains={_domains_for_prompt(allowed_domains)}, session={session_key})",
    )

    # ── Stdin control listener (for Pause/Resume from backend) ────────────────
    # When the backend writes {"action":"resume"}\n to our stdin, we set a
    # threading.Event that the on_step_start hook polls during pause.
    resume_event = asyncio.Event()
    cancel_event = asyncio.Event()  # set by stdin "cancel" or "skip_preview"
    skip_preview_event = asyncio.Event()
    pause_state = {"paused": False}  # mutable so the stdin task can signal

    async def _stdin_control_loop():
        """Read newline-delimited JSON commands from stdin (resume / cancel / skip_preview)."""
        loop = asyncio.get_event_loop()
        try:
            while True:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    break  # stdin closed
                line = line.strip()
                if not line:
                    continue
                try:
                    cmd = json.loads(line)
                except Exception:
                    continue
                action = cmd.get("action")
                if action == "resume":
                    resume_event.set()
                    _emit("info", "▶️  Resume signal received from UI")
                elif action == "cancel":
                    cancel_event.set()
                    skip_preview_event.set()  # also unblocks plan-preview wait
                    try:
                        agent.stop()
                    except Exception:
                        pass
                    _emit("warn", "🛑 Cancel signal received from UI")
                elif action == "skip_preview":
                    skip_preview_event.set()
                    _emit("info", "⏩ Skip-preview signal received from UI")
        except Exception:
            pass  # don't crash the agent on stdin issues

    stdin_task = asyncio.create_task(_stdin_control_loop())

    # ── P2: Plan-preview window ───────────────────────────────────────────────
    # Right before launching the Worker, give the user N seconds to review the
    # Planner's tactical plan and cancel if the prompt was wrong. The plan
    # itself was emitted as an "info" event already; we now emit a dedicated
    # `plan_preview` event the UI watches for, then wait for either timeout,
    # an explicit skip, or a cancel.
    preview_window_sec = int(os.environ.get("AGENT_PLAN_PREVIEW_SEC", "30"))
    if plan_text and preview_window_sec > 0:
        _emit(
            "plan_preview",
            f"Plan ready — Worker starts in {preview_window_sec}s "
            f"(cancel anytime in the run view)",
            plan=plan_text,
            preview_window_sec=preview_window_sec,
        )
        try:
            await asyncio.wait_for(skip_preview_event.wait(), timeout=preview_window_sec)
            if cancel_event.is_set():
                _emit("warn", "🛑 Cancelled during plan preview — Worker not launched")
                try:
                    stdin_task.cancel()
                except Exception:
                    pass
                return False  # exit run_agent cleanly
            _emit("info", "⏩ Plan preview skipped — starting Worker now")
        except asyncio.TimeoutError:
            _emit("info", "Plan preview window elapsed — starting Worker")

    # ── Mid-run Re-planner ────────────────────────────────────────────────────
    # Every REPLAN_EVERY steps (or after consecutive failures), the Planner is
    # invoked with recent action history and injects an updated tactical hint
    # via the worker's message manager. Keeps the worker oriented mid-run.
    replan_every = int(os.environ.get("PLANNER_REPLAN_EVERY", "5"))
    last_replan_step = {"n": 0}  # mutable closure cell
    run_id = os.environ.get("RUN_ID", "")

    # ── Anti-loop detection ───────────────────────────────────────────────────
    # Tracks the last LOOP_WINDOW action signatures. If the SAME signature
    # repeats >= LOOP_THRESHOLD times within the window, we inject a strong
    # corrective context message instead of letting the worker burn more steps.
    loop_window = int(os.environ.get("LOOP_WINDOW", "6"))
    loop_threshold = int(os.environ.get("LOOP_THRESHOLD", "3"))
    recent_action_sigs: list[str] = []
    last_loop_break_step = {"n": -100}

    # ── Smart model escalation ────────────────────────────────────────────────
    # Worker normally runs gemini-3.1-flash-lite (cheap, fast). After 2
    # consecutive failures, escalate to a stronger model for the next few
    # steps, then drop back to the cheap model when it stabilises.
    escalation_model = os.environ.get("AGENT_ESCALATION_MODEL", "gemini-3.1-pro-preview")
    escalation_steps = int(os.environ.get("AGENT_ESCALATION_STEPS", "5"))
    escalation_state = {
        "active": False,
        "remaining": 0,
        "original_llm": None,
        "stronger_llm": None,  # built lazily
    }

    def _action_signature(h) -> str:
        """Compact signature of the LAST action a step took. Same target +
        same action type → same signature → loop detection fires."""
        try:
            mo = getattr(h, "model_output", None)
            if not mo or not getattr(mo, "action", None):
                return ""
            a = mo.action[-1]
            data = a.model_dump(exclude_none=True, mode="json")
            if not data:
                return ""
            kind, args = next(iter(data.items()))
            # Pull the most identifying field of the action
            target = ""
            if isinstance(args, dict):
                for key in ("index", "url", "selector", "text"):
                    if key in args:
                        target = f"{key}={args[key]}"
                        break
            return f"{kind}:{target}"
        except Exception:
            return ""

    # T1.1: capture the form's DOM state every step so we always have a
    # recent snapshot to fall back on if the live page is unreadable when
    # postcondition runs (e.g. Worker called done() and the SPA navigated
    # away). Stored on the closure dict `last_form_snapshot`.
    last_form_snapshot: dict[str, dict] = {"value": {}}

    # R1: deterministic post-TYPE verification.
    # After every step, we compare the DOM <input>.value of fields the Worker
    # just typed into vs the text it claimed to type. Mismatches get emitted
    # as `field-reset` warnings — those flow back to the Worker as events
    # the prompt has been told to react to with a single Ctrl+A retype.
    # `field_reset_seen` tracks (step_idx, field_index) so we don't shout
    # about the same unresolved mismatch on every subsequent step.
    field_reset_seen: set[tuple[int, int]] = set()

    def _last_typed_actions(ag) -> list[dict]:
        """Return the (text, index) pairs from the most recently completed step."""
        try:
            history_items = getattr(getattr(ag, "state", None), "history", None)
            if not history_items:
                return []
            last = history_items.history[-1] if hasattr(history_items, "history") else history_items[-1]
            mo = getattr(last, "model_output", None)
            if not mo or not getattr(mo, "action", None):
                return []
            out: list[dict] = []
            for a in mo.action:
                try:
                    d = a.model_dump(exclude_none=True, mode="json")
                except Exception:
                    continue
                if not isinstance(d, dict):
                    continue
                for key, params in d.items():
                    if key in ("input", "input_text", "type") and isinstance(params, dict):
                        text = params.get("text")
                        index = params.get("index")
                        if text is not None and index is not None:
                            out.append({"text": str(text), "index": int(index)})
            return out
        except Exception:
            return []

    async def _dom_value_for_index(session, page, idx: int) -> tuple[str | None, str]:
        """Read the live <input>.value for a Worker action index.

        Uses browser-use's selector_map (Worker action index → real DOM node)
        and evaluates the node's xpath in the page. Returns (value, label).
        value=None means the node could not be resolved or read.
        """
        if session is None or page is None:
            return None, ""
        try:
            node = await session.get_dom_element_by_index(idx)
        except Exception:
            return None, ""
        if node is None:
            return None, ""
        xpath = ""
        try:
            xpath = node.xpath or ""
        except Exception:
            xpath = ""
        if not xpath:
            return None, ""
        attrs = getattr(node, "attributes", {}) or {}
        ax = getattr(node, "ax_node", None)
        ax_name = getattr(ax, "name", "") if ax is not None else ""
        label = (
            (ax_name or "")
            or attrs.get("aria-label", "")
            or attrs.get("name", "")
            or attrs.get("placeholder", "")
            or f"#{idx}"
        )
        try:
            value = await page.evaluate(
                """(xp) => {
                    const r = document.evaluate(xp, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                    const el = r && r.singleNodeValue;
                    if (!el) return null;
                    if (el.tagName === 'SELECT') {
                        const opt = el.options[el.selectedIndex];
                        return opt ? (opt.textContent || el.value || '') : (el.value || '');
                    }
                    return el.value != null ? String(el.value) : '';
                }""",
                xpath,
            )
        except Exception:
            return None, str(label)[:120]
        if value is None:
            return None, str(label)[:120]
        return str(value), str(label)[:120]

    async def _check_field_reset_after_step(ag) -> None:
        """Compare each just-typed value with what the DOM actually holds.
        Emits a one-shot 'field-reset' warning per (step, index) on mismatch."""
        typed = _last_typed_actions(ag)
        if not typed:
            return
        session = getattr(ag, "browser_session", None)
        page = await _get_current_page_from_agent(ag)
        if session is None or page is None:
            return
        try:
            current_step = int(getattr(getattr(ag, "state", None), "n_steps", 0)) - 1
        except Exception:
            current_step = -1
        for entry in typed:
            idx = entry["index"]
            wanted = entry["text"].strip()
            if not wanted:
                continue
            seen_key = (current_step, idx)
            if seen_key in field_reset_seen:
                continue
            actual, label = await _dom_value_for_index(session, page, idx)
            if actual is None:
                # Node went away (re-render) — Worker will discover via next scan.
                continue
            if _values_equivalent(actual.strip(), wanted):
                continue
            field_reset_seen.add(seen_key)
            _emit(
                "warn",
                f"R1 field-reset: typed {wanted!r} into '{label}' (#{idx}) "
                f"but DOM now shows {actual!r}. Retype once with Ctrl+A.",
                field_index=idx,
                field_label=label,
                expected=wanted[:80],
                actual=actual[:80],
            )

    async def _maybe_capture_form_snapshot(ag) -> None:
        try:
            page = await _get_current_page_from_agent(ag)
            if page is None:
                return
            session = getattr(ag, "browser_session", None)
            # Prefer session-based extraction: it pierces shadow DOM and iframes
            # (rs.ge ExtJS form widgets live there). Fall back to top-level
            # querySelectorAll only when no selector_map is cached yet.
            snap = await _extract_dom_fields_via_session(session, page)
            if not snap.get("fields"):
                snap = await _extract_dom_fields(page)
            if snap.get("fields"):
                # URL may be empty if eval failed; preserve the older one rather
                # than blanking it out, so postcondition has something to check.
                if not snap.get("url") and last_form_snapshot.get("value"):
                    snap["url"] = last_form_snapshot["value"].get("url", "")
                    if not snap.get("title"):
                        snap["title"] = last_form_snapshot["value"].get("title", "")
                last_form_snapshot["value"] = snap
            # R1: compare TYPE actions against live DOM using browser-use's
            # own selector_map. Runs independently of the snapshot above
            # because the snapshot uses array-position indexing, while R1
            # needs the stable backend-node index Worker actually targets.
            await _check_field_reset_after_step(ag)
        except Exception:
            pass  # snapshot capture is best-effort; never break the run

    budget_warning_emitted = {"value": False}
    url_stuck_state = {
        "url": "",
        "unchanged_steps": 0,
        "last_typed_step": 0,
        "last_warning_step": -100,
    }

    async def _current_agent_url(ag) -> str:
        try:
            page = await _get_current_page_from_agent(ag)
            if page is None:
                return ""
            try:
                return str(await _safe_page_eval(page, "() => window.location.href") or "")
            except Exception:
                return str(getattr(page, "url", "") or "")
        except Exception:
            return ""

    async def _emit_budget_and_stuck_warnings(ag, n: int) -> None:
        if max_steps > 0 and not budget_warning_emitted["value"] and n >= max(1, int(max_steps * 0.7)):
            budget_warning_emitted["value"] = True
            _emit(
                "warn",
                f"Step budget warning: {n}/{max_steps} steps used (>=70%). Prioritize the next decisive action or stop with the blocker.",
                step=n,
                max_steps=max_steps,
            )

        url = await _current_agent_url(ag)
        if url and url == url_stuck_state["url"]:
            url_stuck_state["unchanged_steps"] += 1
        else:
            url_stuck_state["url"] = url
            url_stuck_state["unchanged_steps"] = 1 if url else 0

        if _last_typed_actions(ag):
            url_stuck_state["last_typed_step"] = n

        no_typing_for = n - int(url_stuck_state["last_typed_step"])
        if (
            url_stuck_state["unchanged_steps"] >= 5
            and no_typing_for >= 5
            and n - int(url_stuck_state["last_warning_step"]) >= 5
        ):
            url_stuck_state["last_warning_step"] = n
            _emit(
                "warn",
                "Stuck detection: URL has not changed for 5+ steps and no typing has occurred. "
                "Change strategy, use discovery once, or stop and report the blocker.",
                step=n,
                unchanged_steps=url_stuck_state["unchanged_steps"],
                url=url[:200],
            )

    async def on_step_start(ag) -> None:
        # T1.1 fires unconditionally (free-form runs need it too).
        await _maybe_capture_form_snapshot(ag)
        try:
            n = int(getattr(ag.state, "n_steps", 1))
        except Exception:
            n = 1
        await _emit_budget_and_stuck_warnings(ag, n)

        if planner_disabled or not playbook_id:
            return
        try:
            cf = getattr(ag.state, "consecutive_failures", 0)

            # ── Smart model escalation logic ──────────────────────────────
            # If we're not yet escalated AND have 2+ consecutive failures, swap
            # the worker LLM to the stronger model for the next few steps.
            # If we ARE escalated and the failure burst has cleared (cf == 0)
            # AND we've used N escalated steps, swap back to the cheap one.
            try:
                if not escalation_state["active"] and cf >= 2:
                    if escalation_state["stronger_llm"] is None:
                        from browser_use.llm.google.chat import ChatGoogle
                        escalation_state["stronger_llm"] = ChatGoogle(
                            model=escalation_model,
                            api_key=os.environ["GEMINI_API_KEY"],
                            temperature=0,
                        )
                    escalation_state["original_llm"] = ag.llm
                    ag.llm = escalation_state["stronger_llm"]
                    escalation_state["active"] = True
                    escalation_state["remaining"] = escalation_steps
                    _emit(
                        "info",
                        f"⚡ Escalating worker → {escalation_model} for next {escalation_steps} steps "
                        f"(after {cf} consecutive failures)",
                    )
                elif escalation_state["active"]:
                    escalation_state["remaining"] -= 1
                    # De-escalate when we've burned the budget AND failures cleared
                    if escalation_state["remaining"] <= 0 and cf == 0:
                        if escalation_state["original_llm"] is not None:
                            ag.llm = escalation_state["original_llm"]
                        escalation_state["active"] = False
                        escalation_state["remaining"] = 0
                        _emit("info", "✓ De-escalating worker back to cheap model (recovered)")
            except Exception as exc:
                _emit("warn", f"Model escalation hiccup (non-fatal): {exc}")

            # Update sliding window of recent action signatures.
            try:
                last_h = list(ag.history.history)[-1] if ag.history.history else None
            except Exception:
                last_h = None
            sig = _action_signature(last_h) if last_h else ""
            if sig:
                recent_action_sigs.append(sig)
                if len(recent_action_sigs) > loop_window:
                    recent_action_sigs.pop(0)

            # Detect loop: same sig repeated threshold+ times in window.
            sig_count: dict[str, int] = {}
            for s in recent_action_sigs:
                sig_count[s] = sig_count.get(s, 0) + 1
            looped_sig = next((s for s, c in sig_count.items() if c >= loop_threshold and s), None)

            if looped_sig and (n - last_loop_break_step["n"]) >= loop_threshold:
                last_loop_break_step["n"] = n
                _emit(
                    "warn",
                    f"⚠ Loop detected: action {looped_sig!r} repeated {sig_count[looped_sig]}× in last {len(recent_action_sigs)} steps — injecting corrective hint",
                )
                from browser_use.llm.messages import UserMessage
                try:
                    ag._message_manager._add_context_message(
                        UserMessage(content=(
                            f"🚫 LOOP-BREAK at step {n}: you have repeated the action "
                            f"`{looped_sig}` {sig_count[looped_sig]} times with no progress. "
                            "STOP doing this exact action. Take a fresh screenshot, scroll, "
                            "look for a DIFFERENT element (sibling button, alternative tab, "
                            "next/previous control), and try a structurally different approach. "
                            "If the page truly has no other path, mark the current step as "
                            "skipped and proceed to the next playbook step."
                        ))
                    )
                except Exception:
                    pass
                # Reset window so we don't fire again for this same loop
                recent_action_sigs.clear()
                # Still let the planner run below — the new context plus loop
                # break should give it full info to diverge.

            should_replan = (
                (n - last_replan_step["n"] >= replan_every and n > 1)
                or cf >= 2
            )
            if not should_replan:
                return
            last_replan_step["n"] = n

            # Build a short recent-history summary for the planner.
            try:
                hist_items = list(ag.history.history)[-6:]
            except Exception:
                hist_items = []
            recent_lines = []
            for i, h in enumerate(hist_items):
                actions = []
                try:
                    if h.model_output and h.model_output.action:
                        for a in h.model_output.action:
                            d = a.model_dump(exclude_none=True, mode="json")
                            # Compact: take action type + any text/index
                            for k, v in d.items():
                                snippet = str(v)
                                if len(snippet) > 80:
                                    snippet = snippet[:80] + "…"
                                actions.append(f"{k}:{snippet}")
                except Exception:
                    pass
                err = ""
                try:
                    for r in (h.result or []):
                        if r.error:
                            err = f" ERROR={r.error[:80]}"
                            break
                except Exception:
                    pass
                recent_lines.append(f"  - {' | '.join(actions) or '(no action)'}{err}")
            recent_block = "\n".join(recent_lines) if recent_lines else "(no history yet)"

            from browser_use.llm.messages import UserMessage
            planner = build_planner_llm()
            replan_prompt = (
                f"You are the planner for an in-progress browser automation. "
                f"Current step: {n}. Consecutive failures: {cf}.\n"
                f"Original tactical plan was given at task start. Here are the LAST {len(recent_lines)} actions:\n"
                f"{recent_block}\n\n"
                "Based on this, give the worker ONE short paragraph (<6 lines) of tactical guidance for the next step. "
                "Mention: (a) where in the playbook the worker likely is, (b) one concrete next action, "
                "(c) if a loop or dead-end is detected, what alternative to try. Be terse.\n\n"
                "SPECIAL CASE — PAUSE FOR HUMAN: If the worker has hit something the agent CANNOT solve "
                "automatically (CAPTCHA, 2FA / OTP code request, SMS verification, unexpected re-authentication "
                "screen, account-locked notice), start your response with the literal token PAUSE: followed by "
                "one short sentence describing what the human needs to do in the visible Chromium window. "
                "Otherwise do NOT use that token."
            )
            response = await planner.ainvoke([UserMessage(content=replan_prompt)])
            hint = (getattr(response, "completion", None) or "").strip()
            if hint:
                _emit("info", f"[Planner @ step {n}] {hint[:200]}{'…' if len(hint) > 200 else ''}")

                # Pause detection: only fire when the planner EXPLICITLY starts
                # its response with "PAUSE:" — the planner prompt instructs it
                # to do this only when the worker truly cannot proceed
                # (CAPTCHA, 2FA, account-locked, etc.). Substring-anywhere
                # matching produced false positives because the planner
                # frequently mentions those keywords descriptively while
                # explaining the situation, not while asking for a pause.
                should_pause = hint.lstrip().upper().startswith("PAUSE:")
                if should_pause and not pause_state["paused"]:
                    pause_state["paused"] = True
                    reason = hint.split("\n", 1)[0][:300]
                    _emit(
                        "paused",
                        f"Agent paused — manual help needed. {reason}",
                        runId=run_id,
                        reason=reason,
                    )
                    try:
                        ag.pause()
                    except Exception:
                        pass
                    # Wait until the backend signals resume (or 10 minutes max).
                    try:
                        await asyncio.wait_for(resume_event.wait(), timeout=600)
                    except asyncio.TimeoutError:
                        _emit("warn", "Resume timeout (10min) — auto-resuming")
                    resume_event.clear()
                    pause_state["paused"] = False
                    try:
                        ag.resume()
                    except Exception:
                        pass
                    _emit("info", "▶️  Agent resumed")
                else:
                    ag._message_manager._add_context_message(
                        UserMessage(content=f"PLANNER UPDATE (step {n}):\n{hint}")
                    )
        except Exception as exc:
            # Re-planner failures must never break the run.
            _emit("warn", f"Planner mid-run failed: {exc}")

    try:
        _reset_emitted_steps()
        history = await agent.run(max_steps=max_steps, on_step_start=on_step_start)

        # Use the agent's own success signal — don't trust final_result() alone.
        # has_errors() catches consecutive LLM failures, max-step exhaustion, etc.
        succeeded = history.is_successful() is True and not history.has_errors()
        final = history.final_result()
        # Stash for the autonomous-task callback in main().
        global _LAST_FINAL_RESULT
        _LAST_FINAL_RESULT = final if isinstance(final, str) else None
        completion_state = _default_completion_state(safety_mode) if succeeded else COMPLETION_FAILED
        history_errors = history.errors() or []
        session_broken = any(
            "Failed to open a new tab" in str(err) or "Root CDP client not initialized" in str(err)
            for err in history_errors
        )
        if session_broken:
            _emit(
                "warn",
                "Browser session marked broken; live page recovery is unreliable and cleanup/export will be best-effort.",
                session_broken=True,
            )

        # Q5b — Halt-on-dangerous self-success guard.
        # In halt-on-dangerous / dry-run modes the Worker often calls
        # done(success=False) thinking "submit didn't happen, so it's incomplete"
        # — but stopping cleanly IS the success criterion in these modes.
        #
        # has_errors() is intentionally NOT a blocker here: a single
        # "element not found, retried" tick during navigation flips it True
        # for the rest of the run even if everything recovered. We trust K1
        # (every expected non-credential value was actually typed) as the
        # real success signal in halt mode.
        q5b_promoted = False  # track so the negative-markers check below can defer
        if (
            not succeeded
            and safety_mode in ("halt-on-dangerous", "dry-run")
        ):
            _missing_for_guard = _check_authoritative_contract_coverage(history, data_contract)
            if not _missing_for_guard:
                _emit(
                    "info",
                    "Halt-mode self-success guard: Worker reported done(success=False) "
                    "but every expected value was typed; treating clean stop as success.",
                )
                succeeded = True
                completion_state = _default_completion_state(safety_mode)
                q5b_promoted = True

        # K4: URL sanity check — advisory warning if URL still points to a form/edit page.
        # Does not flip succeeded alone (rs.ge keeps the same base URL through form steps).
        _k4_urls = history.urls() or []
        if succeeded and _k4_urls:
            _k4_url = (_k4_urls[-1] or "").lower()
            _K4_FORM_HINTS = ("/new", "/create", "/fill", "/edit", "mode=edit", "action=new")
            if any(p in _k4_url for p in _K4_FORM_HINTS):
                _emit("warn", f"⚠ URL guard: final URL suggests still on form page: {_k4_urls[-1]}")
                # Q3: tell the learning loop where the URL guard fired.
                if expected_portal:
                    asyncio.create_task(report_failure_pattern(
                        domain=expected_portal,
                        failure_type="k4_url",
                        url_pattern=_k4_urls[-1][:200],
                        symptom=f"Run reported success but final URL still looks like a form/edit page: {_k4_urls[-1]}",
                        workaround="Check the final action actually submitted; look for a confirmation page URL pattern.",
                    ))

        # Override succeeded=True when the LLM contradicts itself in the done
        # message: agent calls done(success=True) but the TEXT explicitly admits
        # failure ("unable", "could not", "stuck", etc.). Without this guard,
        # 58-action garbage paths get cached as if they were successful.
        #
        # Skipped when Q5b promoted a halt-mode run: the Worker's "I was
        # unable to submit" pessimism is wrong by definition in that mode
        # (we explicitly told it not to submit), and K1 already confirmed
        # every expected value made it into the form.
        if succeeded and final and not q5b_promoted:
            negative_markers = (
                "unable to ",
                "could not ",
                "couldn't ",
                "i am unable",
                "i was unable",
                "cannot fulfil",
                "cannot complete",
                "cannot proceed",
                "cannot access",
                "failed to ",
                "did not succeed",
                "stuck",
                "repeatedly attempt",
                "unresponsive",
                "unable to fulfil",
                "task was not completed",
            )
            lowered = final.lower()
            hit = next((m for m in negative_markers if m in lowered), None)
            # Benign exceptions — Worker is explaining a guarded stop, not
            # a real failure. The hard verify-limit ("unable to verify... call
            # limit") and halt-on-dangerous ("ready for human review") are
            # the two phrases the prompts deliberately tell Worker to use.
            benign_phrases = (
                "verify_typed_values",
                "call limit",
                "verify call limit",
                "hard limit",
                "ready for human review",
                "ready for review",
                "stopped before submit",
                "stopped before the irreversible",
            )
            benign = any(p in lowered for p in benign_phrases)
            if hit and not benign:
                _emit(
                    "warn",
                    f"Agent self-declared success but message admits failure ('{hit}') — overriding to failed",
                )
                succeeded = False
                completion_state = COMPLETION_FAILED

        # K1 — Post-run typed-value verification (anti-hallucination).
        # The agent can call done(success=True) with a confident message but never
        # have actually typed the values from user_data. Walk the action history,
        # collect every typed string, and require each non-credential expected
        # value to appear at least once. If anything is missing, flip succeeded
        # to false so the cache write below is skipped and no broken path gets
        # replayed on subsequent runs.
        if succeeded:
            missing = _check_authoritative_contract_coverage(history, data_contract)
            if missing:
                preview = ", ".join(str(v.get("key")) for v in missing[:8])
                more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
                _emit(
                    "warn",
                    f"⚠ Hallucinated success: agent reported 'done' but never typed required spreadsheet field(s): "
                    f"[{preview}{more}]. Cache rejected, succeeded=false.",
                )
                succeeded = False
                completion_state = COMPLETION_FAILED
                if expected_portal:
                    asyncio.create_task(report_failure_pattern(
                        domain=expected_portal,
                        failure_type="k1_hallucination",
                        symptom=f"Agent claimed success but never typed required spreadsheet fields: [{preview}{more}]",
                        workaround="Use the authoritative spreadsheet block and verify every required field before success.",
                    ))

        review_payload: dict = {}
        # S8: deterministic trust gate. Verify every TYPE Worker performed
        # actually landed in the DOM. Runs first, even when user_data is
        # empty (free-form mode), so a Worker that "filled" 8 fields with
        # values that the SPA rejected still gets caught and reported.
        if succeeded and os.environ.get("TYPED_LOG_VALIDATOR_DISABLE") != "1":
            try:
                page_for_validation = await _get_current_page_from_agent(agent)
                _matched, _mismatched = await _validate_typed_log_against_dom(
                    agent_tools, page_for_validation
                )
            except Exception as exc:
                _matched, _mismatched = [], []
                _emit("warn", f"S8 typed-log validator threw: {exc}")
            if _mismatched:
                preview = "; ".join(
                    f"#{m.get('index')} expected {m.get('expected')!r} got {m.get('actual')!r}"
                    for m in _mismatched[:5]
                )
                _emit(
                    "warn",
                    f"S8 typed-log mismatch: {len(_mismatched)} of "
                    f"{len(_matched) + len(_mismatched)} typed values are missing in DOM. "
                    f"Worker reported success but the form is not actually filled. "
                    f"Examples: {preview}",
                    matched=len(_matched),
                    mismatched=len(_mismatched),
                    details=_mismatched[:8],
                )
                succeeded = False
                completion_state = COMPLETION_FAILED

        if succeeded and os.environ.get("DOM_VALIDATOR_DISABLE") != "1":
            page = await _get_current_page_from_agent(agent)
            dom_ok, review_payload, dom_error = await _run_dom_postcondition(
                page,
                user_data,
                postcondition_field_map,
                expected_portal,
                data_contract,
                fallback_snapshot=last_form_snapshot.get("value"),
            )
            _emit("pre_submit_review", "DOM postcondition review completed.", review=review_payload)
            if not dom_ok:
                _emit("warn", f"DOM postcondition failed: {dom_error}")
                succeeded = False
                completion_state = COMPLETION_FAILED
                # Q4 + Q3: report each individual postcondition mismatch so the
                # learning loop can surface it next time. The review_payload
                # carries per-field errors when dom_ok is False.
                if expected_portal:
                    _errors = (review_payload or {}).get("errors") or []
                    if not _errors:
                        asyncio.create_task(report_failure_pattern(
                            domain=expected_portal,
                            failure_type="postcondition",
                            symptom=str(dom_error)[:300] or "DOM postcondition failed",
                            workaround="Re-check field-map mapping; verify the form's actual labels via discover_form_fields.",
                        ))
                    else:
                        for _err in _errors[:5]:
                            if isinstance(_err, dict):
                                _flabel = str(_err.get("field") or _err.get("label") or "")[:200] or None
                                _sym = str(_err.get("reason") or _err.get("error") or _err)[:300]
                            else:
                                _flabel = None
                                _sym = str(_err)[:300]
                            asyncio.create_task(report_failure_pattern(
                                domain=expected_portal,
                                failure_type="postcondition",
                                field_label=_flabel,
                                symptom=_sym,
                                workaround="Verify the field actually accepted the typed value; re-type with Ctrl+A if it shows 0.00.",
                            ))
        if succeeded:
            contract_missing = _check_required_contract_items_verified(
                data_contract,
                getattr(agent_tools, "_declario_typed_log", None),
                review_payload,
            )
            if contract_missing:
                preview = ", ".join(str(v.get("key")) for v in contract_missing[:8])
                more = f" (+{len(contract_missing) - 8} more)" if len(contract_missing) > 8 else ""
                _emit(
                    "warn",
                    f"Spreadsheet contract not fully verified: missing required item(s) [{preview}{more}]",
                    contract_missing=contract_missing[:8],
                )
                succeeded = False
                completion_state = COMPLETION_FAILED
                if expected_portal:
                    for item in contract_missing[:5]:
                        asyncio.create_task(report_failure_pattern(
                            domain=expected_portal,
                            failure_type="postcondition",
                            field_label=str(item.get("key") or "")[:200] or None,
                            symptom=str(item.get("reason") or "required spreadsheet item not verified")[:300],
                            workaround="Stop and report unmapped or unverified spreadsheet fields instead of inferring values.",
                        ))

        # M1: Visual Validator — one Gemini Vision call to confirm the final page
        # looks like a real submission confirmation, not the form still open.
        # Runs only if K1 passed. Skipped silently when no screenshots are available.
        if succeeded:
            _m1_screenshots = history.screenshots() or []
            if _m1_screenshots:
                try:
                    _emit("info", "Running visual validator…")
                    _vresult = await _visual_validator(_m1_screenshots[-1], user_data, final or "", safety_mode)
                    _zero_fields = _vresult.get("suspicious_zero_fields") or []
                    _ok, _new_completion_state, _verr = _visual_outcome_for_safety(safety_mode, _vresult)
                    completion_state = _new_completion_state
                    if _vresult.get("reg_number"):
                        _emit("info", f"✓ Registration number confirmed on page: {_vresult['reg_number']}")
                    if _zero_fields:
                        _emit(
                            "warn",
                            f"⚠ Visual validator: {len(_zero_fields)} field(s) still show 0.00: "
                            f"{_zero_fields[:5]}. Cache rejected, succeeded=false.",
                        )
                        succeeded = False
                        completion_state = COMPLETION_FAILED
                        if expected_portal:
                            for _zlabel in _zero_fields[:5]:
                                asyncio.create_task(report_failure_pattern(
                                    domain=expected_portal,
                                    failure_type="m1_zero_field",
                                    field_label=str(_zlabel)[:200],
                                    symptom=f"Field '{_zlabel}' shows 0.00 after typing — likely got reset.",
                                    workaround="Click the field, press Ctrl+A, then re-type.",
                                ))
                    elif not _ok:
                        _emit(
                            "warn",
                            f"⚠ Visual validator: page does not look like a confirmation. "
                            f"({_vresult.get('explanation', '')[:120]}). Cache rejected, succeeded=false.",
                        )
                        succeeded = False
                        completion_state = COMPLETION_FAILED
                        if expected_portal:
                            asyncio.create_task(report_failure_pattern(
                                domain=expected_portal,
                                failure_type="m1_not_confirmation",
                                symptom=f"Not a confirmation page after run. {_vresult.get('explanation','')[:300]}",
                                workaround="Verify the submit/finalise step actually fired; check for stuck loading spinner or popup.",
                            ))
                    else:
                        _emit("info", f"Visual validator accepted final state ({completion_state}).")
                except Exception as _ve:
                    _emit("warn", f"Visual validator error (non-fatal, result unchanged): {_ve}")

        if succeeded and final:
            result_msg = final
        elif history.has_errors():
            errors = history_errors
            last_err = errors[-1] if errors else "unknown error"
            result_msg = f"Agent stopped with errors. Last error: {last_err}"
            succeeded = False
            completion_state = COMPLETION_FAILED
        else:
            result_msg = final or "Agent finished but returned no result"
            succeeded = bool(final)  # no result → treat as incomplete

        recording_path = str(recordings_dir) if record else None

        # On success, capture the action history so subsequent runs of this
        # playbook can replay deterministically without invoking the LLM.
        if succeeded and playbook_id:
            try:
                await _capture_action_history(
                    history,
                    playbook_id,
                    user_data,
                    safety_mode=safety_mode,
                    completion_state=completion_state,
                )
            except Exception as exc:
                _emit("warn", f"Action-history capture failed (non-fatal): {exc}")

        _emit(
            "result",
            result_msg,
            success=succeeded,
            recording=recording_path,
            steps_taken=history.number_of_steps(),
            completion_state=completion_state if succeeded else COMPLETION_FAILED,
        )
        return succeeded

    except Exception as exc:
        _emit(
            "result",
            str(exc),
            success=False,
            recording=str(recordings_dir) if record else None,
            completion_state=COMPLETION_FAILED,
        )
        return False
    finally:
        if getattr(agent, "browser_session", None) is not None:
            try:
                await _export_storage_state_if_possible(
                    agent.browser_session,
                    state_path,
                    context="single-finally",
                )
            except Exception as exc:
                _emit("warn", f"storage_state export failed: {exc}")
        # Stop the stdin control listener so the asyncio loop can exit cleanly.
        try:
            stdin_task.cancel()
        except Exception:
            pass
        await _close_http()


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Declario Browser Agent")
    parser.add_argument("--task", required=False, default="", help="Task description")
    parser.add_argument("--playbook", default="", help="Playbook ID to execute")
    parser.add_argument(
        "--bulk-run-id",
        default="",
        help="Run as a bulk worker for the given bulk_run UUID (mutually exclusive with --task/--playbook)",
    )
    parser.add_argument(
        "--data",
        default="{}",
        help="JSON string with user data (auth credentials, form values, etc.)",
    )
    parser.add_argument("--data-file", help="Path to JSON file with user data")
    parser.add_argument(
        "--max-steps", type=int,
        default=int(os.environ.get("AGENT_MAX_STEPS", 50)),
        help="Max agent steps before giving up",
    )
    parser.add_argument("--no-record", action="store_true", help="Skip video recording")
    parser.add_argument(
        "--safety-mode",
        choices=["auto", "halt-on-dangerous", "dry-run"],
        default="halt-on-dangerous",
        help="auto: run all steps; halt-on-dangerous: stop before submit/send (default); dry-run: skip all dangerous steps with warning",
    )
    parser.add_argument(
        "--allowed-domains",
        default="rs.ge",
        help="Comma-separated allowed browser domains. Defaults to rs.ge.",
    )
    parser.add_argument(
        "--session-key",
        default=DEFAULT_SESSION_KEY,
        help="Persistent browser session key.",
    )
    parser.add_argument(
        "--mode",
        choices=["free", "playbook", "bulk"],
        default="free",
        help="Run mode for policy prompts and audit metadata.",
    )

    args = parser.parse_args()

    # Suppress noisy library logs so our JSON lines stay clean
    logging.basicConfig(level=logging.WARNING)
    # Silence library loggers that would otherwise leak through stderr to the
    # backend's log capture and surface internal library names ("browser_use…",
    # "playwright…") to end users. We keep ERROR so genuine crashes still show.
    for noisy in (
        "httpx",
        "httpcore",
        "playwright",
        "asyncio",
        "browser_use",
        "browser_use.tools",
        "browser_use.tools.service",
        "browser_use.agent",
        "browser_use.controller",
        "browser_use.browser",
        "google_genai",
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    if args.bulk_run_id:
        if args.task or args.playbook:
            _emit("error", "--bulk-run-id is mutually exclusive with --task/--playbook")
            sys.exit(1)
        try:
            asyncio.run(run_bulk(args.bulk_run_id))
            sys.exit(0)
        except KeyboardInterrupt:
            _emit("warn", "Bulk worker interrupted")
            sys.exit(130)
        except Exception as exc:
            _emit("error", f"Bulk worker fatal: {exc}")
            sys.exit(1)

    if not args.task and not args.playbook:
        _emit("error", "--task, --playbook, or --bulk-run-id is required")
        sys.exit(1)

    # Load user data
    if args.data_file:
        with open(args.data_file) as f:
            user_data = json.load(f)
    else:
        try:
            user_data = json.loads(args.data)
        except json.JSONDecodeError as e:
            _emit("error", f"Invalid --data JSON: {e}")
            sys.exit(1)

    success = asyncio.run(
        run_agent(
            task=args.task,
            user_data=user_data,
            max_steps=args.max_steps,
            record=not args.no_record,
            playbook_id=args.playbook,
            safety_mode=args.safety_mode,
            allowed_domains=_coerce_allowed_domains(args.allowed_domains),
            session_key=args.session_key,
            mode=args.mode,
        )
    )

    # Autonomous-task callback: when this run was dispatched for a
    # declaration in free mode (AGENT_CORRELATION_ID set + declaration_id
    # in the data), tell agent-backend the outcome so the declaration's
    # status flips without polling. Best-effort — never change the exit
    # code over a callback failure.
    correlation_id = os.environ.get("AGENT_CORRELATION_ID", "").strip()
    # Generic submission routing: agent-backend stuffs source_id/type +
    # result_path into the task data so we can report back to the right
    # endpoint (VAT declaration, payroll run, …). declaration_id stays a
    # legacy fallback.
    source_id = str(
        user_data.get("source_id") or user_data.get("declaration_id") or ""
    ).strip()
    result_path = str(user_data.get("result_path") or "").strip()
    if correlation_id and source_id:
        try:
            asyncio.run(
                _post_task_callback(
                    source_id=source_id,
                    result_path=result_path,
                    success=success,
                    final_text=_LAST_FINAL_RESULT,
                )
            )
        except Exception as exc:  # noqa: BLE001
            _emit("warn", f"task callback failed: {exc}")

    sys.exit(0 if success else 1)


def _parse_receipt(final_text: str | None) -> str | None:
    """Pull a `receipt=<value>` token out of the agent's final answer.
    The autonomous VAT prompt instructs the agent to end with that line."""
    if not final_text:
        return None
    import re

    m = re.search(r"receipt\s*=\s*([^\s,;]+)", final_text, re.IGNORECASE)
    return m.group(1).strip() if m else None


async def _post_task_callback(
    source_id: str,
    result_path: str,
    success: bool,
    final_text: str | None,
) -> None:
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:3001").rstrip("/")
    payload = {
        "source_id": source_id,
        "result_path": result_path or None,
        "status": "submitted" if success else "failed",
        "receipt": _parse_receipt(final_text),
        "error": None if success else (final_text or "autonomous run failed")[:500],
    }
    async with httpx.AsyncClient(timeout=15, headers=_backend_auth_headers()) as client:
        await client.post(f"{backend_url}/agent/task-callback", json=payload)


if __name__ == "__main__":
    main()
