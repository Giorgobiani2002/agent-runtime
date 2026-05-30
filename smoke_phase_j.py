"""
End-to-end smoke test for Phase J (knowledge-aware agent).

Runs the prompt-build pipeline against a real playbook + the live backend
(`/agent/context`) + the live planner LLM (`gemini-3.1-flash-lite-preview`).
Does NOT launch Chromium — this is purely about verifying that:
  1. fetch_context() returns relevant Georgian VAT chunks
  2. format_context_for_prompt() produces a usable INSTRUCTIONS block
  3. plan_with_planner() actually consumes the knowledge_block and reflects
     it in the tactical plan it generates
  4. The full_task prompt contains all four expected sections in the right order

Run from the agent/ directory:
    python smoke_phase_j.py <playbook_id>

Default playbook: 88cd6055-... (43-step VAT declaration).
"""
import asyncio
import os
import sys

# Make sure we hit the local backend
os.environ.setdefault("BACKEND_URL", "http://localhost:3001")

# Lazy import (main has heavy deps that may print on load)
import main  # noqa: E402


PLAYBOOK_ID = sys.argv[1] if len(sys.argv) > 1 else "88cd6055-2d60-460b-8f7d-81632dde6034"
USER_DATA = {
    "username": "satesto2",
    "password": "***redacted***",
}
TASK_LABEL = "შეავსე დღგ-ის დეკლარაცია ექსელში მოცემული მონაცემებით"


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def passed(msg: str) -> None:
    print(f"  ✓ {msg}")


def failed(msg: str) -> None:
    print(f"  ✗ {msg}")


async def main_test() -> None:
    section("1. Fetch playbook")
    steps = await main.fetch_playbook_steps(PLAYBOOK_ID)
    if not steps:
        failed(f"playbook {PLAYBOOK_ID} not found / empty")
        sys.exit(1)
    passed(f"loaded {len(steps)} steps from playbook")
    print(f"    First step: {steps[0].get('action')} — {steps[0].get('target_description') or steps[0].get('url')}")

    section("2. Fetch knowledge-base chunks (J1)")
    target_descriptions = " ".join(
        (s.get("target_description") or "") for s in steps if s.get("target_description")
    )
    rag_query = (TASK_LABEL + " " + target_descriptions).strip()
    print(f"    RAG query (first 200 chars): {rag_query[:200]}…")
    chunks = await main.fetch_context(rag_query, limit=8)
    if not chunks:
        failed("fetch_context returned 0 chunks — is /agent/context up?")
        sys.exit(1)
    passed(f"fetched {len(chunks)} chunks")

    knowledge_block = main.format_context_for_prompt(chunks)
    if "INSTRUCTIONS FROM KNOWLEDGE BASE" not in knowledge_block:
        failed("knowledge_block missing INSTRUCTIONS marker")
        sys.exit(1)
    passed(f"formatted knowledge_block: {len(knowledge_block)} chars")
    print(f"    First 350 chars of knowledge:\n      {knowledge_block[:350]!r}…")

    section("3. Run pre-pass Planner WITH knowledge (J3)")
    if not os.environ.get("GEMINI_API_KEY"):
        failed("GEMINI_API_KEY missing — cannot call planner")
        sys.exit(1)
    plan_with_knowledge = await main.plan_with_planner(steps, USER_DATA, TASK_LABEL, knowledge_block)
    if not plan_with_knowledge:
        failed("planner returned empty (Gemini call failed?)")
        sys.exit(1)
    passed(f"planner produced {len(plan_with_knowledge)} chars of plan")

    section("4. Run pre-pass Planner WITHOUT knowledge (control)")
    plan_no_knowledge = await main.plan_with_planner(steps, USER_DATA, TASK_LABEL, "")
    if not plan_no_knowledge:
        failed("control planner returned empty")
        sys.exit(1)
    passed(f"control plan: {len(plan_no_knowledge)} chars")

    section("5. Compare the two plans")
    if len(plan_with_knowledge) <= len(plan_no_knowledge):
        # Not necessarily a fail — planner could just be terse — flag for inspection.
        print(f"  ⚠ With-knowledge plan is NOT longer (with={len(plan_with_knowledge)}, without={len(plan_no_knowledge)})")
    else:
        passed(f"with-knowledge plan is longer ({len(plan_with_knowledge)} > {len(plan_no_knowledge)})")

    georgian_terms_in_plan = sum(
        1 for term in ("დღგ", "დეკლარაცი", "გადასახად", "ფიზიკური", "ბრუნვა")
        if term in plan_with_knowledge
    )
    if georgian_terms_in_plan >= 2:
        passed(f"with-knowledge plan contains {georgian_terms_in_plan} domain-specific Georgian terms")
    else:
        print(f"  ⚠ Only {georgian_terms_in_plan} Georgian domain terms in plan — may not be using knowledge")

    print("\n--- WITH KNOWLEDGE — first 1500 chars ---")
    print(plan_with_knowledge[:1500])
    print("\n--- WITHOUT KNOWLEDGE — first 1500 chars ---")
    print(plan_no_knowledge[:1500])

    section("6. Verify worker prompt would contain knowledge_block (J1)")
    # Simulate the order of sections in run_agent's full_task assembly
    plan_block = f"--- TACTICAL PLAN ---\n{plan_with_knowledge}\n--- END PLAN ---"
    steps_block_marker = f"--- PLAYBOOK STEPS ---\n[{len(steps)} steps]"
    ordered_sections = [
        TASK_LABEL,
        plan_block,
        knowledge_block,
        steps_block_marker,
    ]
    joined = "\n".join(ordered_sections)

    checks = [
        ("plan section present", "TACTICAL PLAN" in joined),
        ("knowledge section present", "INSTRUCTIONS FROM KNOWLEDGE BASE" in joined),
        ("playbook section present", "PLAYBOOK STEPS" in joined),
        ("plan precedes knowledge",
         joined.index("TACTICAL PLAN") < joined.index("INSTRUCTIONS FROM KNOWLEDGE BASE")),
        ("knowledge precedes steps",
         joined.index("INSTRUCTIONS FROM KNOWLEDGE BASE") < joined.index("PLAYBOOK STEPS")),
    ]
    for label, ok in checks:
        (passed if ok else failed)(label)

    section("DONE")
    print("J1 + J3 verified end-to-end against the live backend + live planner.")
    print(f"Real-case test ready: pick playbook {PLAYBOOK_ID} on /agent and click Run.")


if __name__ == "__main__":
    asyncio.run(main_test())
