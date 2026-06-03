import unittest
from types import SimpleNamespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import main
from browser_use.tools.views import InputTextAction


class _Action:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self, exclude_none=True, mode="json"):
        return self._payload


class _ModelOutput:
    def __init__(self, actions):
        self.action = actions


class _HistoryItem:
    def __init__(self, actions):
        self.model_output = _ModelOutput(actions)


class _History:
    def __init__(self, actions):
        self.history = [_HistoryItem(actions)]


class AgentLogicTests(unittest.TestCase):
    def test_authoritative_contract_classifies_required_optional_and_sensitive(self):
        contract = main._build_authoritative_data_contract(
            {
                "amount": "150.00",
                "password": "secret",
                "notes": "manual review",
                "zero_field": "0.00",
                "empty_field": "",
            },
            {"amount": "თანხა", "zero_field": "ნულოვანი ველი"},
            "Fill rs.ge declaration",
        )

        items = {item["key"]: item for item in contract["items"]}
        self.assertTrue(items["amount"]["required_for_fill"])
        self.assertTrue(items["password"]["required_for_fill"])
        self.assertFalse(items["notes"]["required_for_fill"])
        self.assertTrue(items["zero_field"]["required_for_fill"])
        self.assertFalse(items["empty_field"]["required_for_fill"])
        self.assertEqual(items["password"]["display_value"], "***")

    def test_authoritative_data_block_prioritizes_required_values(self):
        contract = main._build_authoritative_data_contract(
            {"amount": "150.00", "notes": "manual review"},
            {"amount": "თანხა"},
            "Fill rs.ge declaration",
        )
        block = main._format_authoritative_data_block(contract)

        self.assertIn("AUTHORITATIVE SPREADSHEET DATA", block)
        self.assertIn('-- REQUIRED VALUES TO ENTER --', block)
        self.assertIn('amount -> "თანხა" -> 150.00', block)
        self.assertIn('-- OPTIONAL / IGNORED VALUES --', block)

    def test_typed_value_check_matches_formatted_numbers_and_ignores_zero_fields(self):
        history = _History([
            _Action({"input_text": {"text": "150"}}),
            _Action({"input_text": {"text": "hello"}}),
        ])

        missing = main._check_typed_values(
            history,
            {
                "amount": "150.00",
                "zero_field": "0.00",
                "note": "hello",
                "password": "secret",
            },
        )

        self.assertEqual(missing, [])

    def test_halt_on_dangerous_accepts_ready_for_review_visual_state(self):
        ok, completion_state, error = main._visual_outcome_for_safety(
            "halt-on-dangerous",
            {
                "is_confirmation_page": False,
                "is_ready_for_review": True,
                "final_action_visible": True,
                "suspicious_zero_fields": [],
            },
        )

        self.assertTrue(ok)
        self.assertEqual(completion_state, main.COMPLETION_READY_FOR_REVIEW)
        self.assertIsNone(error)

    def test_auto_requires_confirmation_page(self):
        ok, completion_state, error = main._visual_outcome_for_safety(
            "auto",
            {
                "is_confirmation_page": False,
                "is_ready_for_review": True,
                "final_action_visible": True,
                "suspicious_zero_fields": [],
                "explanation": "Form is still open.",
            },
        )

        self.assertFalse(ok)
        self.assertEqual(completion_state, main.COMPLETION_FAILED)
        self.assertIn("not a confirmation", error or "")

    def test_halt_on_dangerous_rejects_irreversible_execution(self):
        ok, completion_state, error = main._visual_outcome_for_safety(
            "halt-on-dangerous",
            {
                "is_confirmation_page": True,
                "is_ready_for_review": False,
                "irreversible_action_executed": True,
                "suspicious_zero_fields": [],
            },
        )

        self.assertFalse(ok)
        self.assertEqual(completion_state, main.COMPLETION_FAILED)
        self.assertIn("irreversible action executed", error or "")

    def test_visual_zero_fields_fail_cache_postcondition(self):
        ok, completion_state, error = main._visual_outcome_for_safety(
            "halt-on-dangerous",
            {
                "is_ready_for_review": True,
                "suspicious_zero_fields": ["Taxable turnover"],
            },
        )

        self.assertFalse(ok)
        self.assertEqual(completion_state, main.COMPLETION_FAILED)
        self.assertIn("0.00", error or "")

    def test_dangerous_click_text_detects_final_actions(self):
        self.assertTrue(main._is_dangerous_click_text("Submit declaration"))
        self.assertTrue(main._is_dangerous_click_text("confirm payment"))
        self.assertFalse(main._is_dangerous_click_text("Open declarations menu"))

    def test_allowed_domains_expand_rs_ge_and_auth_exception(self):
        domains = main._coerce_allowed_domains("rs.ge")
        self.assertIn("rs.ge", domains)
        self.assertIn("*.rs.ge", domains)
        self.assertIn("id.gov.ge", domains)

    def test_contract_coverage_fails_when_required_value_never_typed(self):
        history = _History([_Action({"input_text": {"text": "hello"}})])
        contract = main._build_authoritative_data_contract(
            {"amount": "150.00", "note": "hello"},
            {"amount": "თანხა"},
            "Fill rs.ge declaration",
        )

        missing = main._check_authoritative_contract_coverage(history, contract)

        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["key"], "amount")

    def test_required_contract_items_verified_only_when_typed_and_dom_matched(self):
        contract = main._build_authoritative_data_contract(
            {"amount": "150.00", "note": "hello"},
            {"amount": "თანხა", "note": "შენიშვნა"},
            "Fill rs.ge declaration",
        )
        typed_log = [{"text": "150.00"}, {"text": "hello"}]
        dom_review = {
            "fields": [
                {"key": "amount", "matched": True},
                {"key": "note", "matched": False},
            ]
        }

        missing = main._check_required_contract_items_verified(contract, typed_log, dom_review)

        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["key"], "note")

    def test_lines_become_row_groups_not_scalar_items(self):
        contract = main._build_authoritative_data_contract(
            {
                "period_year": 2026,
                "period_month": 5,
                "lines": [
                    {"personal_id": "01001000001", "name": "A", "gross": 1000, "income_tax": 196},
                    {"personal_id": "02002000002", "name": "B", "gross": 2000, "income_tax": 400},
                ],
            },
            {},
            "Fill rs.ge payroll declaration",
        )
        # The array is pulled out as a row group, not a junk scalar item.
        self.assertEqual(contract["summary"]["row_groups"], 1)
        self.assertEqual(contract["summary"]["rows"], 2)
        self.assertNotIn("lines", [it["key"] for it in contract["items"]])
        # The rendered block instructs one-by-one entry and lists both ids.
        block = main._format_authoritative_data_block(contract)
        self.assertIn("ROWS TO ENTER ONE-BY-ONE", block)
        self.assertIn("01001000001", block)
        self.assertIn("02002000002", block)

    def test_row_coverage_flags_employee_whose_id_was_never_typed(self):
        contract = main._build_authoritative_data_contract(
            {"lines": [
                {"personal_id": "01001000001", "name": "A", "income_tax": 196},
                {"personal_id": "02002000002", "name": "B", "income_tax": 400},
            ]},
            {},
            "payroll",
        )
        # Agent only typed the first employee's id → the second is missing.
        history = _History([
            _Action({"input_text": {"text": "01001000001"}}),
            _Action({"input_text": {"text": "196"}}),
        ])
        missing = main._check_row_coverage(history, contract)
        self.assertEqual(len(missing), 1)
        self.assertIn("02002000002", missing[0]["label"])

    def test_row_coverage_passes_when_all_ids_typed(self):
        contract = main._build_authoritative_data_contract(
            {"lines": [
                {"personal_id": "01001000001", "name": "A"},
                {"personal_id": "02002000002", "name": "B"},
            ]},
            {},
            "payroll",
        )
        history = _History([
            _Action({"input_text": {"text": "01001000001"}}),
            _Action({"input_text": {"text": "02002000002"}}),
        ])
        self.assertEqual(main._check_row_coverage(history, contract), [])


class _ActorPage:
    def __init__(self, url="https://decl.rs.ge/decls.aspx", actor_element=None):
        self.url = url
        self.pressed = []
        self.readback = ""
        self.actor_element = actor_element
        self.dom_set_values = []
        self.accept_numeric_recovery = False
        self.accept_extjs_set = False
        self.extjs_values = []

    async def evaluate(self, js, *args):
        if "window.location.href" in js:
            return self.url
        if "Ext.getCmp" in js:
            value = args[1]
            self.extjs_values.append(value)
            if self.accept_extjs_set:
                self.readback = value
                return {"status": "ok", "domValue": value, "value": value}
            return "no_ext_cmp"
        if "target.dispatchEvent(new KeyboardEvent('keydown'" in js:
            value = args[1]
            self.dom_set_values.append(value)
            if self.accept_numeric_recovery and value in ("1000.00", "1000,00"):
                self.readback = "1000.00"
            return self.readback or value
        if "target.focus" in js:
            return {
                "isInner": True,
                "tag": "input",
                "type": "text",
                "className": "field decl_input_t number ksc377",
                "inputMode": "decimal",
                "xName": "COL_15",
                "id": "control_377_15",
                "disabled": False,
                "readOnly": False,
                "editable": True,
                "value": "",
            }
        if "return target.value" in js or "return target.value != null" in js:
            return self.readback
        return ""

    async def press(self, key):
        self.pressed.append(key)

    async def get_element(self, _backend_node_id):
        if self.actor_element is None:
            raise RuntimeError("no actor element")
        return self.actor_element


class _ActorElement:
    def __init__(self):
        self.fill_calls = []

    async def fill(self, value, clear=True):
        self.fill_calls.append((value, clear))


class _Node:
    def __init__(self, text="", xpath="//div[@id='wrapper']", backend_node_id=42):
        self.text = text
        self.xpath = xpath
        self.fill_calls = []
        self.backend_node_id = backend_node_id

    async def fill(self, value, clear=True):
        self.fill_calls.append((value, clear))
        self.text = value


class _EventResult:
    def __init__(self, payload=None):
        self.payload = payload

    def __await__(self):
        async def _done():
            return self
        return _done().__await__()

    async def event_result(self, raise_if_any=False, raise_if_none=False):
        return self.payload


class _EventBus:
    def dispatch(self, _event):
        return _EventResult()


class _BrowserSession:
    def __init__(self, node=None, page=None, selector_map=None):
        self._node = node
        self._page = page
        self._cached_selector_map = selector_map or {}
        self.event_bus = _EventBus()
        self._cdp_client_root = object()
        self.agent_focus_target_id = "target-1"

    async def get_element_by_index(self, _index):
        return self._node

    async def get_current_page(self):
        return self._page

    async def take_screenshot(self):
        return "fake-screenshot"


class AsyncAgentLogicTests(unittest.IsolatedAsyncioTestCase):
    def test_page_kind_detection(self):
        actor = _ActorPage()
        playwright = SimpleNamespace(locator=lambda *_args, **_kwargs: None, keyboard=SimpleNamespace())
        self.assertEqual(main._page_kind(actor), "actor_page")
        self.assertEqual(main._page_kind(playwright), "playwright_page")

    async def test_input_action_prefers_actor_compatible_fill(self):
        actor_element = _ActorElement()
        page = _ActorPage(actor_element=actor_element)
        node = _Node()
        session = _BrowserSession(node=node, page=page)
        tools = main._build_tools()
        action = tools.registry.registry.actions["input"].function
        params = InputTextAction(index=7, text="1000", clear=True)

        page.readback = "1000"
        result = await action(params=params, browser_session=session)

        self.assertIsNone(result.error)
        self.assertEqual(actor_element.fill_calls, [("1000", True)])
        self.assertEqual(page.pressed, ["Tab"])
        typed_log = getattr(tools, "_declario_typed_log")
        self.assertEqual(typed_log[-1]["strategy"], "element_fill")
        self.assertEqual(typed_log[-1]["page_kind"], "actor_page")
        self.assertEqual(typed_log[-1]["readback"], "1000")

    async def test_input_action_uses_dom_setter_when_actor_element_unavailable(self):
        page = _ActorPage(actor_element=None)
        node = _Node()
        session = _BrowserSession(node=node, page=page)
        tools = main._build_tools()
        action = tools.registry.registry.actions["input"].function
        params = InputTextAction(index=10, text="1000", clear=True)

        page.readback = "1000"
        result = await action(params=params, browser_session=session)

        self.assertIsNone(result.error)
        typed_log = getattr(tools, "_declario_typed_log")
        self.assertEqual(typed_log[-1]["strategy"], "dom_set")

    async def test_numeric_recovery_tries_decimal_formats(self):
        actor_element = _ActorElement()
        page = _ActorPage(actor_element=actor_element)
        page.readback = "0.00"
        page.accept_numeric_recovery = True
        node = _Node()
        session = _BrowserSession(node=node, page=page)
        tools = main._build_tools()
        action = tools.registry.registry.actions["input"].function
        params = InputTextAction(index=11, text="1000", clear=True)

        result = await action(params=params, browser_session=session)

        self.assertIsNone(result.error)
        typed_log = getattr(tools, "_declario_typed_log")
        self.assertIn("numeric_recovery", typed_log[-1]["strategy"])
        self.assertIn("1000.00", page.dom_set_values)

    async def test_extjs_set_value_recovers_mismatch(self):
        actor_element = _ActorElement()
        page = _ActorPage(actor_element=actor_element)
        page.readback = "0.00"
        page.accept_extjs_set = True
        node = _Node()
        session = _BrowserSession(node=node, page=page)
        tools = main._build_tools()
        action = tools.registry.registry.actions["input"].function
        params = InputTextAction(index=12, text="1000", clear=True)

        result = await action(params=params, browser_session=session)

        self.assertIsNone(result.error)
        typed_log = getattr(tools, "_declario_typed_log")
        self.assertIn("extjs_setValue", typed_log[-1]["strategy"])
        self.assertEqual(page.extjs_values[-1], "1000")

    async def test_input_action_returns_mismatch_warning(self):
        page = _ActorPage()
        node = _Node()
        session = _BrowserSession(node=node, page=page)
        tools = main._build_tools()
        action = tools.registry.registry.actions["input"].function
        params = InputTextAction(index=8, text="1000", clear=True)

        page.readback = "0.00"
        result = await action(params=params, browser_session=session)

        self.assertIsNotNone(result.error)
        self.assertIn("read-back mismatch", result.extracted_content)

    async def test_input_action_blocks_third_identical_repeat(self):
        page = _ActorPage()
        node = _Node()
        session = _BrowserSession(node=node, page=page)
        tools = main._build_tools()
        action = tools.registry.registry.actions["input"].function
        params = InputTextAction(index=8, text="1000", clear=True)

        page.readback = "1000"
        first = await action(params=params, browser_session=session)
        second = await action(params=params, browser_session=session)
        third = await action(params=params, browser_session=session)

        self.assertIsNone(first.error)
        self.assertIsNone(second.error)
        self.assertIsNotNone(third.error)
        self.assertIn("action-dedup", third.extracted_content)

    async def test_input_action_reports_stale_index(self):
        page = _ActorPage()
        session = _BrowserSession(node=None, page=page)
        tools = main._build_tools()
        action = tools.registry.registry.actions["input"].function
        params = InputTextAction(index=9, text="1000", clear=True)

        result = await action(params=params, browser_session=session)

        self.assertIsNotNone(result.error)
        self.assertIn("no longer available", result.extracted_content)

    async def test_discover_form_fields_negative_cache_prevents_repeat_loops(self):
        page = _ActorPage(url="https://decl.rs.ge/decls.aspx")
        session = _BrowserSession(node=None, page=page, selector_map={})
        tools = main._build_tools()
        action = tools.registry.registry.actions["discover_form_fields"].function
        original = main._vision_discover_fields

        async def _fake_vision(_screenshot, _url=""):
            return []

        main._field_discovery_cache.clear()
        main._vision_discover_fields = _fake_vision
        try:
            first = await action(browser_session=session, intent="")
            second = await action(browser_session=session, intent="")
        finally:
            main._vision_discover_fields = original
            main._field_discovery_cache.clear()

        self.assertIn("Do not call discover_form_fields again", first.extracted_content)
        self.assertIn("Do not call discover_form_fields again", second.extracted_content)

    def test_storage_export_guard_skips_when_root_client_missing(self):
        session = SimpleNamespace(
            export_storage_state=lambda *_args, **_kwargs: None,
            _cdp_client_root=None,
            cdp_client=None,
            agent_focus_target_id="target-1",
        )
        ok, reason = main._can_export_storage_state(session)
        self.assertFalse(ok)
        self.assertIn("root CDP client unavailable", reason)


if __name__ == "__main__":
    unittest.main()
