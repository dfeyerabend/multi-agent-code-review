"""
Company-rule checking for the MCP server.
Extracted verbatim from mcp_server.py; imported back by the check_company_rules tool.
The only body change vs. the original is the JSON path in _company_load_rules, which now
comes from config.COMPANY_RULES_PATH instead of __file__ (this module lives elsewhere now).
"""

import ast
import os
import json
import logging
from config import COMPANY_RULES_PATH

logger = logging.getLogger(__name__)


def _company_load_rules() -> dict:
    """
    Loads and validates the company rule set from knowledge_base/company_rules.json.

    Pipeline: helper called by the check_company_rules tool (this file) at the start of
    every run, before any code is checked. The path is resolved relative to this file so
    it works on any clone, mirroring create_database.py's __file__-based lookup.

    Individual malformed rules are skipped with a warning so one bad entry cannot disable
    the whole set; only a missing file, invalid JSON, or a broken top-level structure
    fails the load outright.

    Returns:
        dict with "status": "success" and "rules" (list of validated rule dicts), or
        "status": "error" and "message" naming what failed and the likely cause.
    """
    rules_path = COMPANY_RULES_PATH

    try:
        if not os.path.isfile(rules_path):
            logger.error("_company_load_rules: rule set file not found at %s", rules_path)
            return {"status": "error",
                    "message": f"_company_load_rules failed — rule set file not found at {rules_path}"}

        with open(rules_path, "r", encoding="utf-8") as f:
            raw = f.read()

        # invalid JSON is a config error, not a code problem — report it with the parse detail
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("_company_load_rules: invalid JSON in %s: %s", rules_path, e)
            return {"status": "error",
                    "message": f"_company_load_rules failed — invalid JSON in {rules_path}: {e}"}

        if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
            logger.error("_company_load_rules: bad structure in %s (expected object with a 'rules' list)", rules_path)
            return {"status": "error",
                    "message": f"_company_load_rules failed — {rules_path} must be an object with a 'rules' list"}

        # keep only rules the runner can dispatch: string id (for reporting) + string mechanism (for lookup)
        valid_rules = []
        for i, rule in enumerate(data["rules"]):
            if not isinstance(rule, dict) or not isinstance(rule.get("id"), str) or not isinstance(rule.get("mechanism"), str):
                logger.warning("_company_load_rules: skipping malformed rule at index %d (needs string 'id' and 'mechanism'): %r", i, rule)
                continue
            valid_rules.append(rule)

        if not valid_rules:
            logger.error("_company_load_rules: no usable rule in %s", rules_path)
            return {"status": "error",
                    "message": f"_company_load_rules failed — no usable rule found in {rules_path}"}

        logger.info("_company_load_rules: loaded %d rule(s) from %s", len(valid_rules), rules_path)
        return {"status": "success", "rules": valid_rules}

    except Exception as e:
        logger.error("_company_load_rules failed unexpectedly for %s: %s", rules_path, str(e))
        return {"status": "error",
                "message": f"_company_load_rules failed unexpectedly reading {rules_path}: {str(e)}"}


def _company_run_checks(tree: ast.AST, source_lines: list, rules: list) -> dict:
    """
    Runs every company rule against the parsed code and collects the results.

    Pipeline: helper called by the check_company_rules tool (this file) after the code is
    parsed and the rule set is loaded. Dispatches each rule to its mechanism via the
    _COMPANY_CHECKS registry (this file), isolating per-rule failures so one broken rule
    cannot stop the others.

    Args:
        tree:         Parsed AST of the reviewed code.
        source_lines: Code split into lines, forwarded to every mechanism.
        rules:        Validated rule dicts from _company_load_rules.

    Returns:
        dict with "findings" (list of standard-schema finding dicts, all rules merged)
        and "rule_errors" (dict of rule id → error message for rules that failed or had
        no registered mechanism).
    """
    findings = []
    rule_errors = {}

    try:
        if not isinstance(rules, list):
            logger.error("_company_run_checks: 'rules' must be a list, got %s", type(rules).__name__)
            rule_errors["_runner"] = f"rule set must be a list, got {type(rules).__name__}"
            return {"findings": findings, "rule_errors": rule_errors}

        for rule in rules:
            rule_id = rule.get("id", "<unknown>") if isinstance(rule, dict) else "<unknown>"
            mechanism_name = rule.get("mechanism") if isinstance(rule, dict) else None

            mechanism = _COMPANY_CHECKS.get(mechanism_name)
            if mechanism is None:
                logger.warning("_company_run_checks: rule %s has no registered mechanism '%s'", rule_id, mechanism_name)
                rule_errors[rule_id] = f"no registered mechanism '{mechanism_name}'"
                continue

            # wrap the call so an unforeseen crash in one mechanism can't abort the remaining rules
            try:
                result = mechanism(tree, source_lines, rule)
            except Exception as e:
                logger.error("_company_run_checks: mechanism '%s' crashed on rule %s: %s", mechanism_name, rule_id, e)
                rule_errors[rule_id] = f"mechanism '{mechanism_name}' crashed: {e}"
                continue

            if not isinstance(result, dict) or result.get("status") != "success":
                message = result.get("message", "mechanism returned no message") if isinstance(result, dict) else "mechanism returned a non-dict"
                logger.warning("_company_run_checks: rule %s failed: %s", rule_id, message)
                rule_errors[rule_id] = message
                continue

            findings.extend(result.get("findings", []))

        logger.debug("_company_run_checks: %d finding(s), %d rule error(s)", len(findings), len(rule_errors))
        return {"findings": findings, "rule_errors": rule_errors}

    except Exception as e:
        logger.error("_company_run_checks failed unexpectedly: %s", str(e))
        rule_errors["_runner"] = f"runner failed unexpectedly: {str(e)}"
        return {"findings": findings, "rule_errors": rule_errors}


def _company_check_naming(tree: ast.AST, source_lines: list[str], rule: dict) -> dict:
    """
    Flags functions that access the database but lack the required name prefix (rule 1.1).

    Pipeline: company-rule mechanism in mcp_server.py, dispatched by _company_run_checks
    (same module) via the _COMPANY_CHECKS registry for any rule whose "mechanism" is
    "naming_convention". Reached through the check_company_rules MCP tool the Analyzer calls.

    A function counts as a database function when its body calls any name in
    params["trigger_calls"] (e.g. db.query); if so, its name must start with one of
    params["required_prefixes"]. Each violating function becomes one finding.

    Args:
        tree:         Parsed AST of the reviewed code.
        source_lines: Code split into lines. Unused here, but part of the shared mechanism
                      signature so the runner can dispatch every mechanism the same way.
        rule:         One rule object from company_rules.json.

    Returns:
        dict with "status" ("success" | "error"), "findings" (list), "message" on error.
    """
    try:
        if not isinstance(tree, ast.AST):
            return {"status": "error",
                    "message": f"_company_check_naming: 'tree' must be an ast.AST, got {type(tree).__name__}",
                    "findings": []}
        if not isinstance(rule, dict):
            return {"status": "error",
                    "message": f"_company_check_naming: 'rule' must be a dict, got {type(rule).__name__}",
                    "findings": []}

        params = rule.get("params", {})
        trigger_calls = params.get("trigger_calls")
        required_prefixes = params.get("required_prefixes")
        if not isinstance(trigger_calls, list) or not trigger_calls:
            return {"status": "error",
                    "message": f"_company_check_naming: rule '{rule.get('id')}' needs a non-empty 'trigger_calls' list in params",
                    "findings": []}
        if not isinstance(required_prefixes, list) or not required_prefixes:
            return {"status": "error",
                    "message": f"_company_check_naming: rule '{rule.get('id')}' needs a non-empty 'required_prefixes' list in params",
                    "findings": []}

        trigger_set = set(trigger_calls)
        prefixes = tuple(required_prefixes)          # str.startswith accepts a tuple of options
        findings = []

        # a function is judged only if its body actually calls one of the DB entry points
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            accesses_db = any(
                isinstance(call, ast.Call) and _company_dotted_name(call.func) in trigger_set
                for call in ast.walk(node)
            )
            if not accesses_db:
                continue                             # rule 1.1 applies only to DB functions

            if not node.name.startswith(prefixes):
                # function name in the message keeps distinct violations apart under later dedup
                findings.append({
                    "rule": rule["id"],
                    "message": f"{rule['message']} (function '{node.name}')",
                    "severity": rule.get("severity", "MEDIUM"),
                    "category": rule.get("category", "Maintainability"),
                    "lines": [node.lineno],
                    "occurrences": 1,
                })

        logger.debug("_company_check_naming: rule %s produced %d finding(s)", rule.get("id"), len(findings))
        return {"status": "success", "findings": findings}

    except Exception as e:
        rule_id = rule.get("id") if isinstance(rule, dict) else "<unknown>"
        logger.error("_company_check_naming crashed on rule %s: %s", rule_id, e)
        return {"status": "error",
                "message": f"_company_check_naming failed unexpectedly — likely a malformed AST node: {e}",
                "findings": []}


def _company_dotted_name(node: ast.AST) -> str | None:
    """
    Reconstructs the dotted name of an expression, e.g. a Name/Attribute chain → 'db.query'.

    Pipeline: shared private helper in mcp_server.py for the company-rule mechanisms
    (_company_check_naming, _company_check_raise, _company_check_access) to match calls/attributes
    against the string patterns in company_rules.json.

    Args:
        node: The func/value expression of a Call or Subscript (ast.Name or ast.Attribute).

    Returns:
        The dotted name (e.g. 'os.getenv'), or None if the expression is not a plain
        Name/Attribute chain (e.g. a call on a subscript or list literal).
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _company_dotted_name(node.value)              # recurse into the left side: Name('db') + '.query'
        return f"{base}.{node.attr}" if base else None
    return None                                      # Call/Subscript/etc. have no static dotted name

def _company_check_comment(tree: ast.AST, source_lines: list[str], rule: dict) -> dict:
    """
    Flags functions missing the required marker comment on/below the def line (rule 1.2).

    Pipeline: company-rule mechanism in mcp_server.py, dispatched by _company_run_checks
    (same module) via the _COMPANY_CHECKS registry for any rule whose "mechanism" is
    "required_comment". Reached through the check_company_rules MCP tool the Analyzer calls.

    Comments are not part of the AST, so this reads source_lines directly: the marker must
    appear on the def line or within params["max_lines_below"] lines beneath it.

    Args:
        tree:         Parsed AST of the reviewed code.
        source_lines: Code split into lines — the marker is matched against these.
        rule:         One rule object from company_rules.json.

    Returns:
        dict with "status" ("success" | "error"), "findings" (list), "message" on error.
    """
    try:
        if not isinstance(tree, ast.AST):
            return {"status": "error",
                    "message": f"_company_check_comment: 'tree' must be an ast.AST, got {type(tree).__name__}",
                    "findings": []}
        if not isinstance(source_lines, list):
            return {"status": "error",
                    "message": f"_company_check_comment: 'source_lines' must be a list, got {type(source_lines).__name__}",
                    "findings": []}
        if not isinstance(rule, dict):
            return {"status": "error",
                    "message": f"_company_check_comment: 'rule' must be a dict, got {type(rule).__name__}",
                    "findings": []}

        params = rule.get("params", {})
        marker = params.get("marker")
        max_lines_below = params.get("max_lines_below", 1)
        if not isinstance(marker, str) or not marker:
            return {"status": "error",
                    "message": f"_company_check_comment: rule '{rule.get('id')}' needs a non-empty string 'marker' in params",
                    "findings": []}
        if not isinstance(max_lines_below, int) or isinstance(max_lines_below, bool) or max_lines_below < 0:
            return {"status": "error",
                    "message": f"_company_check_comment: rule '{rule.get('id')}' needs a non-negative int 'max_lines_below' in params",
                    "findings": []}

        findings = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            window = source_lines[node.lineno - 1: node.lineno + max_lines_below]   # def line + lines directly below, where the marker may sit
            if any(marker in line for line in window):
                continue

            findings.append({
                "rule": rule["id"],
                "message": f"{rule['message']} (function '{node.name}')",
                "severity": rule.get("severity", "LOW"),
                "category": rule.get("category", "Maintainability"),
                "lines": [node.lineno],
                "occurrences": 1,
            })

        logger.debug("_company_check_comment: rule %s produced %d finding(s)", rule.get("id"), len(findings))
        return {"status": "success", "findings": findings}

    except Exception as e:
        rule_id = rule.get("id") if isinstance(rule, dict) else "<unknown>"
        logger.error("_company_check_comment crashed on rule %s: %s", rule_id, e)
        return {"status": "error",
                "message": f"_company_check_comment failed unexpectedly — likely malformed source lines: {e}",
                "findings": []}


def _company_check_raise(tree: ast.AST, source_lines: list[str], rule: dict) -> dict:
    """
    Flags raise statements that raise a forbidden built-in exception (rule 1.3).

    Pipeline: company-rule mechanism in mcp_server.py, dispatched by _company_run_checks
    (same module) via the _COMPANY_CHECKS registry for any rule whose "mechanism" is
    "forbidden_raise". Reached through the check_company_rules MCP tool the Analyzer calls.

    Args:
        tree:         Parsed AST of the reviewed code.
        source_lines: Code split into lines. Unused here, but part of the shared mechanism
                      signature so the runner can dispatch every mechanism the same way.
        rule:         One rule object from company_rules.json.

    Returns:
        dict with "status" ("success" | "error"), "findings" (list), "message" on error.
    """
    try:
        if not isinstance(tree, ast.AST):
            return {"status": "error",
                    "message": f"_company_check_raise: 'tree' must be an ast.AST, got {type(tree).__name__}",
                    "findings": []}
        if not isinstance(rule, dict):
            return {"status": "error",
                    "message": f"_company_check_raise: 'rule' must be a dict, got {type(rule).__name__}",
                    "findings": []}

        forbidden = rule.get("params", {}).get("forbidden")
        if not isinstance(forbidden, list) or not forbidden:
            return {"status": "error",
                    "message": f"_company_check_raise: rule '{rule.get('id')}' needs a non-empty 'forbidden' list in params",
                    "findings": []}

        forbidden_set = set(forbidden)
        findings = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue                             # bare 'raise' re-raises the active exception — nothing to name

            target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc   # 'raise Foo(...)' holds the class in .func; 'raise Foo' holds it directly
            name = _company_dotted_name(target)
            if name in forbidden_set:
                findings.append({
                    "rule": rule["id"],
                    "message": f"{rule['message']} (raises '{name}')",
                    "severity": rule.get("severity", "MEDIUM"),
                    "category": rule.get("category", "Logic"),
                    "lines": [node.lineno],
                    "occurrences": 1,
                })

        logger.debug("_company_check_raise: rule %s produced %d finding(s)", rule.get("id"), len(findings))
        return {"status": "success", "findings": findings}

    except Exception as e:
        rule_id = rule.get("id") if isinstance(rule, dict) else "<unknown>"
        logger.error("_company_check_raise crashed on rule %s: %s", rule_id, e)
        return {"status": "error",
                "message": f"_company_check_raise failed unexpectedly — likely malformed AST node: {e}",
                "findings": []}

def _company_check_access(tree: ast.AST, source_lines: list[str], rule: dict) -> dict:
    """
    Flags forbidden direct accesses such as os.getenv(...) or os.environ[...] (rule 1.4).

    Pipeline: company-rule mechanism in mcp_server.py, dispatched by _company_run_checks
    (same module) via the _COMPANY_CHECKS registry for any rule whose "mechanism" is
    "forbidden_access". Reached through the check_company_rules MCP tool the Analyzer calls.

    Covers two AST shapes with one target list: a Call (os.getenv("K")) and a Subscript
    (os.environ["K"]).

    Args:
        tree:         Parsed AST of the reviewed code.
        source_lines: Code split into lines. Unused here, but part of the shared mechanism
                      signature so the runner can dispatch every mechanism the same way.
        rule:         One rule object from company_rules.json.

    Returns:
        dict with "status" ("success" | "error"), "findings" (list), "message" on error.
    """
    try:
        if not isinstance(tree, ast.AST):
            return {"status": "error",
                    "message": f"_company_check_access: 'tree' must be an ast.AST, got {type(tree).__name__}",
                    "findings": []}
        if not isinstance(rule, dict):
            return {"status": "error",
                    "message": f"_company_check_access: 'rule' must be a dict, got {type(rule).__name__}",
                    "findings": []}

        targets = rule.get("params", {}).get("targets")
        if not isinstance(targets, list) or not targets:
            return {"status": "error",
                    "message": f"_company_check_access: rule '{rule.get('id')}' needs a non-empty 'targets' list in params",
                    "findings": []}

        targets_set = set(targets)
        findings = []

        for node in ast.walk(tree):
            # a Call names the target in .func (os.getenv); a Subscript names it in .value (os.environ)
            if isinstance(node, ast.Call):
                name = _company_dotted_name(node.func)
            elif isinstance(node, ast.Subscript):
                name = _company_dotted_name(node.value)
            else:
                continue

            if name in targets_set:
                findings.append({
                    "rule": rule["id"],
                    "message": f"{rule['message']} (uses '{name}')",
                    "severity": rule.get("severity", "HIGH"),
                    "category": rule.get("category", "Security"),
                    "lines": [node.lineno],
                    "occurrences": 1,
                })

        logger.debug("_company_check_access: rule %s produced %d finding(s)", rule.get("id"), len(findings))
        return {"status": "success", "findings": findings}

    except Exception as e:
        rule_id = rule.get("id") if isinstance(rule, dict) else "<unknown>"
        logger.error("_company_check_access crashed on rule %s: %s", rule_id, e)
        return {"status": "error",
                "message": f"_company_check_access failed unexpectedly — likely malformed AST node: {e}",
                "findings": []}


# only wired mechanisms live here; the runner turns a missing mechanism into a rule_error, not a crash
_COMPANY_CHECKS = {
    "naming_convention": _company_check_naming,
    "required_comment": _company_check_comment,
    "forbidden_raise": _company_check_raise,
    "forbidden_access": _company_check_access,
}
