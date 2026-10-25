"""Semantic checks that ItergenGenerator runs after every generated statement.

Each rule looks at one step -- the statement IterGen just produced -- and returns a
violation reason, or None. On a violation the generator calls
`backward(unit="statement", num=1)` and lets the model try that statement again.

The rules live here, instead of inline in ItergenGenerator.generate(), so that

  * the generator can record WHICH rule caused each backtrack, and
  * the exact same code can be replayed offline over already generated outputs with the
    real syncode parser (no model, no GPU), to measure how often a rule would fire and
    how often it fires on a program that would have passed anyway.

The four original rules keep their original order in DEFAULT_RULES, and the first
violation found wins, as before. Two of them had bugs that kept them from ever firing on
ordinary code; both are fixed here (see rule_const_reassign and rule_orphan_break).
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# Grammar symbols whose occurrences the rules read through IterGen.view().
TRACKED_CATEGORIES = (
    "var_decl", "function_declaration", "function_parameter", "primary_safe_non_numeric",
    "expr_safe", "control_flow_statement",
)

# Names that are always in scope in the JavaScript the benchmarks run under node.
BASE_IDENTIFIERS = frozenset([
    "console", "Math", "Object", "Array", "String", "Number", "Boolean",
    "true", "false", "null", "undefined", "NaN", "Infinity",
    "this", "arguments", "window", "document", "global",
    "Error", "Symbol", "Date", "RegExp", "Map", "Set", "WeakMap", "WeakSet",
    "Promise", "JSON", "parseInt", "parseFloat", "isNaN", "isFinite",
    "encodeURI", "decodeURI", "encodeURIComponent", "decodeURIComponent",
    "require", "module", "exports", "process", "Buffer",
    "setTimeout", "clearTimeout", "setInterval", "clearInterval",
])

_PARAM_PATTERN = re.compile(r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*\s*)?\(([^)]*)\)')
_IDENT = re.compile(r'(var|let|const)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)')
_FOR_PAREN = re.compile(r'for\s*\($')  # matches 'for(' at end of a slice


def prompt_identifiers(prompt: str) -> Set[str]:
    """Name and parameters of the last function declared in the prompt.

    e.g. "function pancake_sort(nums){" -> {"pancake_sort", "nums"}. The name is included
    so that recursive calls are not reported as undeclared.
    """
    names: Set[str] = set()
    try:
        matches = _PARAM_PATTERN.findall(prompt)
        if matches:
            last_match = matches[-1]
            if last_match[0] and last_match[0].strip():
                names.add(last_match[0].strip())
            last_params = last_match[1]
            if last_params.strip():
                for p in (p.strip() for p in last_params.split(',')):
                    # Handle "nums", "a", "limit=10"
                    if p:
                        p_name = p.split('=')[0].strip().split()[0]
                        if p_name:
                            names.add(p_name)
    except Exception:
        pass
    return names


@dataclass
class StepContext:
    """Everything a rule may look at for one generation step."""
    full_text: str                           # prompt + everything generated so far
    body_start: int                          # index in full_text where generation began
    new: Dict[str, List[str]]                # symbols that appeared during this step
    new_pos: Dict[str, List[Tuple[int, int]]]  # their (start, end) in full_text
    identifiers: Set[str]                    # every name currently in scope
    decl_at_depth: Dict[str, List[Tuple[int, str]]]   # name -> [(brace depth, keyword)]
    const_identifiers: Set[str]


def build_context(generated: str, full_text: str, post_items: Dict[str, List[str]],
                  pre_counts: Dict[str, int], base_identifiers: Set[str],
                  post_positions: Optional[Dict[str, List[Tuple[int, int]]]] = None) -> StepContext:
    """Scan the text generated so far for declarations and scopes.

    `post_items[cat]` is IterGen.view(cat) after the step and `pre_counts[cat]` its length
    before the step, so `post_items[cat][pre_counts[cat]:]` is what this step added.
    `post_positions[cat]` are the matching (start, end) offsets into `full_text`, from the
    generator's SymbolPosMap; rules that need to know WHERE a symbol is use them.
    """
    # Build a scope-aware declaration map by scanning the generated text.
    # decl_at_depth[var_name] = list of (brace_depth, keyword) for every declaration of
    # that name seen so far. Starting depth is 1 because the prompt already opened the
    # function body with '{'. This distinguishes legitimate inner-scope shadowing
    #   (outer `let i = 0`; inner `for (let i = 0; ...)` at a deeper depth)
    # from true same-scope redeclarations.
    decl_at_depth: Dict[str, List[Tuple[int, str]]] = {}
    const_identifiers: Set[str] = set()
    current_identifiers = set(base_identifiers)

    scan_depth = 1
    scan_i = 0
    scan_text = generated
    paren_depth = 0        # track '(' nesting
    in_for_header = False  # True while inside a for(...) header
    for_paren_depth = 0    # the paren_depth at which the for-header opened
    while scan_i < len(scan_text):
        ch = scan_text[scan_i]
        if ch == '{':
            scan_depth += 1
            scan_i += 1
        elif ch == '}':
            scan_depth = max(1, scan_depth - 1)
            scan_i += 1
        elif ch == '(':
            paren_depth += 1
            # Check if the text up to and including this '(' ends with 'for('
            if _FOR_PAREN.search(scan_text[:scan_i + 1]):
                in_for_header = True
                for_paren_depth = paren_depth
            scan_i += 1
        elif ch == ')':
            if in_for_header and paren_depth == for_paren_depth:
                in_for_header = False
            paren_depth = max(0, paren_depth - 1)
            scan_i += 1
        elif ch in ('"', "'"):          # skip string literals
            q = ch
            scan_i += 1
            while scan_i < len(scan_text) and scan_text[scan_i] != q:
                if scan_text[scan_i] == '\\':
                    scan_i += 1
                scan_i += 1
            scan_i += 1
        elif scan_text[scan_i:scan_i + 2] == '//':  # skip line comments
            while scan_i < len(scan_text) and scan_text[scan_i] != '\n':
                scan_i += 1
        elif scan_text[scan_i:scan_i + 2] == '/*':  # skip block comments
            scan_i += 2
            while scan_i < len(scan_text) - 1 and scan_text[scan_i:scan_i + 2] != '*/':
                scan_i += 1
            scan_i += 2
        else:
            m = _IDENT.match(scan_text, scan_i)
            # Only match at a word boundary (not mid-identifier)
            if m and (scan_i == 0 or not (scan_text[scan_i - 1].isalnum() or scan_text[scan_i - 1] == '_')):
                kw, vname = m.group(1), m.group(2)
                current_identifiers.add(vname)
                if not in_for_header:
                    # for(let i) / for(const x) have their own per-iteration scope;
                    # exclude from same-scope redeclaration tracking and const tracking.
                    decl_at_depth.setdefault(vname, []).append((scan_depth, kw))
                    if kw == 'const':
                        const_identifiers.add(vname)
                scan_i = m.end()
            else:
                scan_i += 1

    # Also collect function parameters into current_identifiers
    for param_str in post_items["function_parameter"]:
        p_name = (param_str.split('=')[0].strip() if '=' in param_str else param_str.strip())
        if p_name:
            current_identifiers.add(p_name)

    # Implicit declarations (catch, for-in/of, arrow params)
    current_identifiers.update(re.findall(r'catch\s*\(\s*([a-zA-Z_$][a-zA-Z0-9_$]*)', full_text))
    current_identifiers.update(re.findall(r'for\s*\(\s*(?:var|let|const\s+)?([a-zA-Z_$][a-zA-Z0-9_$]*)\s+(?:of|in)', full_text))
    # Arrow params: single-form `x =>` and parenthesized `(a, b, c) =>`.
    # Missing the multi-param case is a real bug: callbacks like
    # `.map((word, index) => ...)` leave `word`/`index` undeclared and
    # trigger false-positive aborts the moment the body references them.
    current_identifiers.update(re.findall(r'(?:^|[\W])([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=>', full_text))
    for params_str in re.findall(r'\(\s*([^()]*?)\s*\)\s*=>', full_text):
        for p in params_str.split(','):
            p_name = p.strip().split('=')[0].strip().lstrip('.').split()[-1] if p.strip() else ''
            if p_name and re.match(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$', p_name):
                current_identifiers.add(p_name)

    new = {cat: post_items[cat][pre_counts.get(cat, 0):] for cat in post_items}
    new_pos = {cat: list(post_positions[cat][pre_counts.get(cat, 0):]) for cat in post_positions} \
        if post_positions else {}
    return StepContext(full_text=full_text, body_start=len(full_text) - len(generated),
                       new=new, new_pos=new_pos, identifiers=current_identifiers,
                       decl_at_depth=decl_at_depth, const_identifiers=const_identifiers)


def rule_redeclaration(ctx: StepContext) -> Optional[str]:
    """`let`/`const x` at brace depth D, when `x` is already declared at depth D."""
    for decl_str in ctx.new["var_decl"]:
        m = re.search(r'(var|let|const)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', decl_str)
        if not m:
            continue
        kw, var_name = m.group(1), m.group(2)
        if kw not in ('let', 'const'):
            continue
        entries = ctx.decl_at_depth.get(var_name, [])
        if len(entries) >= 2:
            new_depth = entries[-1][0]
            if any(d == new_depth for (d, _) in entries[:-1]):
                return f"Redeclared '{var_name}' at depth {new_depth}"
    return None


def rule_undeclared(ctx: StepContext) -> Optional[str]:
    """A plain identifier in expression position that nothing has declared.

    SPM's primary_safe_non_numeric only captures variable-position identifiers, not
    property names (.prop) or method names, so false positives are low. Word-boundary
    check: IterGen's view sometimes returns partial-BPE-token fragments (e.g. 'oL' while
    the model is mid-way through emitting 'toLowerCase'). Require the token to appear as a
    standalone identifier in the generated text -- if it only exists as a substring inside
    a longer identifier, it's a parser artifact, not a real variable use.
    """
    for prim_str in ctx.new["primary_safe_non_numeric"]:
        token = prim_str.strip()
        if not (re.match(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$', token) and token not in ctx.identifiers):
            continue
        standalone = re.search(r'(?<![a-zA-Z0-9_$])' + re.escape(token) + r'(?![a-zA-Z0-9_$])', ctx.full_text)
        if not standalone:
            continue
        return f"Undeclared: '{token}'"
    return None


_ASSIGNMENT_OPS = sorted(["=", "+=", "-=", "*=", "/=", "%=", "**=", ">>=", "<<=", ">>>=", "&=",
                          "^=", "&&=", "||=", "??="], key=len, reverse=True)
_OPS_PATTERN = '|'.join(map(re.escape, _ASSIGNMENT_OPS))


_UPDATE = re.compile(r'^\s*(?:(?:\+\+|--)\s*([a-zA-Z_$][a-zA-Z0-9_$]*)|([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(?:\+\+|--))')


def rule_const_reassign(ctx: StepContext) -> Optional[str]:
    """`constVar = ...` (or `+=`, ...) where the left-hand side is not a member access.

    FIX: the original checked assignment operators only, so `count++` / `--count` on a
    const -- a TypeError at run time -- was never caught.
    """
    for expr_str in ctx.new["expr_safe"]:
        m = re.match(r'^\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(' + _OPS_PATTERN + ')', expr_str)
        if m:
            lhs = m.group(1)
            is_member_access = bool(re.match(r'^\s*[a-zA-Z_$][a-zA-Z0-9_$]*\s*[.\[]', expr_str.lstrip()))
            if lhs in ctx.const_identifiers and not is_member_access:
                return f"Reassigned const: '{lhs}'"
        u = _UPDATE.match(expr_str)
        if u:
            name = u.group(1) or u.group(2)
            if name in ctx.const_identifiers:
                return f"Updated const: '{name}'"
    return None


_LEADING_KEYWORD = re.compile(r'\s*(break|continue)\b')


def _matching_open_paren(text: str, close: int) -> int:
    depth = 0
    for i in range(close, -1, -1):
        if text[i] == ')':
            depth += 1
        elif text[i] == '(':
            depth -= 1
            if depth == 0:
                return i
    return -1


def _block_kind(text: str, brace: int) -> str:
    """What opened the construct starting at text[brace] -- normally a `{` -- judged by
    the text right before it: 'loop', 'switch', 'function' or 'block'."""
    head = text[:brace].rstrip()
    if head.endswith('=>'):
        return 'function'
    word = re.search(r'([A-Za-z_$][\w$]*)$', head)
    if word and word.group(1) == 'do':
        return 'loop'
    if word and word.group(1) in ('else', 'try', 'finally'):
        return 'block'
    if head.endswith(')'):
        before = head[:_matching_open_paren(head, len(head) - 1)].rstrip()
        kw = re.search(r'([A-Za-z_$][\w$]*)$', before)
        kw = kw.group(1) if kw else ''
        if kw in ('for', 'while'):
            return 'loop'
        if kw == 'switch':
            return 'switch'
        if kw in ('if', 'catch', 'with'):
            return 'block'
        return 'function'                   # function name(...) { / method(...) {
    return 'block'                           # object literal or bare block


def _in_unbraced_loop_body(text: str, start: int, pos: int) -> bool:
    """True if the statement at `pos` is the (possibly nested) unbraced body of a loop,
    e.g. `for (...) break;` or `while (...) if (x) break; else continue;`."""
    p = pos
    for _ in range(8):
        head = text[start:p].rstrip()
        if not head:
            return False
        word = re.search(r'([A-Za-z_$][\w$]*)$', head)
        if word and word.group(1) == 'do':
            return True
        if word and word.group(1) == 'else':
            # `if (c) a; else <stmt>`: keep walking back past the if's then-branch
            semi = head.rfind(';', 0, len(head) - 4)
            if semi < 0:
                return False
            p = start + semi
            head = text[start:p].rstrip()
            # the then-branch statement itself is unbraced; jump to the `if (...)` before it
            ifs = list(re.finditer(r'\bif\s*\(', head))
            if not ifs:
                return False
            p = start + ifs[-1].start()
            continue
        if head.endswith(')'):
            open_at = _matching_open_paren(head, len(head) - 1)
            before = head[:open_at].rstrip()
            kw = re.search(r'([A-Za-z_$][\w$]*)$', before)
            kw = kw.group(1) if kw else ''
            if kw in ('for', 'while'):
                return True
            if kw == 'if':
                p = start + len(before) - len(kw)
                continue
        return False
    return False


def _enclosing_blocks(text: str, start: int, pos: int) -> List[str]:
    """Kinds of the blocks still open at `pos`, scanning text[start:pos], innermost last."""
    stack: List[str] = []
    i = start
    while i < pos:
        ch = text[i]
        if ch in ('"', "'", '`'):
            q = ch
            i += 1
            while i < pos and text[i] != q:
                i += 2 if text[i] == '\\' else 1
            i += 1
            continue
        if text.startswith('//', i):
            nl = text.find('\n', i)
            i = pos if nl < 0 else nl
            continue
        if text.startswith('/*', i):
            end = text.find('*/', i + 2)
            i = pos if end < 0 else end + 2
            continue
        if ch == '{':
            stack.append(_block_kind(text, i))
        elif ch == '}' and stack:
            stack.pop()
        i += 1
    return stack


def rule_orphan_break(ctx: StepContext) -> Optional[str]:
    """`break` / `continue` that no enclosing loop (or, for break, switch) can receive.

    FIX: the original took `cf_str.split()[0]`, which for the usual `break;` is the string
    "break;" -- never equal to "break" -- so the rule could not fire. It also accepted any
    break as soon as the words for/while/do appeared ANYWHERE in the prompt or the code.
    This version looks at the blocks actually enclosing the statement, and treats a
    function boundary as a wall, which also catches `arr.forEach(x => { break; })`.
    """
    starts = [s for s, _ in ctx.new_pos.get("control_flow_statement", [])]
    for idx, cf_str in enumerate(ctx.new["control_flow_statement"]):
        m = _LEADING_KEYWORD.match(cf_str)
        if not m:
            continue
        kw = m.group(1)
        pos = starts[idx] if idx < len(starts) else ctx.full_text.rfind(cf_str)
        if pos < 0:
            continue
        if _in_unbraced_loop_body(ctx.full_text, ctx.body_start, pos):
            continue                          # for (...) break;   for (...) if (x) break;
        # the prompt opened the function body: everything generated is inside a function
        for kind in reversed(['function'] + _enclosing_blocks(ctx.full_text, ctx.body_start, pos)):
            if kind == 'loop' or (kind == 'switch' and kw == 'break'):
                break
            if kind == 'function':
                return f"'{kw}' outside any loop" + ("" if kw == 'continue' else " or switch")
    return None


RULES = {
    "redeclaration": rule_redeclaration,
    "undeclared": rule_undeclared,
    "const_reassign": rule_const_reassign,
    "orphan_break": rule_orphan_break,
}

# The four original rules, in their original order.
DEFAULT_RULES = ("redeclaration", "undeclared", "const_reassign", "orphan_break")


def first_violation(ctx: StepContext, rules=DEFAULT_RULES) -> Optional[Tuple[str, str]]:
    """(rule name, reason) of the first rule that fires, or None."""
    for name in rules:
        reason = RULES[name](ctx)
        if reason:
            return name, reason
    return None
