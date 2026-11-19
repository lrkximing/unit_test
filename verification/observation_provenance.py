"""Local syntactic observation checks; not a proof of an oracle or alias analysis."""
import re

from validation.test_inspector import C_PARSER, _node_text, _walk
from verification.memory_observations import Captures, byte_arrays, integer
from verification.object_relations import resolve_path, expression_path


def _path(node, data, aliases=None):
    """Resolve only explicit lvalues and local pointer aliases, not points-to sets."""
    if node is None:
        return ""
    if node.type == "identifier":
        name = _node_text(data, node)
        return resolve_path(name, aliases or {})
    if node.type == "field_expression":
        base = _path(node.child_by_field_name("argument"), data, aliases)
        field = node.child_by_field_name("field")
        return resolve_path(base + "." + _node_text(data, field), aliases or {}) if base and field else ""
    if node.type == "subscript_expression":
        base = _path(node.child_by_field_name("argument"), data, aliases)
        index = node.child_by_field_name("index")
        # A nonconstant index may alias any element of this array.
        value = _node_text(data, index) if index and index.type == "number_literal" else "*"
        return base + "[" + value + "]" if base else ""
    if node.type in {"pointer_expression", "parenthesized_expression", "cast_expression"}:
        child = node.child_by_field_name("argument") or node.child_by_field_name("value")
        if child is None and node.type == "parenthesized_expression" and node.named_children:
            child = node.named_children[0]
        return _path(child, data, aliases)
    return ""


def _reads(node, data, aliases=None):
    if node is None:
        return set()
    path = _path(node, data, aliases)
    if path:
        return {path}
    return set().union(*(_reads(c, data, aliases) for c in node.named_children))


def _overlaps(left, right):
    a = re.findall(r"[^.\[\]]+", left)
    b = re.findall(r"[^.\[\]]+", right)
    return bool(a and b) and all(x == y or "*" in (x, y) for x, y in zip(a, b))


def _field_observation_matches(path, target):
    data = ("void f(void){ " + target + "; }").encode()
    tree = C_PARSER.parse(data)
    expressions = [node for node in _walk(tree.root_node) if node.type == "expression_statement"]
    if not expressions or not expressions[0].named_children:
        return False
    expected = _path(expressions[0].named_children[0], data)
    expected_parts = re.findall(r"[^.\[\]]+", expected)[1:]
    actual_parts = re.findall(r"[^.\[\]]+", path)
    return bool(expected_parts) and len(actual_parts) > len(expected_parts) and all(
        left == right or "*" in (left, right) for left, right in zip(actual_parts[-len(expected_parts):], expected_parts))


def captured_mock_arguments(binding, *, mock_name=None, parameter_index=None):
    """Follow connected fakes into local helpers with explicit argument mapping."""
    if not binding.support_source or not binding.mock_names:
        return Captures()
    from validation.test_inspector import _function_name_from_definition, inspect_test_source
    data = binding.support_source.encode()
    tree = C_PARSER.parse(data)
    functions = {_function_name_from_definition(fn, data): fn for fn in _walk(tree.root_node)
                 if fn.type == "function_definition"}
    globals_ = set(inspect_test_source(binding.support_source).global_identifier_names)
    captures = Captures()
    arrays = byte_arrays(binding.support_source)
    production_arrays = byte_arrays(binding.observation_context.get('source', ''))

    def text(node):
        return _node_text(data, node) if node is not None else ""

    def declaration_name(node):
        while node is not None and node.type != "identifier":
            node = node.child_by_field_name("declarator")
        return text(node)

    def parameters(fn):
        declarator = fn.child_by_field_name("declarator")
        while declarator is not None and declarator.type != "function_declarator":
            declarator = declarator.child_by_field_name("declarator")
        args = declarator.child_by_field_name("parameters") if declarator is not None else None
        return [declaration_name(p.child_by_field_name("declarator")) for p in args.named_children
                if p.type == "parameter_declaration"] if args is not None else []

    def visit(name, aliases, tainted, stack, numeric=None):
        if name not in functions or name in stack:
            return
        fn = functions[name]
        aliases, tainted, numeric = dict(aliases), set(tainted), dict(numeric or {})
        locals_ = set(parameters(fn))
        body = fn.child_by_field_name("body")
        for node in _walk(body):
            if node.type == "declaration":
                locals_.update(declaration_name(d) for d in node.children_by_field_name("declarator"))
        def derived(node):
            return node is not None and any(text(n) in tainted for n in _walk(node) if n.type == "identifier")
        events = [node for node in _walk(body) if node.type in
                  {"call_expression", "assignment_expression", "init_declarator", "update_expression"}]
        for node in sorted(events, key=lambda n: (n.end_byte, -n.start_byte)):
            if node.type == "call_expression":
                callee = text(node.child_by_field_name("function"))
                args = node.child_by_field_name("arguments")
                values = list(args.named_children) if args is not None else []
                if callee in {'memcpy', 'memmove', 'memset'} and len(values) == 3:
                    destination = _path(values[0], data, aliases)
                    root = re.split(r'[.\[]', destination)[0]
                    if root in globals_ and root not in locals_:
                        captures.invalidate(destination, _overlaps)
                        size = integer(text(values[2]), numeric, arrays)
                        if callee != 'memset' and derived(values[1]) and size is not None and size > 0 and destination in arrays:
                            captures.add(destination)
                            captures.byte_ranges[destination] = min(size, arrays[destination])
                    continue
                if callee not in functions:
                    continue
                names = parameters(functions[callee])
                mapping, flows, numbers = {}, set(), {}
                for formal, actual in zip(names, values):
                    if not formal:
                        continue
                    path = _path(actual, data, aliases)
                    if path:
                        mapping[formal] = path
                    if derived(actual):
                        flows.add(formal)
                    numbers[formal] = integer(text(actual), numeric, arrays)
                visit(callee, mapping, flows, stack | {name}, numbers)
                continue
            if node.type == "update_expression":
                lhs, rhs = node.child_by_field_name("argument"), None
            elif node.type == "assignment_expression":
                lhs, rhs = node.child_by_field_name("left"), node.child_by_field_name("right")
            else:
                lhs, rhs = node.child_by_field_name("declarator"), node.child_by_field_name("value")
            if lhs is None:
                continue
            path = _path(lhs, data, aliases)
            root = re.split(r"[.\[]", path)[0]
            if root in globals_ and root not in locals_:
                captures.invalidate(path, _overlaps)
                if rhs is not None and derived(rhs):
                    captures.add(path)
            local = declaration_name(lhs)
            if lhs.type == "identifier" or node.type == "init_declarator":
                if rhs is not None and derived(rhs):
                    tainted.add(local)
                else:
                    tainted.discard(local)
                aliases.pop(local, None)
                numeric[local] = integer(text(rhs), numeric, arrays) if rhs is not None else None
                if rhs is not None:
                    target = _path(rhs, data, aliases)
                    if target and re.split(r"[.\[]", target)[0] in globals_:
                        aliases[local] = target

    for name in binding.mock_names:
        if name in functions and (mock_name is None or mock_name == name):
            names = parameters(functions[name])
            selected = names if parameter_index is None else names[parameter_index:parameter_index + 1]
            numbers = {}
            boundaries = [b for b in binding.boundary_candidates if name in binding.boundary_mocks.get(b.get('candidate_id'), [])]
            facts = [f for b in boundaries for f in binding.source_facts if f.get('fact_id') in b.get('fact_ids', []) and f.get('kind') == 'CALL']
            for index, formal in enumerate(names):
                values = [integer(f.get('metadata', {}).get('arguments', [])[index], {}, production_arrays)
                          for f in facts if len(f.get('metadata', {}).get('arguments', [])) > index]
                if values and all(v is not None and v == values[0] for v in values):
                    numbers[formal] = values[0]
            visit(name, {}, set(selected), set(), numbers)
    return captures


def local_observations(binding, wrapper):
    """Return the live return aliases, actual argument roots, and prior target calls."""
    if C_PARSER is None or not binding.source_text or not wrapper:
        return None
    code = binding.source_text.encode()
    tree = C_PARSER.parse(code)
    returns, states, literals = set(), set(), {}
    aliases, overwritten, snapshots, value_identity = {}, set(), set(), {}
    stores, target_aliases = {}, {}
    from verification.object_relations import accessor_write, expression_path, parameters
    models = binding.observation_context.get('accessors', {})
    support = binding.support_source.encode()
    helpers = {}
    from validation.test_inspector import _function_name_from_definition
    for fn in _walk(C_PARSER.parse(support).root_node):
        if fn.type == 'function_definition':
            helpers[_function_name_from_definition(fn, support)] = fn

    def setup_call(name, arguments, stack=()):
        if name in {'memcpy', 'memmove', 'memset'} and arguments and arguments[0]:
            if calls:
                overwritten.add(arguments[0])
            return
        write = accessor_write(name, arguments, models, stores)
        if write:
            slot, value = write
            for old in list(stores):
                if _overlaps(old, slot):
                    del stores[old]
            stores[slot] = value
            return
        if name not in helpers or name in stack or name == wrapper:
            return
        fn = helpers[name]
        mapping = dict(zip(parameters(fn, support), arguments))
        # Only unconditional setup operations in the called helper's own body.
        body = fn.child_by_field_name('body')
        for statement in body.named_children:
            if statement.type != 'expression_statement' or not statement.named_children:
                continue
            expression = statement.named_children[0]
            if expression.type == 'call_expression':
                callee = _node_text(support, expression.child_by_field_name('function'))
                actual = [expression_path(_node_text(support, arg), mapping, models, stores=stores)
                          for arg in expression.child_by_field_name('arguments').named_children]
                setup_call(callee, actual, (*stack, name))
            elif expression.type == 'assignment_expression':
                if _node_text(support, expression.child_by_field_name('operator')) != '=':
                    continue
                left = expression_path(_node_text(support, expression.child_by_field_name('left')), mapping, models)
                right = expression_path(_node_text(support, expression.child_by_field_name('right')), mapping, models, stores=stores)
                if left and right:
                    stores[left] = right
                elif left:
                    stores.pop(left, None)
                if left and calls:
                    overwritten.add(left)

    def source_objects(arguments):
        source = binding.observation_context.get('source', '').encode()
        fn = next((n for n in _walk(C_PARSER.parse(source).root_node) if n.type == 'function_definition'), None)
        mapping = dict(zip(parameters(fn, source), arguments)) if fn is not None else {}
        pointer_parameters = {key for fact in binding.source_facts
                              for key, type_ in fact.get('metadata', {}).get('function_parameter_types', {}).items()
                              if '*' in type_}
        object_names = set(pointer_parameters)
        for fact in sorted(binding.source_facts, key=lambda f: f.get('metadata', {}).get('source_order', 0)):
            meta = fact.get('metadata', {})
            if fact.get('kind') != 'CALL' or meta.get('control_context'):
                continue
            destination = meta.get('result_assignee', '').lstrip('*')
            if not re.fullmatch(r'[A-Za-z_]\w*', destination):
                continue
            value = expression_path(fact.get('code', ''), mapping, models, stores=stores)
            mapping[destination] = value
            if value:
                object_names.add(destination)
        return {key: value for key, value in mapping.items() if value and key in object_names}
    captures = captured_mock_arguments(binding)
    calls = []
    def text(node):
        return _node_text(code, node) if node is not None else ""
    def identifiers(node):
        return {text(n) for n in _walk(node) if n.type == "identifier"} if node is not None else set()
    def has_target(node):
        return any(n.type == "call_expression" and text(n.child_by_field_name("function")) == wrapper
                   for n in _walk(node))
    pointer_objects = set()
    for decl in _walk(tree.root_node):
        if decl.type in {"pointer_declarator", "array_declarator"}:
            pointer_objects.update(identifiers(decl))
    def live(path):
        return (bool(calls) and (any(_overlaps(path, root) for root in states) or captures.observes(path, binding))
                and not any(_overlaps(path, dead) for dead in overwritten)) or path in snapshots
    events = [n for n in _walk(tree.root_node) if n.type in {"call_expression", "init_declarator", "assignment_expression", "update_expression"}
              and n.start_byte < binding.start_byte]
    for node in sorted(events, key=lambda n: (n.end_byte, n.start_byte)):
        if node.type == "call_expression":
            callee = text(node.child_by_field_name("function"))
            if callee == wrapper:
                calls.append(node.start_byte)
                refreshed = set(captures)
                args = node.child_by_field_name("arguments")
                actual = [expression_path(text(arg), aliases, models, stores=stores)
                          for arg in args.named_children] if args is not None else []
                target_aliases = source_objects(actual)
                states.update(target_aliases.values())
                refreshed.update(target_aliases.values())
                for arg in args.named_children if args is not None else []:
                    if text(arg).lstrip().startswith("&"):
                        path = _path(arg, code, aliases)
                        if path:
                            states.add(path)
                            refreshed.add(path)
                    elif arg.type == "identifier" and text(arg) in pointer_objects:
                        path = _path(arg, code, aliases)
                        states.add(path)
                        refreshed.add(path)
                overwritten = {p for p in overwritten if not any(_overlaps(p, r) for r in refreshed)}
            elif callee in {'memcpy', 'memmove', 'memset'}:
                args = node.child_by_field_name('arguments').named_children
                if len(args) == 3:
                    destination = _path(args[0], code, aliases)
                    if destination:
                        if calls:
                            overwritten.add(destination)
                        if callee != 'memset':
                            origin = _path(args[1], code, aliases)
                            if origin:
                                value_identity[destination] = value_identity.get(origin, origin)
            elif callee == "KUNIT_ASSERT_PTR_EQ":
                args = node.child_by_field_name("arguments")
                values = args.named_children[1:] if args is not None else []
                if len(values) == 2:
                    paths = [_path(value, code, aliases) for value in values]
                    if any(p in returns and p in pointer_objects for p in paths):
                        states.update(p for p in paths if p)
            elif not callee.startswith(('KUNIT_', 'kunit_')):
                args = node.child_by_field_name('arguments').named_children
                actual = [expression_path(text(arg), aliases, models, stores=stores) for arg in args]
                setup_call(callee, actual)
            continue
        if node.type == "update_expression":
            path = _path(node.child_by_field_name("argument"), code, aliases)
            if path:
                overwritten.add(path)
                snapshots.discard(path)
                value_identity.pop(path, None)
            continue
        lhs = node.child_by_field_name("left") if node.type == "assignment_expression" else node.child_by_field_name("declarator")
        rhs = node.child_by_field_name("right") if node.type == "assignment_expression" else node.child_by_field_name("value")
        if lhs is None or rhs is None:
            continue
        while lhs.type in {"pointer_declarator", "parenthesized_declarator", "array_declarator"} and lhs.child_by_field_name("declarator"):
            lhs = lhs.child_by_field_name("declarator")
        path = text(lhs) if lhs.type == "identifier" and text(lhs) in pointer_objects else _path(lhs, code, aliases)
        if path:
            for old in list(stores):
                if _overlaps(path, old):
                    del stores[old]
        observation_copy = any(live(p) for p in _reads(rhs, code, aliases))
        if calls and path:
            overwritten.add(path)
            snapshots = {p for p in snapshots if not _overlaps(p, path)}
        if lhs.type != "identifier":
            continue
        name = text(lhs)
        # Reject only explicit copies of the same observed value, not two
        # independent target invocations or independently derived expectations.
        if rhs.type == "identifier":
            value_identity[name] = value_identity.get(text(rhs), text(rhs))
        elif rhs.type == "call_expression":
            value_identity[name] = "call@" + str(rhs.start_byte)
        else:
            value_identity.pop(name, None)
        derived = has_target(rhs) or bool(identifiers(rhs) & returns)
        if observation_copy and name not in pointer_objects:
            snapshots.add(name)
        if name in pointer_objects:
            rhs_path = _path(rhs, code, aliases)
            aliases.pop(name, None)
            if rhs_path:
                aliases[name] = rhs_path
        returns.discard(name)
        if derived:
            returns.add(name)
            if name in pointer_objects:
                states.add(name)
                overwritten.discard(name)
        value = text(rhs).strip()
        literals[name] = value if re.fullmatch(r"-?(?:0[xX][0-9a-fA-F]+|\d+)[uUlL]*", value) else literals.get(value)
    return {"returns": returns, "states": states, "calls": calls, "literals": literals,
            "aliases": aliases, "overwritten": overwritten, "snapshots": snapshots, "captures": captures,
            "value_identity": value_identity, "target_aliases": target_aliases}


def observed_target(binding, check, wrapper, value_args):
    evidence = local_observations(binding, wrapper)
    if evidence is None:
        # Legacy callers without source/context only get the vacuity screen.
        return True
    target = str(check.get("target", ""))
    kind = str(check.get("kind", ""))
    expressions = " ".join(value_args)
    ids = set(re.findall(r"\b[A-Za-z_]\w*\b", expressions))
    direct_return = bool(re.search(r"\b" + re.escape(wrapper) + r"\s*\(", expressions))
    returns = direct_return or bool(ids & evidence["returns"])
    if kind.startswith("Return") or target == "return_value":
        return returns
    if kind.startswith("Boundary") and kind != "BoundaryArgument":
        # Callback counters are authored in fake state, not necessarily exported
        # hook counters. They remain witness checks and cannot be the primary oracle.
        return True
    operand_data = ("void f(void){ " + "; ".join(value_args) + "; }").encode()
    operand_tree = C_PARSER.parse(operand_data)
    paths = set()
    for node in _walk(operand_tree.root_node):
        if node.type == "expression_statement":
            paths.update(_reads(node, operand_data, evidence["aliases"]))
    live_paths = {p for p in paths if p in evidence["snapshots"] or
                  (evidence["calls"] and (any(_overlaps(p, r) for r in evidence["states"])
                                         or evidence['captures'].observes(p, binding))
                   and not any(_overlaps(p, dead) for dead in evidence["overwritten"]))}
    if kind == "BoundaryArgument":
        captures = []
        for mock in binding.boundary_mocks.get(check.get("boundary_id"), []):
            captures.append(captured_mock_arguments(binding, mock_name=mock,
                                                     parameter_index=check.get("argument_index")))
        return bool(evidence["calls"]) and any(capture.observes(p, binding) for p in live_paths for capture in captures)
    if "->" in target or "." in target:
        projected = set()
        for mock_name, parameter_index in binding.field_projections.get(target, []):
            projected.update(captured_mock_arguments(binding, mock_name=mock_name, parameter_index=parameter_index))
        root = re.match(r'[A-Za-z_]\w*', target)
        grounded = (expression_path(target, evidence['target_aliases'], binding.observation_context.get('accessors', {}))
                    if root and root[0] in evidence['target_aliases'] else '')
        matches = (any(p == grounded for p in live_paths) if grounded else
                   any(_field_observation_matches(p, target) for p in live_paths))
        return (matches
                or any(_overlaps(p, capture) for p in live_paths for capture in projected)
                or bool(live_paths & evidence["snapshots"]))
    return returns or bool(live_paths)
