"""Syntactic installation evidence, distinct from runtime boundary reachability."""

from validation.test_inspector import C_PARSER, _node_text, _walk, _function_name_from_definition


def hook_installations(code, setters):
    if C_PARSER is None:
        return []
    data = code.encode()
    result = []
    for node in _walk(C_PARSER.parse(data).root_node):
        if node.type != "call_expression":
            continue
        function = node.child_by_field_name("function")
        args = node.child_by_field_name("arguments")
        name = _node_text(data, function) if function else ""
        if name not in setters or not args or len(args.named_children) != 1:
            continue
        arg = args.named_children[0]
        if arg.type == "pointer_expression":
            arg = arg.child_by_field_name("argument")
        if arg is not None and arg.type == "identifier":
            replacement = _node_text(data, arg)
            if replacement != "NULL":
                result.append((name, replacement))
    return result


def connected(code, replacement, *, setter="", field=""):
    if not replacement or C_PARSER is None:
        return False
    if setter:
        return (setter, replacement) in hook_installations(code, {setter})
    data = code.encode()
    for node in _walk(C_PARSER.parse(data).root_node):
        if node.type not in {"assignment_expression", "initializer_pair"}:
            continue
        value = node.child_by_field_name("right") or node.child_by_field_name("value")
        if value is None:
            continue
        names = {_node_text(data, n) for n in _walk(value) if n.type == "identifier"}
        if replacement not in names:
            continue
        left = node.child_by_field_name("left") or node.child_by_field_name("designator")
        if left is None:
            continue
        fields = [_node_text(data, n) for n in _walk(left) if n.type == "field_identifier"]
        if fields and (not field or fields[-1] == field):
            return True
    return False


def test_support(test, code, info, wrapper):
    """Keep only this test's pre-invocation dependency closure, never siblings'."""
    data = code.encode()
    tree = C_PARSER.parse(data)
    functions = {_function_name_from_definition(n, data): n for n in tree.root_node.named_children
                 if n.type == "function_definition"}
    root = functions.get(test.name)
    if root is None:
        return test.full_text
    target_calls = [n for n in _walk(root) if n.type == "call_expression"
                    and n.child_by_field_name("function") is not None
                    and _node_text(data, n.child_by_field_name("function")) == wrapper]
    # Keep the call's argument expressions for dependency discovery, but exclude
    # installs occurring only after it. Path feasibility is checked at runtime.
    end = min((n.end_byte for n in target_calls), default=root.end_byte)
    roots = { _node_text(data, n) for n in _walk(root)
              if n.type == "identifier" and n.end_byte <= end }
    roots.discard(test.name)
    referenced = set(roots)
    pending = list(roots)
    while pending:
        name = pending.pop()
        for dependency in (info.global_identifier_names.get(name, []) + info.function_identifier_names.get(name, [])):
            if dependency == test.name:
                continue
            if dependency not in referenced:
                referenced.add(dependency)
                pending.append(dependency)
    pieces = [data[root.start_byte:end].decode() + ";}"]
    other_tests = {t.name for t in info.test_functions if t.name != test.name and t.name not in roots}
    for node in tree.root_node.named_children:
        if node.type == "function_definition":
            name = _function_name_from_definition(node, data)
            if name != test.name and name in referenced and name not in other_tests:
                pieces.append(_node_text(data, node))
        elif node.type in {"declaration", "type_definition", "preproc_def", "preproc_function_def"}:
            owners = []
            for decl in node.children_by_field_name("declarator"):
                while decl.child_by_field_name("declarator") is not None:
                    decl = decl.child_by_field_name("declarator")
                if decl.type == "identifier":
                    owners.append(_node_text(data, decl))
            if set(owners) & referenced:
                pieces.append(_node_text(data, node))
    return "\n".join(pieces)
