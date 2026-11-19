"""Source-backed, straight-line accessor relations, never inferred from API names."""
from validation.test_inspector import C_PARSER, _node_text, _walk, _function_name_from_definition


def parameters(fn, data):
    node = fn.child_by_field_name('declarator')
    while node is not None and node.type != 'function_declarator':
        node = node.child_by_field_name('declarator')
    args = node.child_by_field_name('parameters') if node is not None else None
    names = []
    for arg in args.named_children if args is not None else []:
        if arg.type != 'parameter_declaration':
            continue
        name = arg.child_by_field_name('declarator')
        while name is not None and name.type != 'identifier':
            name = name.child_by_field_name('declarator')
        names.append(_node_text(data, name) if name is not None else '')
    return names


def accessor_definitions(source):
    data = source.encode()
    result = {}
    for fn in _walk(C_PARSER.parse(data).root_node):
        if fn.type != 'function_definition':
            continue
        body = fn.child_by_field_name('body')
        statements = [n for n in body.named_children if n.type != 'comment']
        if len(statements) != 1 or statements[0].type not in {'return_statement', 'expression_statement'}:
            continue
        expression = next(iter(statements[0].named_children), None)
        if expression is None or any(n.type in {'conditional_expression', 'update_expression'} for n in _walk(expression)):
            continue
        name = _function_name_from_definition(fn, data)
        result[name] = {'parameters': parameters(fn, data),
                        'returns': statements[0].type == 'return_statement',
                        'expression': _node_text(data, expression)}
    return result


def resolve_path(path, aliases):
    for _ in range(len(aliases) + 1):
        key = next((k for k in sorted(aliases, key=len, reverse=True)
                    if path == k or path.startswith(k + '.') or path.startswith(k + '[')), None)
        if key is None:
            return path
        updated = aliases[key] + path[len(key):]
        if updated == path:
            return path
        path = updated
    return ''  # A cyclic relation is not evidence.


def expression_path(expression, aliases, models, *, stores=None, stack=()):
    """Resolve field/getter expressions using actual production helper bodies."""
    from verification.observation_provenance import _path
    data = ('void f(void){' + expression + ';}').encode()
    statement = next((n for n in _walk(C_PARSER.parse(data).root_node) if n.type == 'expression_statement'), None)
    if statement is None or not statement.named_children:
        return ''

    def evaluate(node):
        if node.type in {'cast_expression', 'parenthesized_expression', 'pointer_expression'}:
            child = node.child_by_field_name('value') or node.child_by_field_name('argument')
            return evaluate(child or node.named_children[0])
        if node.type == 'call_expression':
            name = _node_text(data, node.child_by_field_name('function'))
            args = node.child_by_field_name('arguments').named_children
            # container_of is a kernel layout operation, not an object-data API heuristic.
            if name == 'container_of':
                from verification.assertion_quality import _split_call_args
                raw = _split_call_args(_node_text(data, node))
                if len(raw) != 3:
                    return ''
                base = expression_path(raw[0], aliases, models, stores=stores, stack=stack)
                member = raw[2].strip()
                return base[:-len(member) - 1] if base.endswith('.' + member) else ''
            model = models.get(name)
            if not model or not model['returns'] or name in stack:
                return ''
            actual = [evaluate(arg) for arg in args]
            mapping = dict(zip(model['parameters'], actual))
            if not all(mapping.values()):
                return ''
            return expression_path(model['expression'], mapping, models, stores=stores, stack=(*stack, name))
        if node.type == 'field_expression':
            base = evaluate(node.child_by_field_name('argument'))
            path = base + '.' + _node_text(data, node.child_by_field_name('field')) if base else ''
        else:
            path = _path(node, data)
        path = resolve_path(path, aliases)
        return resolve_path(path, stores or {}) if path else ''

    return evaluate(statement.named_children[0])


def accessor_write(name, arguments, models, stores, stack=()):
    """Return a concrete data-slot write only for a proven straight-line setter."""
    model = models.get(name)
    if not model or model['returns'] or name in stack:
        return None
    mapping = dict(zip(model['parameters'], arguments))
    if not all(mapping.values()):
        return None
    data = ('void f(void){' + model['expression'] + ';}').encode()
    expression = next(n for n in _walk(C_PARSER.parse(data).root_node) if n.type == 'expression_statement').named_children[0]
    if expression.type == 'assignment_expression':
        if _node_text(data, expression.child_by_field_name('operator')) != '=':
            return None
        left = expression_path(_node_text(data, expression.child_by_field_name('left')), mapping, models)
        right = expression_path(_node_text(data, expression.child_by_field_name('right')), mapping, models, stores=stores)
        return (left, right) if left and right else None
    if expression.type == 'call_expression':
        callee = _node_text(data, expression.child_by_field_name('function'))
        actual = [expression_path(_node_text(data, arg), mapping, models, stores=stores)
                  for arg in expression.child_by_field_name('arguments').named_children]
        return accessor_write(callee, actual, models, stores, (*stack, name))
    return None
