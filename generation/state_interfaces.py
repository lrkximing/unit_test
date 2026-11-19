"""Typed accessors for mutable scalar inputs referenced by the target closure."""
from tree_sitter_languages import get_parser


def scalar_inputs(parsed, target_name):
    functions = {f.name: f for f in parsed.functions}
    pending, seen, referenced = [target_name], set(), set()
    while pending:
        name = pending.pop()
        if name in seen or name not in functions:
            continue
        seen.add(name)
        referenced.update(functions[name].global_refs)
        pending.extend(parsed.call_graph.get(name, []))
    referenced &= set(parsed.file_level_defs.get('globals', []))
    raw = parsed.source_code.encode()
    tree = get_parser('c').parse(raw)
    found = {}

    def declarations(node):
        for child in node.named_children:
            if child.type == 'declaration':
                yield child
            elif child.type.startswith('preproc_'):
                yield from declarations(child)

    for declaration in declarations(tree.root_node):
        type_node = declaration.child_by_field_name('type')
        if type_node is None or type_node.type in {'struct_specifier', 'union_specifier', 'enum_specifier'}:
            continue
        if any(raw[n.start_byte:n.end_byte].decode() == 'const' for n in declaration.named_children if n.type == 'type_qualifier'):
            continue
        type_text = raw[type_node.start_byte:type_node.end_byte].decode()
        for item in declaration.children_by_field_name('declarator'):
            declarator = item.child_by_field_name('declarator') if item.type == 'init_declarator' else item
            if declarator is None or declarator.type != 'identifier':
                continue
            name = raw[declarator.start_byte:declarator.end_byte].decode()
            if name in referenced:
                found.setdefault(name, set()).add(type_text)
    return {name: next(iter(types)) for name, types in found.items() if len(types) == 1}


def state_export_block(parsed, target_name, export_gpl=True):
    from generation.generate_stub import _build_export_wrapper
    blocks, interfaces = [], []
    for name, type_text in sorted(scalar_inputs(parsed, target_name).items()):
        for operation in ('get', 'set'):
            symbol = f'raca_state_{target_name}_{name}_{operation}'
            block, interface = _build_export_wrapper(
                export_func_name=symbol, return_type=type_text if operation == 'get' else 'void',
                param_list_text='(void)' if operation == 'get' else f'({type_text} value)',
                body_lines=[f'    return {name};\n'] if operation == 'get' else [f'    {name} = value;\n'],
                export_gpl=export_gpl, source_symbol=name, source_kind='global_state',
                description=f'{operation} mutable production input {name}; save its value and restore it after the test',
                prototype_comment=f'/* Original mutable input: {name} */')
            blocks.append(block)
            interfaces.append(interface)
    return '\n'.join(blocks), interfaces
