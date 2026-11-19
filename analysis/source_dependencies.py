"""Read-only source evidence, separate from definitions copied into test units."""

import hashlib
from pathlib import Path
import re

from tree_sitter_languages import get_parser
from analysis.header_context import defined_macro_names, include_search_roots
from analysis.local_dependencies import identifiers, walk


def _declarator_name(node, data):
    while node is not None and node.type != 'identifier':
        node = node.child_by_field_name('declarator')
    return data[node.start_byte:node.end_byte].decode() if node is not None else ''


def _source_references(code, macro_names=()):
    data = code.encode()
    tree = get_parser('c').parse(data)
    function = next(n for n in tree.root_node.named_children if n.type == 'function_definition')
    declarator = function.child_by_field_name('declarator')
    while declarator is not None and declarator.type != 'function_declarator':
        declarator = declarator.child_by_field_name('declarator')
    parameters = declarator.child_by_field_name('parameters') if declarator is not None else None
    locals_ = []
    for parameter in parameters.named_children if parameters is not None else []:
        if parameter.type == 'parameter_declaration':
            locals_.append((_declarator_name(parameter.child_by_field_name('declarator'), data), function, function.start_byte))
    for node in walk(function):
        if node.type != 'declaration':
            continue
        scope = node.parent
        while scope is not None and scope.type not in {'compound_statement', 'for_statement', 'function_definition'}:
            scope = scope.parent
        for declaration in node.children_by_field_name('declarator'):
            locals_.append((_declarator_name(declaration, data), scope, node.start_byte))
    name = _declarator_name(function.child_by_field_name('declarator'), data)
    references = set()
    for node in walk(function):
        if node.type != 'identifier':
            continue
        value = data[node.start_byte:node.end_byte].decode()
        if value == name or any(value == local and scope is not None and start <= node.start_byte
                                and scope.start_byte <= node.start_byte < scope.end_byte
                                for local, scope, start in locals_):
            continue
        if (node.parent.type == 'call_expression' and node == node.parent.child_by_field_name('function')
                and value not in macro_names):
            continue
        references.add(value)
    return references


def collect_source_dependencies(parsed, function):
    """Follow exact in-file calls/data and referenced literal-header constants."""
    driver = Path(parsed.path).resolve()
    functions = {item.name: item for item in parsed.functions}
    closure, pending = set(), [function.name]
    while pending:
        name = pending.pop()
        if name in closure or name not in functions:
            continue
        closure.add(name)
        pending.extend(parsed.call_graph.get(name, []))
    units, owners = [], {}

    def file_scope_units(node):
        for child in node.named_children:
            if child.type in {'preproc_def', 'preproc_function_def', 'declaration', 'enum_specifier'}:
                yield child
            elif child.type.startswith('preproc_'):
                yield from file_scope_units(child)

    def index(path, headers=False):
        data = parsed.source_code.encode() if path == driver and getattr(parsed, 'source_code', '') else path.read_bytes()
        tree = get_parser('c').parse(data)
        for item in file_scope_units(tree.root_node):
            names, kind = set(), ''
            if item.type in {'preproc_def', 'preproc_function_def'}:
                name = item.child_by_field_name('name')
                if name is not None:
                    names.add(data[name.start_byte:name.end_byte].decode())
                kind = 'header_constant' if headers else 'macro'
            elif item.type in {'declaration', 'enum_specifier'}:
                enums = [n for n in walk(item) if n.type == 'enum_specifier' and n.child_by_field_name('body')]
                for enum in enums:
                    names.update(data[n.child_by_field_name('name').start_byte:n.child_by_field_name('name').end_byte].decode()
                                 for n in walk(enum) if n.type == 'enumerator')
                names.update(_declarator_name(d, data) for d in item.children_by_field_name('declarator'))
                kind = 'header_constant' if headers else 'global_declaration'
            if not names:
                continue
            text = data[item.start_byte:item.end_byte].decode()
            dependencies = identifiers(text) - names
            parameters = item.child_by_field_name('parameters') if item.type == 'preproc_function_def' else None
            if parameters is not None:
                dependencies -= identifiers(data[parameters.start_byte:parameters.end_byte].decode())
            unit = {'kind': kind, 'path': str(path), 'line': item.start_point[0] + 1,
                    'symbols': sorted(names), 'code': text, 'dependency_symbols': sorted(dependencies),
                    'sha256': hashlib.sha256(text.encode()).hexdigest()}
            units.append(unit)
            for name in names:
                owners.setdefault(name, len(units) - 1)

    index(driver)
    kernel = next((p for p in driver.parents if (p / 'include').is_dir()), None)
    search_roots = include_search_roots(driver)
    pending_paths, visited, headers, macro_names = [driver], set(), [], set()
    while pending_paths:
        path = pending_paths.pop()
        if path in visited:
            continue
        visited.add(path)
        text = parsed.source_code if path == driver and getattr(parsed, 'source_code', '') else path.read_text(errors='replace')
        macro_names.update(defined_macro_names(text))
        if path != driver:
            headers.append((path, set(re.findall(r'\b[A-Za-z_]\w*\b', text))))
        for opening, include in re.findall(r'^\s*#\s*include\s*([<"])([^>"\n]+)[>"]', text, re.M):
            for root in ([path.parent] if opening == '"' else []) + search_roots:
                candidate = (root / include).resolve()
                if candidate.is_file():
                    pending_paths.append(candidate)
                    break

    roots = set().union(*(_source_references(functions[name].code, macro_names) for name in closure))
    indexed_headers = set()
    while True:
        needed, visiting = set(roots), list(roots)
        while visiting:
            owner = owners.get(visiting.pop())
            if owner is None:
                continue
            for symbol in set(units[owner]['dependency_symbols']) - needed:
                needed.add(symbol)
                visiting.append(symbol)
        unresolved = needed - owners.keys()
        candidates = [(path, names) for path, names in headers
                      if path not in indexed_headers and unresolved.intersection(names)]
        if not candidates:
            break
        for path, _ in candidates:
            index(path, headers=True)
            indexed_headers.add(path)

    selected, pending = set(), list(roots)
    while pending:
        name = pending.pop()
        index_ = owners.get(name)
        if index_ is None or index_ in selected:
            continue
        selected.add(index_)
        pending.extend(units[index_]['dependency_symbols'])
    records = [units[i] for i in sorted(selected)]
    records.extend({'kind': 'local_software_dependency', 'path': str(driver),
                    'symbols': [name], 'code': functions[name].code,
                    'sha256': hashlib.sha256(functions[name].code.encode()).hexdigest()}
                   for name in sorted(closure - {function.name}))
    constraints = {}
    if kernel and (kernel / '.config').is_file():
        frame = re.search(r'^CONFIG_FRAME_WARN=(\d+)$', (kernel / '.config').read_text(), re.M)
        if frame:
            constraints['kernel_compile_frame_warning_bytes'] = int(frame[1])
    # Accessor bodies are verifier evidence, not new prompt content or test definitions.
    from verification.object_relations import accessor_definitions
    models, model_sources, scanned = {}, {}, set()
    needed = {callee for name in closure for callee in parsed.call_graph.get(name, [])}
    while True:
        candidates = [(path, names) for path, names in headers if path not in scanned and needed.intersection(names)]
        if not candidates:
            break
        next_needed = set()
        for path, _ in candidates:
            scanned.add(path)
            text = path.read_text(errors='replace')
            found = accessor_definitions(text)
            for name, model in found.items():
                if name in models and models[name] != model:
                    models[name] = None  # Ambiguous preprocessor alternatives are not proof.
                elif name not in models:
                    models[name] = model
                model_sources[name] = {'path': str(path), 'sha256': hashlib.sha256(text.encode()).hexdigest()}
            for name in needed:
                if found.get(name):
                    next_needed.update(re.findall(r'\b([A-Za-z_]\w*)\s*\(', found[name]['expression']))
        needed.update(next_needed)
    for record in records:
        data = record['code'].encode()
        tree = get_parser('c').parse(data)
        for item in walk(tree.root_node):
            if item.type != 'preproc_function_def':
                continue
            name, params, value = (item.child_by_field_name(k) for k in ('name', 'parameters', 'value'))
            if name is None or params is None or value is None:
                continue
            expression = data[value.start_byte:value.end_byte].decode().strip()
            # Only a single expression macro can act as a field/layout projection.
            if ';' in expression or '{' in expression or '#' in expression:
                continue
            key = data[name.start_byte:name.end_byte].decode()
            models[key] = {'parameters': [data[p.start_byte:p.end_byte].decode() for p in params.named_children],
                           'returns': True, 'expression': expression}
            model_sources[key] = {'path': record['path'], 'sha256': record['sha256']}
    return {'policy': 'Read-only production source evidence. Do not redefine these private globals or software helpers in tests.',
            'records': records, 'kernel_compile_constraints': constraints,
            'observation': {'source': function.code, 'accessors': {k: v for k, v in models.items() if v},
                            'provenance': model_sources}}
