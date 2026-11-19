"""Explicit same-function value paths from private fields to boundary arguments."""

from validation.test_inspector import C_PARSER, _walk
from verification.observation_provenance import _path, _reads


def expression_paths(expression):
    data = ('void f(void){' + expression + ';}').encode()
    tree = C_PARSER.parse(data)
    statements = [node for node in _walk(tree.root_node) if node.type == 'expression_statement']
    if not statements or not statements[0].named_children:
        return set()
    node = statements[0].named_children[0]
    if any(item.type in {'call_expression', 'assignment_expression', 'update_expression'} for item in _walk(node)):
        return set()
    return _reads(node, data)


def field_argument_projections(context):
    """Do not infer flow from field-name suffixes or the graph's broad relations."""
    registry = context.get('scenario_registry', context)
    facts = registry.get('source_facts', [])
    by_id = {fact['fact_id']: fact for fact in facts}
    checks = [check for contract in registry.get('scenario_contracts', [])
              for check in contract.get('scenario_checks', [])
              if check.get('kind', '').startswith('Field')]
    if not checks:
        return {}

    def depends(expression, target, function, order, seen):
        for path in expression_paths(expression):
            if path == target:
                return True
            key = (path, order)
            if key in seen:
                continue
            preceding = [fact for fact in facts if fact.get('function') == function
                         and fact.get('kind') == 'ASSIGNMENT'
                         and fact.get('metadata', {}).get('source_order', order) < order
                         and path in expression_paths(fact.get('metadata', {}).get('left', ''))]
            if not preceding:
                continue
            last = max(preceding, key=lambda fact: fact['metadata']['source_order'])
            meta = last['metadata']
            # A conditional reaching definition needs a CFG merge; do not guess
            # that a lexical write dominated the call on every feasible path.
            if meta.get('control_context'):
                continue
            raw = ('void f(void){' + last.get('code', '') + ';}').encode()
            assignments = [n for n in _walk(C_PARSER.parse(raw).root_node) if n.type == 'assignment_expression']
            operator = assignments[0].child_by_field_name('operator') if assignments else None
            if operator is None or raw[operator.start_byte:operator.end_byte] != b'=':
                continue
            if depends(meta.get('right', ''), target, function, meta['source_order'], seen | {key}):
                return True
        return False

    projections = {}
    for boundary in registry.get('boundary_candidates', []):
        if boundary.get('source_fact_kind') != 'CALL':
            continue
        for fact_id in boundary.get('fact_ids', []):
            call = by_id.get(fact_id, {})
            if call.get('kind') != 'CALL':
                continue
            meta = call.get('metadata', {})
            if not isinstance(meta.get('source_order'), int):
                continue
            for check in checks:
                origins = {by_id[fact_id].get('function') for fact_id in check.get('evidence_ids', []) if fact_id in by_id}
                origins = origins or {registry.get('target_function')}
                if call.get('function') not in origins:
                    continue
                target_paths = expression_paths(check.get('target', ''))
                if len(target_paths) != 1:
                    continue
                target = next(iter(target_paths))
                for index, argument in enumerate(meta.get('arguments', [])):
                    if depends(argument, target, call.get('function'), meta['source_order'], set()):
                        projections.setdefault(check['target'], set()).add((boundary['candidate_id'], index))
    return projections
