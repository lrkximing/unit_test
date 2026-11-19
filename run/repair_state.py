"""Continue an unresolved repair without evaluating an unchanged artifact again."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


def source_build_failure(log):
    """Only compiler source diagnostics can outlive a rejected source proposal."""
    infrastructure = r'No space left|Cannot allocate memory|Killed signal|Input/output error|Permission denied|Read-only file system|No such file or directory|Connection (?:reset|timed out)|No rule to make target'
    if re.search(infrastructure, log, re.IGNORECASE):
        return False
    return bool(re.search(r'^.+:\d+(?::\d+)?:\s+(?:fatal )?error:', log, re.MULTILINE))


def input_identity(source, context, frozen_tests, paths):
    files = {}
    for name in paths:
        path = Path(name)
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    payload = {'source': source, 'context': context, 'frozen_tests': sorted(frozen_tests), 'files': files}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class RepairState:
    def __init__(self):
        self.rejections = []
        self.pending = None

    @staticmethod
    def _key(stage, source):
        stage = re.sub(r'_retry\d+$', '', stage)
        return stage, hashlib.sha256(source.encode()).hexdigest()

    def reject(self, stage, source, rejection):
        stage, checksum = self._key(stage, source)
        self.rejections.append({'stage': stage, 'source_sha256': checksum,
            'iteration': rejection.get('iteration'), 'phase': rejection.get('phase'),
            'reason': rejection.get('reason'), 'hard_errors': deepcopy(rejection.get('hard_errors', [])),
            'candidate': rejection.get('candidate'), 'audit_diff': rejection.get('audit_diff')})

    def feedback(self, stage, source, problem):
        stage, checksum = self._key(stage, source)
        rejected = [item for item in self.rejections
                    if item['stage'] == stage and item['source_sha256'] == checksum]
        if not rejected:
            return problem
        return '\n\n'.join([problem, 'Prior rejected proposals for this unchanged accepted source '
            '(these constraints still apply; this is not a new test failure):\n' +
            json.dumps(rejected, ensure_ascii=False, indent=2)])

    def defer(self, identity, payload, iteration):
        self.pending = {'identity': identity, 'payload': deepcopy(payload), 'source_iteration': iteration}

    def continuation(self, identity):
        if self.pending and self.pending['identity'] == identity:
            return deepcopy(self.pending)
        self.pending = None
        return None

    def clear(self):
        self.pending = None
