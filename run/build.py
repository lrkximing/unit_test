import subprocess
import os
import re
import signal
import time
from typing import List, Set, Dict, Optional


class QemuRunError(RuntimeError):
    def __init__(self, log: str):
        super().__init__("QEMU execution failed")
        self.log = log


_KCONFIG_INDEX: Dict[str, str] = {}
_KCONFIG_INDEX_ROOT: str = ""
_KCONFIG_FILE_CACHE: Dict[str, List[str]] = {}
_KCONFIG_DEP_CACHE: Dict[str, Set[str]] = {}


def enable_kernel_config_buildroot(linux_dir: str, config: str, value: str = "y"):
    """
    Enable a kernel config in Buildroot-managed Linux kernel tree.
    """
    script = os.path.join(linux_dir, "scripts", "config")
    if not os.path.exists(script):
        raise RuntimeError(f"scripts/config not found in {linux_dir}")

    subprocess.run(
        [script, "--set-val", config, value],
        cwd=linux_dir,
        check=True,
        capture_output=True,
        text=True,
    )


def rebuild_kernel_buildroot(buildroot_dir: str):
    subprocess.run(
        ["make", "linux-rebuild"],
        cwd=buildroot_dir,
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        ["make"],
        cwd=buildroot_dir,
        check=True,
        capture_output=True,
        text=True,
    )

def enable_driver_and_kunit_test(
    buildroot_dir: str,
    linux_dir: str,
    driver_config: str,
    driver_kunit_config: str,
    driver_kconfig_path: str,
):
    # 1. enable driver
    configs_to_enable = [driver_config]
    deps = collect_kconfig_dependencies(linux_dir, driver_config, driver_kconfig_path)
    configs_to_enable.extend(sorted(deps))
    configs_to_enable.append(driver_kunit_config)
    seen = set()
    # print(f"enable driver and kunit test: {configs_to_enable}")
    for cfg in configs_to_enable:
        if cfg in seen:
            continue
        seen.add(cfg)
        enable_kernel_config_buildroot(linux_dir, cfg, "y")

    # 3. regenerate config
    subprocess.run(
        ["make", "olddefconfig"],
        cwd=linux_dir,
        check=True,
        capture_output=True,
        text=True,
    )

    # 4. rebuild kernel via buildroot
    rebuild_kernel_buildroot(buildroot_dir)


def _register_kconfig_file(path: str) -> None:
    global _KCONFIG_INDEX, _KCONFIG_FILE_CACHE
    if not path or not os.path.exists(path):
        return
    if path in _KCONFIG_FILE_CACHE:
        return
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except OSError:
        _KCONFIG_FILE_CACHE[path] = []
        return
    _KCONFIG_FILE_CACHE[path] = content.splitlines()
    for match in re.finditer(r"^(?:config|menuconfig)\s+([A-Z0-9_]+)\b", content, re.MULTILINE):
        symbol = match.group(1)
        _KCONFIG_INDEX.setdefault(symbol, path)


def _ensure_kconfig_index(linux_dir: str, extra_path: str = "") -> None:
    global _KCONFIG_INDEX, _KCONFIG_INDEX_ROOT, _KCONFIG_FILE_CACHE, _KCONFIG_DEP_CACHE
    if not (_KCONFIG_INDEX and _KCONFIG_INDEX_ROOT == linux_dir):
        _KCONFIG_INDEX = {}
        _KCONFIG_FILE_CACHE = {}
        _KCONFIG_DEP_CACHE = {}
        _KCONFIG_INDEX_ROOT = linux_dir
        for root, _, files in os.walk(linux_dir):
            for fn in files:
                if fn != "Kconfig":
                    continue
                _register_kconfig_file(os.path.join(root, fn))
    if extra_path:
        _register_kconfig_file(extra_path)


def _get_kconfig_block_lines(linux_dir: str, symbol: str, hint_path: str = "") -> List[str]:
    _ensure_kconfig_index(linux_dir, hint_path)
    path = _KCONFIG_INDEX.get(symbol)
    if not path:
        return []
    if path not in _KCONFIG_FILE_CACHE:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                _KCONFIG_FILE_CACHE[path] = f.read().splitlines()
        except OSError:
            _KCONFIG_FILE_CACHE[path] = []
    lines = _KCONFIG_FILE_CACHE.get(path, [])
    if not lines:
        return []
    start_idx = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("config ") or stripped.startswith("menuconfig "):
            if stripped.split()[1] == symbol:
                start_idx = idx
                break
    if start_idx is None:
        return []
    block: List[str] = []
    for line in lines[start_idx + 1 :]:
        if line and not line[0].isspace():
            break
        block.append(line)
    return block


def _extract_dependency_symbols(expr: str) -> Set[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+", expr)
    ignored = {"n", "y", "m", "N", "Y", "M"}
    return {tok for tok in tokens if tok and not tok.isdigit() and tok not in ignored}


def _parse_dependencies_from_block(block_lines: List[str]) -> Set[str]:
    deps: Set[str] = set()
    i = 0
    while i < len(block_lines):
        line = block_lines[i].strip()
        if line.startswith("depends on"):
            expr = line[len("depends on") :].strip()
            while expr.endswith("\\") and i + 1 < len(block_lines):
                i += 1
                expr = expr[:-1].strip() + " " + block_lines[i].strip()
            deps.update(_extract_dependency_symbols(expr))
        i += 1
    return deps


def _get_symbol_dependencies(linux_dir: str, symbol: str, hint_path: str = "") -> Set[str]:
    if symbol in _KCONFIG_DEP_CACHE:
        return _KCONFIG_DEP_CACHE[symbol]
    block = _get_kconfig_block_lines(linux_dir, symbol, hint_path)
    deps = _parse_dependencies_from_block(block) if block else set()
    _KCONFIG_DEP_CACHE[symbol] = deps
    return deps


def collect_kconfig_dependencies(linux_dir: str, symbol: str, hint_path: str = "") -> Set[str]:
    resolved: Set[str] = set()
    stack = list(_get_symbol_dependencies(linux_dir, symbol, hint_path))
    visited = {symbol}
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        resolved.add(current)
        stack.extend(_get_symbol_dependencies(linux_dir, current, hint_path))
    resolved.discard(symbol)
    return resolved


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def cleanup_stale_kunit_qemu_processes(grace_seconds: float = 2.0) -> List[int]:
    """Stop leftover same-user KUnit QEMU instances before launching a new one."""
    current_pid = os.getpid()
    try:
        result = subprocess.run(
            ["pgrep", "-u", str(os.getuid()), "-f", "qemu-system.*kunit.filter_glob="],
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, AttributeError):
        return []
    pids: List[int] = []
    for line in (result.stdout or "").splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid and pid != current_pid:
            pids.append(pid)
    if not pids:
        return []

    terminated: List[int] = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            terminated.append(pid)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue

    if grace_seconds > 0 and terminated:
        time.sleep(grace_seconds)
    for pid in terminated:
        if not _pid_alive(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            continue
    return terminated

def run_qemu_direct(
    buildroot_dir: str,
    log_path: str,
    commands: List[str],
    extra_boot_args: Optional[List[str]] = None,
) -> str:
    host_share = os.path.join(buildroot_dir, "qemu-share")
    cleaned_qemu_pids = cleanup_stale_kunit_qemu_processes()
    append_args = ["root=/dev/sda", "console=ttyS0", "kunit.enable_late_initcall=1", "panic=-1"]
    append_args.extend(arg for arg in (extra_boot_args or []) if arg)
    cmd = [
        "qemu-system-x86_64",
        "-kernel", "output/images/bzImage",
        "-append", " ".join(append_args),
        "-hda", "output/images/rootfs.ext2",
        "-fsdev", f"local,id=fsdev0,path={host_share},security_model=none",
        "-device", "virtio-9p-pci,fsdev=fsdev0,mount_tag=hostshare",
        "-nographic",
    ]

    cmd_queue = list(commands or [])
    if not cmd_queue or cmd_queue[-1].strip().lower() != "poweroff":
        cmd_queue.append("poweroff")

    output_lines: List[str] = []
    buffer = ""
    login_sent = False
    with open(log_path, "w") as f:
        if cleaned_qemu_pids:
            f.write(f"[RACA] cleaned stale KUnit QEMU processes before run: {cleaned_qemu_pids}\n")
            f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=buildroot_dir,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if proc.stdout is None:
            raise RuntimeError("Failed to capture QEMU output.")

        while True:
            chunk = proc.stdout.read(1)
            if chunk == "" and proc.poll() is not None:
                break
            if chunk == "":
                continue
            f.write(chunk)
            f.flush()
            output_lines.append(chunk)
            buffer += chunk
            if len(buffer) > 512:
                buffer = buffer[-512:]

            if not login_sent and "login:" in buffer:
                if proc.stdin is None:
                    raise RuntimeError("QEMU stdin not available for automation.")
                proc.stdin.write("root\n")
                proc.stdin.flush()
                login_sent = True
                buffer = ""
                continue

            if login_sent and cmd_queue and buffer.endswith("# "):
                next_cmd = cmd_queue.pop(0)
                proc.stdin.write(next_cmd + "\n")
                proc.stdin.flush()
                buffer = ""

        proc.wait()

    combined_output = "".join(output_lines)
    if proc.returncode != 0:
        raise QemuRunError(combined_output)

    return combined_output
