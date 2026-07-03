import os
import re
from typing import Optional

# 修改makefile文件，添加obj-$(CONFIG_{config_name}) += {obj_name}
def ensure_makefile(makefile_path: str, config_name: str, obj_name: str):
    line = f"obj-$(CONFIG_{config_name}) += {obj_name}\n"

    with open(makefile_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if any(line.strip() == l.strip() for l in lines):
        return  # already exists

    with open(makefile_path, "a", encoding="utf-8") as f:
        f.write("\n" + line)

def find_driver_config_from_makefile(driver_c_path: str) -> Optional[str]:
    """
    Given path to a driver .c file, find the CONFIG_XXX
    that controls its compilation via Makefile.

    Returns:
        config name without 'CONFIG_' prefix, e.g. 'LEDS_LP3944'
        or None if not found.
    """
    driver_dir = os.path.dirname(driver_c_path)
    driver_c = os.path.basename(driver_c_path)
    driver_o = os.path.splitext(driver_c)[0] + ".o"

    makefile_path = os.path.join(driver_dir, "Makefile")
    if not os.path.exists(makefile_path):
        return None

    with open(makefile_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Match patterns like:
    # obj-$(CONFIG_LEDS_LP3944) += leds-lp3944.o
    pattern = re.compile(
        rf"obj-\$\(\s*CONFIG_([A-Z0-9_]+)\s*\)\s*\+=\s*.*\b{re.escape(driver_o)}\b"
    )

    for line in lines:
        m = pattern.search(line)
        if m:
            return m.group(1)

    return None


def config_exists_in_kconfig(driver_dir: str, config_name: str) -> bool:
    """
    Check whether 'config <name>' exists in Kconfig files
    under the driver directory.
    """
    for root, _, files in os.walk(driver_dir):
        for fn in files:
            if fn == "Kconfig":
                path = os.path.join(root, fn)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    if re.search(rf"^\s*config\s+{config_name}\b", f.read(), re.MULTILINE):
                        return True
    return False

def derive_driver_config(driver_c_path: str) -> str:
    """
    Derive the original driver CONFIG name for a driver .c file.

    Strategy:
    1. Parse Makefile to find obj-$(CONFIG_XXX) += driver.o
    2. Validate that CONFIG_XXX exists in Kconfig (best-effort)
    3. Raise error if not found (fail fast)
    """
    driver_dir = os.path.dirname(driver_c_path)

    config = find_driver_config_from_makefile(driver_c_path)
    if not config:
        raise RuntimeError(
            f"Failed to find driver CONFIG from Makefile for {driver_c_path}"
        )

    if not config_exists_in_kconfig(driver_dir, config):
        # Not fatal, but worth warning
        print(
            f"[WARN] CONFIG_{config} not found in Kconfig under {driver_dir}, "
            "but it is referenced in Makefile."
        )

    return config

def ensure_kunit_kconfig(kconfig_path: str, driver_c_path: str, config_name: str):

    driver_config = derive_driver_config(driver_c_path)
    block = f"""
            config {config_name}
                tristate "KUnit tests for {config_name.replace('_KUNIT_TEST','')}"
                depends on KUNIT && {driver_config}
            """

    with open(kconfig_path, "a", encoding="utf-8") as f:
        f.write("\n" + block)
    
    return driver_config
