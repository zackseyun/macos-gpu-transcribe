"""Opt in to keeping Nemotron resident across dictation app/worker restarts."""
import os
import plistlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nemotron_resident import AGENT, LABEL, ROOT, STATE

if __name__ == "__main__":
    assert (ROOT / ".mlx-next/models/nemotron/config.json").is_file(), "Install Nemotron first"
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    STATE.chmod(0o700)
    AGENT.parent.mkdir(parents=True, exist_ok=True)
    AGENT.write_bytes(plistlib.dumps(dict(Label=LABEL, RunAtLoad=True, KeepAlive=True,
        ProcessType="Interactive", ThrottleInterval=10, WorkingDirectory=str(ROOT),
        ProgramArguments=[str(ROOT / ".venv/bin/python"), str(ROOT / "nemotron_resident.py")],
        StandardOutPath=str(STATE / "service.log"), StandardErrorPath=str(STATE / "service.log"))))
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", domain + "/" + LABEL], capture_output=True)
    subprocess.run(["launchctl", "bootstrap", domain, str(AGENT)], check=True)
