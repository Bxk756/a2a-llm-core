import os
import time
from typing import Dict, Any

from a2a.policy_engine.loader import load_policy_with_packs
from a2a.policy_engine.engine import PolicyEngine


class PolicyManager:
    def __init__(self, config_path: str, packs_dir: str, poll_interval: float = 1.0):
        self.config_path = config_path
        self.packs_dir = packs_dir
        self.poll_interval = poll_interval

        self._last_mtime = 0.0
        self._engine: PolicyEngine | None = None
        self._last_check = 0.0

        self._reload_if_needed(force=True)

    def _collect_mtime(self) -> float:
        mtimes = []

        if os.path.exists(self.config_path):
            mtimes.append(os.path.getmtime(self.config_path))

        if os.path.isdir(self.packs_dir):
            for fname in os.listdir(self.packs_dir):
                if fname.endswith(".yaml"):
                    mtimes.append(
                        os.path.getmtime(os.path.join(self.packs_dir, fname))
                    )

        return max(mtimes) if mtimes else 0.0

    def _reload_if_needed(self, force: bool = False) -> None:
        now = time.time()

        if not force and now - self._last_check < self.poll_interval:
            return

        self._last_check = now
        current_mtime = self._collect_mtime()

        if force or current_mtime > self._last_mtime:
            policy = load_policy_with_packs(
                self.config_path,
                self.packs_dir,
            )
            self._engine = PolicyEngine(policy)
            self._last_mtime = current_mtime

    def get_engine(self) -> PolicyEngine:
        self._reload_if_needed()
        assert self._engine is not None
        return self._engine

    def status(self) -> Dict[str, Any]:
        return {
            "config_path": self.config_path,
            "packs_dir": self.packs_dir,
            "last_reload": self._last_mtime,
            "poll_interval": self.poll_interval,
        }
