import json
import subprocess
from pathlib import Path

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class CodexClient:
    """
    Stateless Codex CLI client.

    Each generate() call:
    - starts a fresh Codex thread
    - does not resume previous conversations
    - does not read Codex memories
    - does not generate Codex memories
    - does not persist the session rollout
    - runs in read-only sandbox
    Usage statistics are accumulated locally in this Python object.
    """

    def __init__(
        self,
        workdir: Union[str, Path],
        codex_bin: str = "codex",
        model: Optional[str] = None,
        timeout: Optional[int] = None,

        # Isolation settings
        use_memories: bool = False,
        generate_memories: bool = False,
        ephemeral: bool = True,
        ignore_user_config: bool = True,
        ignore_rules: bool = True,

        # Codex sandbox
        sandbox: str = "read-only",
    ):
        self.workdir = Path(workdir).resolve()

        if not self.workdir.exists():
            raise FileNotFoundError(
                f"workdir does not exist: {self.workdir}"
            )

        self.codex_bin = codex_bin
        self.model = model
        self.timeout = timeout

        self.use_memories = use_memories
        self.generate_memories = generate_memories

        self.ephemeral = ephemeral
        self.ignore_user_config = ignore_user_config
        self.ignore_rules = ignore_rules

        self.sandbox = sandbox

        self.reset_usage()

    # ============================================================
    # Prompt construction
    # ============================================================

    @staticmethod
    def _messages_to_prompt(
        messages: List[Dict[str, str]]
    ) -> str:
        """
        Convert system/user messages into a single Codex input prompt.

        Example:

        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]

        becomes:

        [SYSTEM]
        ...

        [USER]
        ...
        """

        if not isinstance(messages, list) or not messages:
            raise ValueError(
                "messages must be a non-empty list"
            )

        parts = []

        for message in messages:

            if not isinstance(message, dict):
                raise TypeError(
                    "Each message must be a dict."
                )

            role = str(
                message.get("role", "user")
            ).strip().lower()

            content = message.get("content", "")

            if not isinstance(content, str):
                raise TypeError(
                    "message content must be a string"
                )

            content = content.strip()

            if not content:
                continue

            parts.append(
                f"[{role.upper()}]\n{content}"
            )

        if not parts:
            raise ValueError(
                "messages cannot be empty"
            )

        return "\n\n".join(parts)

    # ============================================================
    # CLI command
    # ============================================================

    def _build_cmd(self) -> List[str]:
        """
        Every call creates a completely new Codex exec session.
        No resume is used.
        """

        cmd = [
            self.codex_bin,

            # Do not inject historical Codex memories
            "-c",
            (
                "memories.use_memories="
                f"{str(self.use_memories).lower()}"
            ),

            # Do not use this session to create memories
            "-c",
            (
                "memories.generate_memories="
                f"{str(self.generate_memories).lower()}"
            ),

            "exec",

            "--json",
        ]

        # Do not persist rollout/session files
        if self.ephemeral:
            cmd.append("--ephemeral")

        # Avoid personal Codex configuration influencing the run
        if self.ignore_user_config:
            cmd.append("--ignore-user-config")

        # Avoid user/project execpolicy rules influencing the run
        if self.ignore_rules:
            cmd.append("--ignore-rules")

        # Sandbox
        if self.sandbox:
            cmd.extend([
                "--sandbox",
                self.sandbox,
            ])

        # Explicit model
        if self.model:
            cmd.extend([
                "--model",
                self.model,
            ])

        # Read complete prompt from stdin
        cmd.append("-")

        return cmd

    # ============================================================
    # Generation
    # ============================================================

    def generate(
        self,
        messages: List[Dict[str, str]],
    ) -> str:

        prompt = self._messages_to_prompt(
            messages
        )

        cmd = self._build_cmd()

        proc = subprocess.Popen(
            cmd,
            cwd=self.workdir,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout, stderr = proc.communicate(
                input=prompt,
                timeout=self.timeout,
            )

        except subprocess.TimeoutExpired:

            proc.kill()

            stdout, stderr = proc.communicate()

            raise TimeoutError(
                f"Codex timed out after "
                f"{self.timeout} seconds\n"
                f"stdout:\n{stdout[-2000:]}\n"
                f"stderr:\n{stderr[-2000:]}"
            )

        if proc.returncode != 0:
            raise RuntimeError(
                f"Codex failed with return code "
                f"{proc.returncode}\n"
                f"stderr:\n{stderr}\n"
                f"stdout:\n{stdout[-2000:]}"
            )

        final_messages = []

        usage_item = {
            "input_tokens": 0,
            "uncached_input_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
        }

        # --------------------------------------------------------
        # Parse Codex JSONL
        # --------------------------------------------------------

        for raw_line in stdout.splitlines():

            line = raw_line.strip()

            if not line:
                continue

            try:
                event = json.loads(line)

            except json.JSONDecodeError:
                continue

            event_type = event.get("type")

            # ----------------------------------------------------
            # Agent response
            # ----------------------------------------------------

            if event_type == "item.completed":

                item = event.get("item", {})

                if item.get("type") == "agent_message":

                    text = item.get("text")

                    if (
                        isinstance(text, str)
                        and text.strip()
                    ):
                        final_messages.append(
                            text.strip()
                        )

            # ----------------------------------------------------
            # Usage
            # ----------------------------------------------------

            elif event_type == "turn.completed":

                usage = event.get(
                    "usage",
                    {},
                ) or {}

                input_tokens = (
                    usage.get(
                        "input_tokens",
                        0,
                    )
                    or 0
                )

                cache_read_tokens = (
                    usage.get(
                        "cached_input_tokens",
                        0,
                    )
                    or 0
                )

                cache_write_tokens = (
                    usage.get(
                        "cache_write_input_tokens",
                        0,
                    )
                    or 0
                )

                output_tokens = (
                    usage.get(
                        "output_tokens",
                        0,
                    )
                    or 0
                )

                reasoning_output_tokens = (
                    usage.get(
                        "reasoning_output_tokens",
                        0,
                    )
                    or 0
                )

                uncached_input_tokens = max(
                    0,
                    input_tokens
                    - cache_read_tokens
                    - cache_write_tokens,
                )

                usage_item = {
                    "input_tokens":
                        input_tokens,

                    "uncached_input_tokens":
                        uncached_input_tokens,

                    "cache_read_tokens":
                        cache_read_tokens,

                    "cache_write_tokens":
                        cache_write_tokens,

                    "output_tokens":
                        output_tokens,

                    "reasoning_output_tokens":
                        reasoning_output_tokens,
                }

        if not final_messages:
            raise ValueError(
                "No agent_message found.\n"
                f"stderr:\n{stderr}\n"
                f"stdout:\n{stdout[-3000:]}"
            )

        # --------------------------------------------------------
        # Store per-call usage
        # --------------------------------------------------------

        self.usage_history.append(
            usage_item
        )

        # --------------------------------------------------------
        # Accumulate usage
        # --------------------------------------------------------

        self.total_input_tokens += (
            usage_item["input_tokens"]
        )

        self.total_uncached_input_tokens += (
            usage_item[
                "uncached_input_tokens"
            ]
        )

        self.total_cache_read_tokens += (
            usage_item[
                "cache_read_tokens"
            ]
        )

        self.total_cache_write_tokens += (
            usage_item[
                "cache_write_tokens"
            ]
        )

        self.total_output_tokens += (
            usage_item[
                "output_tokens"
            ]
        )

        self.total_reasoning_output_tokens += (
            usage_item[
                "reasoning_output_tokens"
            ]
        )

        self.num_calls += 1

        # Return only final agent response
        return final_messages[-1]

    # ============================================================
    # Usage
    # ============================================================

    def get_usage(self) -> Dict[str, Any]:
        """
        Return accumulated usage for all generate() calls
        since construction/reset_usage().
        """

        return {
            "input_tokens":
                self.total_input_tokens,

            "uncached_input_tokens":
                self.total_uncached_input_tokens,

            "cache_read_tokens":
                self.total_cache_read_tokens,

            "cache_write_tokens":
                self.total_cache_write_tokens,

            "output_tokens":
                self.total_output_tokens,

            "reasoning_output_tokens":
                self.total_reasoning_output_tokens,

            "num_calls":
                self.num_calls,

            "usage_history":
                self.usage_history,
        }

    def get_last_usage(
        self
    ) -> Optional[Dict[str, Any]]:

        if not self.usage_history:
            return None

        return self.usage_history[-1]

    def reset_usage(self):
        """
        Reset local usage counters only.

        This has no effect on Codex sessions because every
        generate() call is already a fresh ephemeral session.
        """

        self.usage_history = []

        self.total_input_tokens = 0
        self.total_uncached_input_tokens = 0

        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0

        self.total_output_tokens = 0
        self.total_reasoning_output_tokens = 0

        self.num_calls = 0

    # ============================================================
    # Compatibility helpers
    # ============================================================

    def start_new_task(
        self,
        reset_usage: bool = True,
    ):
        """
        Kept for compatibility with existing code.

        generate() is already stateless, so starting a new task
        only optionally resets local usage statistics.
        """

        if reset_usage:
            self.reset_usage()

    def end_task(self):
        """
        No persistent thread exists, so nothing needs to be closed.
        """
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ):
        self.close()