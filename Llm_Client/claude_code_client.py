import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional, Union, List, Dict, Any


import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ClaudeCodeChatClient:
    """
    Stateless Claude Code CLI wrapper.

    Intended for independent LLM generation calls:
    - no previous conversation/session
    - no auto memory
    - no CLAUDE.md
    - no skills/plugins/hooks/MCP customizations
    - no tools
    - custom system prompt replaces Claude Code's default system prompt
    - default structured JSON output
    - full usage statistics collected from JSON output
    """

    def __init__(
        self,
        workdir: Union[str, Path],
        claude_bin: str = "claude",
        model: Optional[str] = None,
        timeout: Optional[int] = None,

        # Isolation
        safe_mode: bool = True,
        no_session_persistence: bool = True,
        disable_tools: bool = True,

        # One pure generation turn
        max_turns: Optional[int] = 1,

        # System prompt
        system_prompt_mode: str = "replace",

        # Output cleanup
        strip_code_fence: bool = True,

        # HF-compatible arguments.
        # Claude Code CLI does not directly map these.
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,

        # Structured JSON output
        json_schema: Optional[Dict[str, Any]] = None,
    ):
        self.workdir = Path(workdir).resolve()

        if not self.workdir.exists():
            raise FileNotFoundError(
                f"workdir does not exist: {self.workdir}"
            )

        self.claude_bin = claude_bin
        self.model = model
        self.timeout = timeout

        self.safe_mode = safe_mode
        self.no_session_persistence = no_session_persistence
        self.disable_tools = disable_tools

        self.max_turns = max_turns
        self.system_prompt_mode = system_prompt_mode
        self.strip_code_fence_enabled = strip_code_fence

        # Compatibility only
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        # Default: constrain model-generated output to a JSON object.
        # Pass a specific schema for strict fields and types.
        self.json_schema = (
            json_schema if json_schema is not None
            else {"type": "object"}
        )

        self.reset_usage()

    # ============================================================
    # Text helpers
    # ============================================================

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        text = text.strip()

        if not text.startswith("```"):
            return text

        lines = text.splitlines()

        if len(lines) < 2:
            return text

        if lines[-1].strip() != "```":
            return text

        first = lines[0].strip().lower()

        if first in {
            "```",
            "```json",
            "```python",
            "```text",
        }:
            return "\n".join(lines[1:-1]).strip()

        return text

    @staticmethod
    def _messages_to_parts(
        messages: List[Dict[str, str]]
    ) -> tuple[str, str]:
        """
        Convert messages such as:

        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]

        into:
            system_prompt
            user_prompt
        """

        if not isinstance(messages, list) or not messages:
            raise ValueError(
                "messages must be a non-empty list"
            )

        system_parts = []
        conversation_parts = []

        for message in messages:
            if not isinstance(message, dict):
                raise TypeError(
                    "Each message must be a dict."
                )

            role = str(
                message.get("role", "")
            ).strip().lower()

            content = message.get("content", "")

            if not role:
                raise ValueError(
                    f"message missing role: {message}"
                )

            if not isinstance(content, str):
                raise TypeError(
                    f"message content must be str: {message}"
                )

            content = content.strip()

            if not content:
                continue

            if role == "system":
                system_parts.append(content)

            elif role == "user":
                conversation_parts.append(content)

            else:
                # Kept for compatibility if multi-message input is used.
                conversation_parts.append(
                    f"[{role.upper()}]\n{content}"
                )

        system_prompt = "\n\n".join(
            system_parts
        ).strip()

        user_prompt = "\n\n".join(
            conversation_parts
        ).strip()

        if not user_prompt:
            raise ValueError(
                "No user content found in messages."
            )

        return system_prompt, user_prompt

    # ============================================================
    # CLI construction
    # ============================================================

    def _build_cmd(
        self,
        system_prompt: str
    ) -> List[str]:

        cmd = [
            self.claude_bin,
            "-p",
            "--output-format",
            "json",
            "--input-format",
            "text",
        ]

        # --------------------------------------------------------
        # Structured JSON output (enabled by default)
        # --------------------------------------------------------

        cmd.extend([
            "--json-schema",
            json.dumps(
                self.json_schema,
                ensure_ascii=False,
            ),
        ])

        # --------------------------------------------------------
        # Disable Claude Code customizations and persistent memory
        # --------------------------------------------------------

        if self.safe_mode:
            cmd.append("--safe-mode")

        # --------------------------------------------------------
        # Do not save this session
        # --------------------------------------------------------

        if self.no_session_persistence:
            cmd.append("--no-session-persistence")

        # --------------------------------------------------------
        # Pure LLM generation: disable all built-in tools
        # --------------------------------------------------------

        if self.disable_tools:
            cmd.extend([
                "--tools",
                "",
            ])

        # --------------------------------------------------------
        # Model
        # --------------------------------------------------------

        if self.model:
            cmd.extend([
                "--model",
                self.model,
            ])

        # --------------------------------------------------------
        # Only one agentic turn
        # --------------------------------------------------------

        if self.max_turns is not None:
            cmd.extend([
                "--max-turns",
                str(self.max_turns),
            ])

        # --------------------------------------------------------
        # Custom system prompt
        # --------------------------------------------------------

        if system_prompt:

            if self.system_prompt_mode == "replace":
                cmd.extend([
                    "--system-prompt",
                    system_prompt,
                ])

            elif self.system_prompt_mode == "append":
                cmd.extend([
                    "--append-system-prompt",
                    system_prompt,
                ])

            else:
                raise ValueError(
                    "system_prompt_mode must be "
                    "'replace' or 'append'"
                )

        return cmd

    # ============================================================
    # Generation
    # ============================================================

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> str:
        """
        Execute one independent Claude Code generation.

        max_new_tokens / temperature / do_sample are accepted
        only for compatibility with HF-style clients.

        Returns:
            JSON string.
        """

        system_prompt, user_prompt = (
            self._messages_to_parts(messages)
        )

        cmd = self._build_cmd(system_prompt)

        started = time.time()

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
                input=user_prompt,
                timeout=self.timeout,
            )

        except subprocess.TimeoutExpired:
            proc.kill()

            stdout, stderr = proc.communicate()

            raise TimeoutError(
                f"Claude Code timed out after "
                f"{self.timeout} seconds\n"
                f"cmd: {shlex.join(cmd)}\n"
                f"stdout:\n{stdout.strip()}\n"
                f"stderr:\n{stderr.strip()}"
            )

        elapsed = time.time() - started

        if proc.returncode != 0:
            raise RuntimeError(
                f"Claude Code failed with return code "
                f"{proc.returncode}\n"
                f"cmd: {shlex.join(cmd)}\n"
                f"stdout:\n{stdout.strip()}\n"
                f"stderr:\n{stderr.strip()}"
            )

        raw = stdout.strip()

        if not raw:
            raise ValueError(
                "Claude Code returned empty stdout.\n"
                f"stderr:\n{stderr.strip()}"
            )

        # --------------------------------------------------------
        # Parse Claude Code JSON result
        # --------------------------------------------------------

        try:
            result_obj = json.loads(raw)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Claude Code did not return valid JSON "
                "despite --output-format json.\n"
                f"stdout:\n{raw}"
            ) from exc

        # --------------------------------------------------------
        # Generated response
        # --------------------------------------------------------

        if result_obj.get("is_error"):
            raise RuntimeError(
                f"Claude Code reported an error: "
                f"{result_obj.get('result', '')}"
            )

        structured_output = result_obj.get(
            "structured_output"
        )

        if structured_output is None:
            raise ValueError(
                "Claude Code returned no structured_output. "
                f"Response: {result_obj}"
            )

        # Keep generate() -> str for compatibility.
        # The caller can use json.loads(response).
        result_text = json.dumps(
            structured_output,
            ensure_ascii=False,
        )

        # --------------------------------------------------------
        # Usage
        # --------------------------------------------------------

        usage = result_obj.get("usage") or {}

        input_tokens = (
            usage.get("input_tokens", 0) or 0
        )

        output_tokens = (
            usage.get("output_tokens", 0) or 0
        )

        cache_creation_input_tokens = (
            usage.get(
                "cache_creation_input_tokens",
                0,
            )
            or 0
        )

        cache_read_input_tokens = (
            usage.get(
                "cache_read_input_tokens",
                0,
            )
            or 0
        )

        # Anthropic's input_tokens does not include the
        # cache creation/read components.
        total_input_tokens = (
            input_tokens
            + cache_creation_input_tokens
            + cache_read_input_tokens
        )

        total_tokens = (
            total_input_tokens
            + output_tokens
        )

        total_cost_usd = result_obj.get(
            "total_cost_usd"
        )

        usage_item = {
            # Fresh / uncached input
            "input_tokens": input_tokens,

            # Prompt caching
            "cache_creation_input_tokens":
                cache_creation_input_tokens,

            "cache_read_input_tokens":
                cache_read_input_tokens,

            # Convenient aggregate
            "total_input_tokens":
                total_input_tokens,

            "output_tokens":
                output_tokens,

            "total_tokens":
                total_tokens,

            # Claude Code metadata
            "total_cost_usd":
                total_cost_usd,

            "num_turns":
                result_obj.get("num_turns"),

            "duration_ms":
                result_obj.get("duration_ms"),

            "duration_api_ms":
                result_obj.get("duration_api_ms"),

            "elapsed_seconds":
                elapsed,

            "session_id":
                result_obj.get("session_id"),

            "is_error":
                result_obj.get("is_error"),

            "stop_reason":
                result_obj.get("stop_reason"),
        }

        self.usage_history.append(
            usage_item
        )

        # --------------------------------------------------------
        # Accumulate client-level statistics
        # --------------------------------------------------------

        self.total_input_tokens += (
            input_tokens
        )

        self.total_cache_creation_input_tokens += (
            cache_creation_input_tokens
        )

        self.total_cache_read_input_tokens += (
            cache_read_input_tokens
        )

        self.total_output_tokens += (
            output_tokens
        )

        self.total_tokens += (
            total_tokens
        )

        if isinstance(
            total_cost_usd,
            (int, float),
        ):
            self.total_cost_usd += (
                total_cost_usd
            )

        self.num_calls += 1

        return result_text

    # ============================================================
    # Usage
    # ============================================================

    def get_usage(self) -> Dict[str, Any]:

        total_input_tokens = (
            self.total_input_tokens
            + self.total_cache_creation_input_tokens
            + self.total_cache_read_input_tokens
        )

        return {
            # Actually processed as new input
            "input_tokens":
                self.total_input_tokens,

            # Cache
            "cache_creation_input_tokens":
                self.total_cache_creation_input_tokens,

            "cache_read_input_tokens":
                self.total_cache_read_input_tokens,

            # All logical input tokens
            "total_input_tokens":
                total_input_tokens,

            # Output
            "output_tokens":
                self.total_output_tokens,

            # Input + output
            "total_tokens":
                self.total_tokens,

            # Cost reported by Claude Code
            "total_cost_usd":
                self.total_cost_usd,

            "num_calls":
                self.num_calls,

            "usage_history":
                self.usage_history,
        }

    def reset_usage(self):

        self.usage_history = []

        self.total_input_tokens = 0
        self.total_cache_creation_input_tokens = 0
        self.total_cache_read_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0

        self.total_cost_usd = 0.0
        self.num_calls = 0

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