"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

# Optional `execute` arguments, with the Python types each may take. A model
# may send null for an omitted argument, so None always means "use the default".
_EXECUTE_OPTIONAL_ARGUMENTS: dict[str, tuple[type, ...]] = {
    "shell": (bool,),
    "cwd": (str,),
    "timeout": (int, float),
    "env": (dict,),
}


def _error(message: str) -> str:
    """A tool observation telling the model its call failed and why."""
    return f"<error>{message}</error>"


class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        # TODO(Part 1.3): Make the `execute` and `send_message` tools available
        # to the agent.
        # The base class may already have added invoke_skill; keep it.
        self.tools = [EXECUTE_TOOL, SEND_MESSAGE_TOOL, *self.tools]

        # TODO(1.1.b): Construct the system prompt and task_prompt. These
        # should be usable by the `Agent.build_prompt` method.
        # TODO(1.4): If any skills are available to the agent, make their
        # descriptions/metadata available to the agent in the prompt.
        self.system_prompt = self._build_system_prompt()
        self.task_prompt = self._build_task_prompt()

    def _build_system_prompt(self) -> str:
        """Standing instructions: role, environment, way of working, skills."""

        system_information = json.dumps(
            {
                "machine": self.env.machine,
                "release": self.env.release,
                "system": self.env.system,
                "version": self.env.version,
            },
            indent=2,
        )
        prompt = (
            "You are an autonomous software engineering agent. You resolve "
            "issues in a code repository by running commands in a sandboxed "
            "terminal.\n"
            "\n"
            "<system_information>\n"
            f"{system_information}\n"
            "</system_information>\n"
            "\n"
            "How to work:\n"
            "- Act only through tool calls. Each response should contain at "
            "least one tool call; text alone does nothing.\n"
            "- Start by exploring the repository and reproducing the problem, so "
            "you can confirm the cause before changing anything.\n"
            "- Make the smallest change to the source code that fixes the root "
            "cause. Do not weaken or delete existing tests to make them pass.\n"
            "- Preserve existing interfaces: return values and responses should "
            "keep the same shape (the same keys and types) as in the normal case, "
            "unless the issue asks for a change. Look at how the surrounding code "
            "builds its results before writing an early return.\n"
            "- Verify the fix against the issue, not just the absence of an "
            "error: rerun your reproduction, then go through every expected "
            "behaviour the issue states and check each one explicitly against "
            "the actual output (for example with assertions in a small script). "
            "Then run the relevant existing tests.\n"
            "- Keep command output short; read files in ranges rather than whole.\n"
            "- When the task is complete, call `send_message` with a short "
            "summary of what you changed and how you verified it. This ends "
            "the session, so do it only once all work is done."
        )
        if self.skills:
            catalog = "\n\n".join(skill["metadata"] for skill in self.skills.values())
            prompt += (
                "\n\nReusable skills are available. Call `invoke_skill` with a "
                "skill's name to load its full instructions, and follow them in "
                "place of your default approach.\n"
                f"<skills>\n{catalog}\n</skills>"
            )
        return prompt

    def _build_task_prompt(self) -> str:
        """The task statement, framed as a request from a user."""

        return (
            "Resolve the following issue in the repository at "
            f"`{self.env.cwd}`.\n"
            "\n"
            f"<issue>\n{self.task.strip()}\n</issue>"
        )

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        # TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.
        observations = []
        for call in tool_calls:
            function = call.get("function") or {}
            content = self._run_tool(
                function.get("name"), function.get("arguments")
            )
            observations.append(
                {"role": "tool", "tool_call_id": call.get("id", ""), "content": content}
            )
        return observations

    def _run_tool(self, name: Any, raw_arguments: Any) -> str:
        """Run one tool call and return the observation text for the model."""

        available = [tool["function"]["name"] for tool in self.tools]
        if name not in available:
            return _error(
                f"Unknown tool {name!r}. Available tools: {', '.join(available)}."
            )

        # Arguments arrive as a JSON string written by the model; it may be
        # broken, so parse defensively. An empty string means no arguments.
        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
        except (json.JSONDecodeError, TypeError) as exc:
            return _error(f"Arguments for {name} are not valid JSON: {exc}")
        if not isinstance(arguments, dict):
            return _error(f"Arguments for {name} must be a JSON object.")

        if name == "execute":
            return self._execute(arguments)
        if name == "send_message":
            return self._send_message(arguments)
        return self._invoke_skill(arguments)

    def _execute(self, arguments: dict[str, Any]) -> str:
        command = arguments.get("command")
        is_argv = isinstance(command, list) and all(
            isinstance(part, str) for part in command
        )
        if not (isinstance(command, str) and command.strip()) and not is_argv:
            return _error(
                "execute needs `command`: a non-empty string or a list of strings."
            )

        options = {}
        for key, types in _EXECUTE_OPTIONAL_ARGUMENTS.items():
            value = arguments.get(key)
            if value is None:
                continue  # keep Environment.execute's default
            # bool is a subclass of int; do not accept true as a timeout.
            if not isinstance(value, types) or (
                key == "timeout" and isinstance(value, bool)
            ):
                return _error(f"execute argument `{key}` has the wrong type.")
            options[key] = value

        result = self.env.execute(command, **options)
        # `output` already interleaves stdout and stderr; sending the separate
        # streams too would show the model everything twice.
        observation = {
            "output": result.get("output", ""),
            "returncode": result.get("returncode"),
        }
        if result.get("exception_info"):
            observation["exception_info"] = result["exception_info"]
        return format_tool_output(observation)

    def _send_message(self, arguments: dict[str, Any]) -> str:
        summary = arguments.get("summary")
        if not isinstance(summary, str):
            return _error("send_message needs a string `summary`.")

        # The message ends the task. Record the submission if one was made,
        # so it is available on the agent object as well as in the sandbox.
        self.finished = True
        if self.skills:
            result = self.env.execute("cat patch.txt")
            if result.get("returncode") == 0:
                self.submitted_patch = result.get("output", "")
        return "Message sent. The task is complete."

    def _invoke_skill(self, arguments: dict[str, Any]) -> str:
        name = arguments.get("name")
        if not isinstance(name, str):
            return _error("invoke_skill needs a string `name`.")
        skill = self.skills.get(name)
        if skill is None:
            return _error(
                f"No skill named {name!r}. Available skills: "
                f"{', '.join(self.skills)}."
            )
        return skill["content"]
