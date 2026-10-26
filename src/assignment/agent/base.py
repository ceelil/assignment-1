"""The domain-independent ReAct loop shared by both agents.

Part 1 completes the generic loop here; the two subclasses in this package
supply only their own tools and tool executors.
"""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
import yaml

from assignment.env import Environment
from assignment.agent.tools import INVOKE_SKILL_TOOL

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_COMPACTION_KEEP_RECENT_STEPS = 1
DEFAULT_COMPACTION_MAX_TOKENS = 1_200
MAX_OBSERVATION_CHARS = 10_000

# TODO(Part 2): Write instructions that make the model produce concise working
# memory for a software agent. The prompt should preserve concrete progress,
# failures, test results, constraints, and next steps without copying raw output.
COMPACTION_SYSTEM_PROMPT = """You maintain the working memory of an autonomous agent that is partway \
through a task. You will receive the task, any earlier working memory, and a transcript of the \
agent's older steps (its actions and the tool results it observed). Those older steps are about \
to be deleted from the agent's context; your summary is the only record of them that will remain.

Write a concise, factual working memory that lets the agent continue without repeating work. \
Use these headings, and keep every fact that is still relevant:

- Objective: the goal and any constraints or rules the agent must respect.
- Key facts: files, functions, paths, and line numbers involved; how the environment is set up.
- Actions and results: commands run and edits made, with their concrete outcomes \
(errors seen, test names and pass/fail counts, values printed).
- Failed approaches: what was tried and did not work, and why.
- Current status and blockers.
- Next action: the single most useful next step.

Rules: state facts, not speculation. Do not copy raw tool output; quote only short exact \
strings that matter (an error message, a test id, a changed line). Merge the earlier working \
memory with the new steps rather than repeating it. Keep it under 400 words. Output only the \
working memory."""

# Per-message character budget when an old step is shown to the summarizer.
# Observations are already capped near 10,000 characters; the summarizer needs
# the gist, not every line, so this keeps the compaction request itself small.
COMPACTION_SOURCE_MAX_CHARS = 3_000

# Sent back when the model replies without calling a tool. It names no tool or
# domain, so the shared loop stays usable by every agent.
NO_TOOL_CALL_MESSAGE = (
    "Your last response did not contain a tool call, so nothing was executed. "
    "Continue the task by calling one of the available tools."
)


def _clip(text: str, limit: int = COMPACTION_SOURCE_MAX_CHARS) -> str:
    """Keep both ends of a long string, as format_tool_output does."""

    if len(text) <= limit:
        return text
    keep = (limit - 60) // 2
    return f"{text[:keep]}\n[{len(text) - 2 * keep} characters omitted]\n{text[-keep:]}"


def render_transcript(messages: list[dict[str, Any]]) -> str:
    """Show messages to the summarizer as plain text, one block per message.

    Sending old messages as text inside one user message (rather than as real
    assistant/tool turns) keeps the request valid without tools, and makes it
    clear to the summarizer that these are records to summarize, not a
    conversation to continue.
    """

    blocks = []
    for message in messages:
        role = message.get("role", "unknown")
        lines = [f"[{role}]"]
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            lines.append(_clip(content))
        elif isinstance(content, list):
            # Structured content: keep only the text parts.
            text = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
            if text.strip():
                lines.append(_clip(text))
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            lines.append(
                f"tool call {call.get('id', '')}: {function.get('name')}"
                f"({_clip(str(function.get('arguments', '')))})"
            )
        if role == "tool":
            lines[0] = f"[tool result for {message.get('tool_call_id', '')}]"
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _parse_skill_frontmatter(content: str, source: Path) -> tuple[str, str]:
    """Return (name, description) from a SKILL.md's leading YAML block."""

    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{source} must start with a '---' frontmatter line")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise ValueError(f"{source} frontmatter has no closing '---' line") from None

    try:
        frontmatter = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as exc:
        raise ValueError(f"{source} frontmatter is not valid YAML: {exc}") from exc
    if not isinstance(frontmatter, dict):
        raise ValueError(f"{source} frontmatter must be a mapping of keys to values")

    fields = []
    for key in ("name", "description"):
        value = frontmatter.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{source} frontmatter needs a non-empty {key!r}")
        fields.append(value.strip())
    return fields[0], fields[1]


class StepLimitError(Exception):
    """Raised when an agent exhausts its model-call budget."""


def format_tool_output(output: dict[str, Any]) -> str:
    """Format a terminal result as a compact, tagged model observation."""

    elements: list[str] = []
    for key in sorted(output):
        value = output[key]
        if isinstance(value, str) and len(value) > MAX_OBSERVATION_CHARS:
            # Leave room for the elision notice so the formatted value itself,
            # not just its retained source slices, stays below the limit.
            retained_at_each_end = 4_900
            omitted = len(value) - (2 * retained_at_each_end)
            value = (
                f"{value[:retained_at_each_end]}\n"
                f"[{omitted} characters elided; read a narrower range]\n"
                f"{value[-retained_at_each_end:]}"
            )
        elements.append(f"<{key}>{value}</{key}>")
    return "\n".join(elements)


def rough_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate prompt tokens without a provider-specific tokenizer."""

    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(serialized) / 4))


class Agent:
    """Base class for a ReAct agent with pluggable tools."""

    def __init__(
        self,
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
        self.env = environment
        self.model = model or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise RuntimeError("OPENAI_MODEL is not set.")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise RuntimeError("OPENAI_BASE_URL is not set.")
        try:
            max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
        except ValueError as exc:
            raise RuntimeError("OPENAI_MAX_RETRIES must be an integer.") from exc
        if max_retries < 0:
            raise RuntimeError("OPENAI_MAX_RETRIES must be non-negative.")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
        )

        self.logs_save_path = logs_save_path
        self.step_limit = step_limit
        self.auto_stop_environment = auto_stop_environment
        if compact_threshold_tokens is not None and compact_threshold_tokens <= 0:
            raise ValueError("compact_threshold_tokens must be positive or None")
        if (
            compaction_keep_recent_steps is not None
            and compaction_keep_recent_steps < 1
        ):
            raise ValueError("compaction_keep_recent_steps must be at least 1")
        if compaction_max_tokens is not None and compaction_max_tokens < 1:
            raise ValueError("compaction_max_tokens must be positive")
        # A None threshold turns compaction off. The other two settings then
        # describe a compaction that never happens, so fall back to the
        # defaults rather than leaving a None for later code to trip over.
        self.compact_threshold_tokens = compact_threshold_tokens
        self.compaction_keep_recent_steps = (
            DEFAULT_COMPACTION_KEEP_RECENT_STEPS
            if compaction_keep_recent_steps is None
            else compaction_keep_recent_steps
        )
        self.compaction_max_tokens = (
            DEFAULT_COMPACTION_MAX_TOKENS
            if compaction_max_tokens is None
            else compaction_max_tokens
        )

        # Each agent supplies its own opening messages: the standing
        # instructions, and the task statement that starts the run.
        self.system_prompt: str = ""
        self.task_prompt: str = ""

        self.api_prompts: list[list[dict[str, Any]]] = []
        self.api_responses: list[dict[str, Any]] = []
        self.compaction_events: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self.finished = False
        self.steps_taken = 0

        self.skills_path = Path(skills_path) if skills_path is not None else None
        self.skills: dict[str, dict[str, str]] = (
            self.load_skills(self.skills_path) if self.skills_path is not None else {}
        )

        if self.skills:
            self.tools.append(INVOKE_SKILL_TOOL)

        # TODO(1.1.a): Add machinery to maintain agent state as it takes actions
        # and observes the results.
        # The interaction so far, in order: each assistant action, the tool
        # observations linked to it, and any corrective user messages. The
        # system and task prompts are kept apart because subclasses set them
        # after this constructor runs, and build_prompt reads them fresh.
        self.history: list[dict[str, Any]] = []
        # Model-written summary of steps that compaction removed from history.
        # None until the first compaction (Part 2).
        self.working_memory: str | None = None

    def load_skills(self, skills_path: Path) -> dict[str, dict[str, str]]:
        """Load the skill folders exposed to this agent."""

        # TODO(1.4): Validate ``skills_path``, discover one ``SKILL.md``
        # per child directory, parse its YAML frontmatter (what's between the
        # `---` tags at the head of the file), and return a mapping
        # keyed by the frontmatter ``name``. Each value must contain a concise
        # ``metadata`` string for the model's skill catalog and the complete
        # ``content`` of the skill file for ``invoke_skill``. Reject duplicate
        # names and malformed or missing frontmatter with a clear
        # ``ValueError``.
        skills_path = Path(skills_path)
        if not skills_path.is_dir():
            raise ValueError(f"Skills path is not a directory: {skills_path}")

        skills: dict[str, dict[str, str]] = {}
        # Sorted so the catalog order is the same on every machine.
        for skill_dir in sorted(p for p in skills_path.iterdir() if p.is_dir()):
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.is_file():
                raise ValueError(f"Skill folder {skill_dir} has no SKILL.md")

            content = skill_file.read_text(encoding="utf-8")
            name, description = _parse_skill_frontmatter(content, skill_file)
            if name in skills:
                raise ValueError(
                    f"Duplicate skill name {name!r} in {skill_file}"
                )
            skills[name] = {
                # Level 1 of progressive disclosure: always in the prompt.
                "metadata": f"name: {name}\ndescription: {description}",
                # Level 2: shown only when the model calls invoke_skill.
                "content": content,
            }
        return skills

    def query_language_model(self) -> dict[str, Any]:
        """Send one tool-enabled Chat Completions request and normalize it."""

        messages = self.build_prompt()
        self.api_prompts.append(deepcopy(messages))
        step_number = self.steps_taken + 1
        print(
            f"[agent] step {step_number}/{self.step_limit}: requesting action",
            flush=True,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                reasoning_effort="medium",
                max_completion_tokens=4096,
            )
        except Exception as exc:
            print(
                f"[agent] step {step_number}: model request failed after retries "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
            raise
        self.api_responses.append(response.model_dump(mode="json"))
        self.steps_taken += 1
        message = self.process_response(response)
        tool_names = [
            call.get("function", {}).get("name", "unknown")
            for call in message.get("tool_calls", [])
            if isinstance(call, dict)
        ]
        if tool_names:
            print(
                f"[agent] step {step_number}: tool call(s): {', '.join(tool_names)}",
                flush=True,
            )
        else:
            print(
                f"[agent] step {step_number}: response contained no parsed tool call; "
                "the loop should preserve the response and continue",
                flush=True,
            )
        return message

    def process_response(self, response: Any) -> dict[str, Any]:
        """Return relevant parts of the language model's response."""

        return response.choices[0].message.model_dump(exclude_none=True)

    def build_prompt(self) -> list[dict[str, Any]]:
        # TODO(1.1.a): Construct a sequence of messages that form the language
        # model prompt. This should include standing instructions, task
        # specification, prior interaction including observations, reasoning,
        # and actions from previous turns. Note that this method should be
        # domain-agnostic and construct the prompt in a way that would apply
        # to any of the inheriting domain-specific agents.

        # You want to be careful about which attributes of the class you modify
        # here as they may also be handled by the subclasses.

        # Read-only: building a prompt never changes the agent's state, so it
        # is safe to call any number of times (token estimates call it too).
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.task_prompt},
        ]
        if self.working_memory:
            # Stands in for the removed steps, between the task and the
            # steps kept verbatim.
            messages.append({"role": "user", "content": self._working_memory_message()})
        messages.extend(deepcopy(self.history))
        return messages

    def _working_memory_message(self) -> str:
        return (
            "Earlier steps of this task were compacted to save context. This is "
            "the working memory recorded from them:\n"
            f"<working_memory>\n{self.working_memory}\n</working_memory>\n"
            "Your most recent steps follow verbatim. Continue from there."
        )

    def estimate_active_prompt_tokens(self) -> int:
        """Estimate the next prompt, calibrated by the provider's latest usage."""

        current_prompt = self.build_prompt()
        rough_current = rough_message_tokens(current_prompt)
        if not self.api_prompts or not self.api_responses:
            return rough_current

        usage = self.api_responses[-1].get("usage") or {}
        actual_previous = usage.get("prompt_tokens")
        if not isinstance(actual_previous, int):
            return rough_current

        rough_previous = rough_message_tokens(self.api_prompts[-1])
        added_since_previous_request = max(0, rough_current - rough_previous)
        return actual_previous + added_since_previous_request

    @property
    def compaction_enabled(self) -> bool:
        """Whether this agent compacts its context at all."""

        return self.compact_threshold_tokens is not None

    def compact_context(self):
        """Replace parts of prompt with model-generated working memory. Changes the
        content that `build_prompt` emits."""

        # TODO(2.1): Prompt the model to compact the context. The system
        # prompt should ask for concise factual working memory and preserve
        # the objective, constraints, files, commands, edits, concrete
        # results, failed approaches, tests, blockers, and next action.
        # Summarize only an old prefix; retain the original system/task
        # messages verbatim and at least the latest complete assistant action
        # with all linked tool observations. The resulting summary should change
        # what `build_prompt` emits, and reduce the length of the prompt.

        # Split history at the start of the k-th most recent assistant action.
        # Everything from there on is kept verbatim: each kept action stays
        # together with all of its linked tool observations, so the prompt
        # remains a valid conversation. Everything before it is summarized.
        assistant_indexes = [
            i for i, m in enumerate(self.history) if m.get("role") == "assistant"
        ]
        split = assistant_indexes[-self.compaction_keep_recent_steps]
        old_steps, recent_steps = self.history[:split], self.history[split:]

        source = f"<task>\n{self.task_prompt}\n</task>\n\n"
        if self.working_memory:
            source += (
                "<previous_working_memory>\n"
                f"{self.working_memory}\n"
                "</previous_working_memory>\n\n"
            )
        source += (
            "<steps_to_summarize>\n"
            f"{render_transcript(old_steps)}\n"
            "</steps_to_summarize>\n\n"
            "Write the updated working memory."
        )
        compaction_prompt = [
            {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
            {"role": "user", "content": source},
        ]

        ### Do not modify this section ###
        compaction_response = self.client.chat.completions.create(
            model=self.model,
            messages=compaction_prompt,
            reasoning_effort="medium",
            max_completion_tokens=self.compaction_max_tokens,
        )
        ##################################

        # Use `compaction_response` to update what `build_prompt` emits, but
        # DO NOT modify the object itself. Let the method return it unchanged.
        summary = (compaction_response.choices[0].message.content or "").strip()
        if summary:
            self.working_memory = summary
            self.history = recent_steps
        else:
            # An empty summary (for example, the token budget went to
            # reasoning) would erase the old steps with nothing in their place.
            # Keep the context as it was; the event is still logged.
            print(
                "[agent] compaction returned an empty summary; context kept",
                flush=True,
            )

        ### Do not modify this section ###
        return compaction_prompt, compaction_response.model_dump(mode="json")
        ##################################

    def maybe_compact_context(self) -> bool:
        """Compact before the next action request when the threshold is reached."""

        if not self.compaction_enabled:
            return False

        # Context too short to compact yet
        if self.estimate_active_prompt_tokens() < self.compact_threshold_tokens:
            return False

        prompt_before = deepcopy(self.build_prompt())

        # Not enough steps (each assistant turn corresponds to a step) to force
        # compaction yet
        if (
            len([m for m in prompt_before if m.get("role") == "assistant"])
            <= self.compaction_keep_recent_steps
        ):
            return False

        compaction_prompt, compaction_response = self.compact_context()
        prompt_after = deepcopy(self.build_prompt())
        self.compaction_events.append(
            {
                "step": self.steps_taken,
                "estimated_tokens_before": rough_message_tokens(prompt_before),
                "estimated_tokens_after": rough_message_tokens(prompt_after),
                "active_prompt_before": deepcopy(prompt_before),
                "compaction_prompt": compaction_prompt,
                "compaction_response": compaction_response,
            }
        )
        return True

    def run(self) -> None:
        """Run ReAct steps, always saving the trajectory and stopping Modal."""

        try:
            # TODO(1.2) Run the ReAct loop. Orchestrate the sequence of
            # prompting the language model to produce reasoning and actions,
            # extracting the tool calls produced by the model, and executing
            # the tool calls to obtain the agent's observation for the next
            # step. Ensure you identify when the agent has completed the task
            # by setting `Agent.finished`. If the agent exceeds the
            # `step_limit`, raise `StepLimitError`.

            # TODO(2.2) Call `maybe_compact_context()` before each new action
            # request in your shared loop. It already estimates active tokens
            # and handles the threshold, and tracks compaction events for
            # logging.

            while not self.finished:
                # Check the budget before spending another model call, so a run
                # makes exactly step_limit calls before giving up.
                if self.steps_taken >= self.step_limit:
                    raise StepLimitError(
                        f"Agent did not finish within {self.step_limit} steps."
                    )

                # Summarize old steps first if the context has grown too big.
                # A no-op when compaction is off or the threshold is not met.
                self.maybe_compact_context()

                # Reason + Act: one model call returns one assistant message,
                # which may hold text, reasoning, and zero or more tool calls.
                message = self.query_language_model()
                self.history.append(message)

                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    # Text-only reply: keep it (it is part of what happened)
                    # and answer with a user message, so the conversation stays
                    # valid and the model is steered back to acting.
                    self.history.append(
                        {"role": "user", "content": NO_TOOL_CALL_MESSAGE}
                    )
                    continue

                # Observe: one linked tool message per call. The subclass may
                # set self.finished while executing (for example send_message).
                observations = self.execute_tool_calls(tool_calls)
                self.history.extend(observations)
        finally:
            # This block is provided infrastructure. Do not modify it: a
            # trajectory is required even when a run fails.
            if self.logs_save_path:
                path = Path(self.logs_save_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "prompts": self.api_prompts,
                            "responses": self.api_responses,
                            "compactions": self.compaction_events,
                        },
                        indent=2,
                    )
                )
            if self.auto_stop_environment:
                stop = getattr(self.env, "stop", None)
                if callable(stop):
                    stop()

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute domain-specific calls and return linked tool observations."""

        # You do not need to implement anything here. This method is
        # domain-specific and implemented by the relevant subclasses
        raise NotImplementedError
