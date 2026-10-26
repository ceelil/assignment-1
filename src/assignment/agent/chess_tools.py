"""Chess tool implementations, decoupled from the agent that registers them.

Every function here takes the HTTP client explicitly instead of reading it off
an agent, so the same code can run in the agent process or inside the sandbox
beside the server it talks to.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

CHESS_PORT = 8000


def _request_state(
    client: httpx.Client, method: str, endpoint: str, **kwargs: Any
) -> dict[str, Any]:
    """Make one chess API request and validate its JSON response."""

    response = client.request(method, endpoint, **kwargs)
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Chess server returned non-JSON ({response.status_code})."
        ) from exc
    if response.status_code >= 400:
        detail = (
            payload.get("detail", payload) if isinstance(payload, dict) else payload
        )
        raise ValueError(str(detail))
    if not isinstance(payload, dict):
        raise RuntimeError("Chess server response must be a JSON object.")
    return payload


def _error(message: str) -> str:
    """A recoverable observation: the model reads it and tries again."""
    return f"<chess_error>{message}</chess_error>"


def _parse_arguments(arguments: Any, tool: str) -> dict[str, Any]:
    """Decode one tool call's raw JSON arguments into a dict, or raise ValueError."""
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        raise ValueError(f"{tool} arguments must be a JSON object string.")
    try:
        parsed = json.loads(arguments) if arguments.strip() else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"{tool} arguments are not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{tool} arguments must be a JSON object.")
    return parsed


def _check_keys(parsed: dict[str, Any], allowed: set[str], tool: str) -> None:
    extra = sorted(set(parsed) - allowed)
    if extra:
        raise ValueError(f"{tool} got unexpected argument(s): {', '.join(extra)}.")


def _call_server(client: httpx.Client, endpoint: str, payload: dict[str, Any]) -> str:
    """POST to the chess API and return the JSON state, or a <chess_error>."""
    try:
        state = _request_state(client, "POST", endpoint, json=payload)
    except ValueError as exc:  # the server rejected the position or move
        return _error(f"Rejected by the chess server: {exc}")
    except httpx.HTTPError as exc:  # timeout, connection refused, ...
        return _error(f"Could not reach the chess server: {type(exc).__name__}: {exc}")
    except RuntimeError as exc:  # non-JSON or non-object response
        return _error(str(exc))
    return json.dumps(state)


def _simulate_move(client: httpx.Client, arguments: str) -> str:
    """New tool: inspect FEN or simulate one ply without changing the game.

    Takes the raw JSON arguments of one tool call and returns the observation
    to send back, so a bad argument or a server error reaches the model as a
    recoverable ``<chess_error>`` instead of ending the run.
    """
    # TODO(Part 3.3.b): Parse the arguments, call the provided
    # /api/simulate endpoint with fen and optional move, and return its JSON.
    # Catch any errors raised by the tool and return an error message between
    # `<chess_error></chess_error>` for the agent to address. Cover malformed
    # JSON arguments, arguments that are not an object, a missing or
    # non-string fen, a non-string move, a position or move the server rejects,
    # and a transport failure.
    try:
        parsed = _parse_arguments(arguments, "simulate_move")
        _check_keys(parsed, {"fen", "move"}, "simulate_move")
    except ValueError as exc:
        return _error(str(exc))
    fen, move = parsed.get("fen"), parsed.get("move")
    if not isinstance(fen, str) or not fen.strip():
        return _error("simulate_move needs `fen`: a complete six-field FEN string.")
    if move is not None and not isinstance(move, str):
        return _error("simulate_move `move` must be a UCI string or null.")

    payload: dict[str, Any] = {"fen": fen}
    if move is not None:
        payload["move"] = move
    return _call_server(client, "/api/simulate", payload)


def _play_move(client: httpx.Client, arguments: str) -> str:
    """Existing tool: play one move as White and return the resulting state.

    Takes the raw JSON arguments of one tool call. Returns the new state, or a
    `<chess_error>` observation if the move could not be played.
    """
    # TODO(3.1.b): Parse the arguments and POST {"move": <uci move>} to
    # /api/move. Return its JSON object. Catch any errors raised by the
    # tool and return an error message between `<chess_error></chess_error>`
    # for the agent to address. Cover malformed JSON arguments, arguments
    # that are not an object, a missing or non-string fen, a non-string move,
    # a position or move the server rejects, and a transport failure.
    try:
        parsed = _parse_arguments(arguments, "play_move")
        _check_keys(parsed, {"move"}, "play_move")
    except ValueError as exc:
        return _error(str(exc))
    move = parsed.get("move")
    if not isinstance(move, str) or not move.strip():
        return _error("play_move needs `move`: a UCI move string such as e2e4.")
    return _call_server(client, "/api/move", {"move": move})


def _run_python(env: Any, port: int, arguments: str) -> str:
    """New tool: run Python with access to the existing registered tools.

    The snippet runs inside the sandbox, which already has the tool
    implementations and the chess server, so code the model wrote never
    executes in the agent process.
    """
    # TODO(3.4): parse the arguments and run the code in the
    # sandbox with the registered tools available by name.
    #
    # `/opt/assignment/sandbox_python.py` is a script on the `env` sandbox
    # that has access to the same tool definitions in this file. Use it to run
    # the code that the model produced as an argument to the run_python tool.
    # The script accepts two positional arguments -- `port` and a base64-encoded
    # string of code (to prevent issues with quoting). Implement this tool
    # call.
    #
    # The script prints one JSON object with `stdout`, `stderr`, and `error`
    # from running the code -- return that string as it is.
    #
    # A non-zero returncode means the sandbox itself failed, not the model's
    # code. Report `exception_info` or `stderr` as a <chess_error>.
    #
    # Return <chess_error>{message}</chess_error> if there are issues like type
    # mismatches or parsing failures.
    try:
        parsed = _parse_arguments(arguments, "run_python")
        _check_keys(parsed, {"code"}, "run_python")
    except ValueError as exc:
        return _error(str(exc))
    code = parsed.get("code")
    if not isinstance(code, str) or not code.strip():
        return _error("run_python needs `code`: a non-empty Python source string.")
    if not isinstance(port, int) or isinstance(port, bool):
        return _error("run_python was configured with a non-integer port.")

    # Base64 keeps quotes, newlines, and $ in the code away from any shell; an
    # argv list with shell=False means no shell parses the command at all.
    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    result = env.execute(
        ["python", "/opt/assignment/sandbox_python.py", str(port), encoded],
        shell=False,
    )
    if result.get("returncode") != 0:
        detail = (
            result.get("exception_info")
            or result.get("stderr")
            or result.get("output")
            or f"exit code {result.get('returncode')}"
        )
        return _error(f"The Python sandbox failed to run the snippet: {detail}")
    # The runner prints exactly one JSON object; pass it through unchanged.
    return result["stdout"] if "stdout" in result else result.get("output", "")


def _invoke_skill(skills: dict[str, dict[str, str]], arguments: str) -> str:
    """Existing tool: load one skill's instructions into the conversation."""
    # TODO(3.5): parse the arguments and return the named skill's content.
    # Return <chess_error>{message}</chess_error> if there are issues like type
    # mismatches or parsing failures.
    try:
        parsed = _parse_arguments(arguments, "invoke_skill")
        _check_keys(parsed, {"name"}, "invoke_skill")
    except ValueError as exc:
        return _error(str(exc))
    name = parsed.get("name")
    if not isinstance(name, str):
        return _error("invoke_skill needs `name`: a skill name string.")
    skill = skills.get(name)
    if skill is None:
        available = ", ".join(skills) or "none"
        return _error(f"No skill named {name!r}. Available skills: {available}.")
    return skill["content"]


def _game_state(client: httpx.Client, reset: bool = False) -> dict:
    """Read the live game, or start a new one and read the opening position."""

    method, endpoint = ("POST", "/api/reset") if reset else ("GET", "/api/state")
    return _request_state(client, method, endpoint)
