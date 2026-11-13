"""
Tool definitions shared by every provider.

A Tool is a plain Python callable plus a JSON schema describing its arguments.
The easiest way to make one is the @tool decorator, which infers the schema
from the function signature and docstring:

    @tool
    def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
        '''Return the current weather for a city.

        Args:
            city: Name of the city.
            unit: Temperature unit.
        '''
        ...

If the function declares a parameter annotated with ToolContext, the harness
injects the current context (working directory, env, ...) into it; that
parameter is not exposed to the model.
"""

import asyncio
import inspect
import json
import re
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Union, get_args, get_origin


@dataclass
class ToolContext:
    """Runtime context shared by all tool calls during one agent run."""

    work_dir: str
    # Extra environment variables for subprocesses started by tools.
    env: dict[str, str] = field(default_factory=dict)
    # Cap on characters returned by any single tool call, to protect the context.
    max_output_chars: int = 30000
    # Free-form slot for application state your custom tools need.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    """A function the model can call."""

    name: str
    description: str
    # JSON schema ("type": "object") for the arguments.
    parameters: dict[str, Any]
    func: Callable[..., Any]
    # Name of the parameter that receives the ToolContext, if any.
    context_param: str | None = None
    # True for tools that the Claude Agent SDK implements natively (Read, Bash,
    # ...). The claude provider enables the SDK's own version instead of
    # wrapping this function.
    claude_native: bool = False

    async def ainvoke(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """Run the tool and return its output as text. Never raises."""
        kwargs = dict(args or {})
        if self.context_param:
            kwargs[self.context_param] = ctx
        try:
            if inspect.iscoroutinefunction(self.func):
                out = await self.func(**kwargs)
            else:
                # Sync tools (subprocess, file I/O) must not block the event loop.
                out = await asyncio.to_thread(self.func, **kwargs)
        except TypeError as e:
            return f"Error: bad arguments for {self.name}: {e}"
        except Exception as e:  # noqa: BLE001 - errors go back to the model
            return f"Error executing {self.name}: {e}"
        return _to_text(out)


def _to_text(value: Any) -> str:
    if value is None:
        return "(no output)"
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, default=str)
    except TypeError:
        return str(value)


# ── @tool decorator ────────────────────────────────────────────────────────


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    claude_native: bool = False,
):
    """Turn a function into a Tool. Usable as @tool or @tool(name=...)."""

    def wrap(f: Callable[..., Any]) -> Tool:
        summary, arg_docs = _parse_docstring(inspect.getdoc(f) or "")
        parameters, context_param = _schema_from_signature(f, arg_docs)
        return Tool(
            name=name or f.__name__,
            description=description or summary or f.__name__,
            parameters=parameters,
            func=f,
            context_param=context_param,
            claude_native=claude_native,
        )

    if func is not None:
        return wrap(func)
    return wrap


def _schema_from_signature(f: Callable[..., Any], arg_docs: dict[str, str]):
    sig = inspect.signature(f)
    hints = typing.get_type_hints(f)
    properties: dict[str, Any] = {}
    required: list[str] = []
    context_param = None

    for pname, param in sig.parameters.items():
        annotation = hints.get(pname, str)
        if annotation is ToolContext:
            context_param = pname
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        prop = _json_type(annotation)
        if pname in arg_docs:
            prop["description"] = arg_docs[pname]
        properties[pname] = prop
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return schema, context_param


_SIMPLE_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


def _json_type(annotation: Any) -> dict[str, Any]:
    """Map a Python type annotation to a JSON schema fragment."""
    if annotation in _SIMPLE_TYPES:
        return {"type": _SIMPLE_TYPES[annotation]}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        values = list(args)
        schema = {"enum": values}
        if all(isinstance(v, str) for v in values):
            schema["type"] = "string"
        return schema
    if origin in (Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _json_type(non_none[0])
        return {"anyOf": [_json_type(a) for a in non_none]}
    if origin in (list, tuple, set):
        schema = {"type": "array"}
        if args:
            schema["items"] = _json_type(args[0])
        return schema
    if origin is dict:
        return {"type": "object"}

    # Unknown / Any: let the model send a string.
    return {"type": "string"}


def _parse_docstring(doc: str):
    """Split a Google-style docstring into (summary, {arg: description})."""
    if not doc:
        return "", {}
    parts = re.split(r"^\s*(?:Args|Arguments|Parameters):\s*$", doc, maxsplit=1, flags=re.MULTILINE)
    summary = parts[0].strip()
    arg_docs: dict[str, str] = {}
    if len(parts) > 1:
        current = None
        for line in parts[1].splitlines():
            if re.match(r"^\s*(Returns|Raises|Yields|Examples?):\s*$", line):
                break
            m = re.match(r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line)
            if m:
                current = m.group(1)
                arg_docs[current] = m.group(2).strip()
            elif current and line.strip():
                arg_docs[current] += " " + line.strip()
    return summary, arg_docs


# ── registry ───────────────────────────────────────────────────────────────


class ToolRegistry:
    """An ordered set of tools, addressable by name."""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"Duplicate tool name '{t.name}'")
        self._tools[t.name] = t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    async def dispatch(self, name: str, args: dict[str, Any], ctx: ToolContext) -> str:
        t = self._tools.get(name)
        if t is None:
            return f"Error: unknown tool '{name}'"
        return await t.ainvoke(args, ctx)

    def openai_schemas(self) -> list[dict[str, Any]]:
        """Function-calling definitions for the OpenAI Responses API."""
        return [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self
        ]
