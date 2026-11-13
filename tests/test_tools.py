from typing import Literal

import pytest

from omniharness import BUILTIN_TOOLS, ToolContext, ToolRegistry, tool


@tool
def sample(ctx: ToolContext, city: str, days: int = 3, unit: Literal["c", "f"] = "c",
           tags: list[str] | None = None) -> dict:
    """Get a forecast.

    Args:
        city: City name.
        days: How many days,
            at most 10.
        unit: Unit.
    """
    return {"city": city, "days": days, "unit": unit, "cwd": ctx.work_dir}


def test_schema_inference():
    assert sample.name == "sample"
    assert sample.description == "Get a forecast."
    assert sample.context_param == "ctx"
    params = sample.parameters
    assert params["required"] == ["city"]
    assert "ctx" not in params["properties"]
    assert params["properties"]["city"] == {"type": "string", "description": "City name."}
    assert params["properties"]["days"]["type"] == "integer"
    assert params["properties"]["days"]["description"] == "How many days, at most 10."
    assert params["properties"]["unit"] == {"enum": ["c", "f"], "type": "string",
                                            "description": "Unit."}
    assert params["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}


async def test_invoke_injects_context_and_serializes(tmp_path):
    ctx = ToolContext(work_dir=str(tmp_path))
    out = await sample.ainvoke({"city": "Paris"}, ctx)
    assert '"city": "Paris"' in out
    assert str(tmp_path) in out


async def test_invoke_reports_errors():
    @tool
    def boom() -> str:
        """Fails."""
        raise RuntimeError("kaput")

    ctx = ToolContext(work_dir=".")
    assert await boom.ainvoke({}, ctx) == "Error executing boom: kaput"
    assert (await boom.ainvoke({"x": 1}, ctx)).startswith("Error: bad arguments for boom")


async def test_async_tool():
    @tool(name="Echo", description="Echo back.")
    async def echo(text: str) -> str:
        return text

    assert echo.name == "Echo"
    assert await echo.ainvoke({"text": "hi"}, ToolContext(work_dir=".")) == "hi"


async def test_registry():
    reg = ToolRegistry([sample])
    with pytest.raises(ValueError):
        reg.add(sample)
    ctx = ToolContext(work_dir=".")
    assert await reg.dispatch("nope", {}, ctx) == "Error: unknown tool 'nope'"
    schema = reg.openai_schemas()[0]
    assert schema["type"] == "function" and schema["name"] == "sample"


async def test_builtin_tools(tmp_path):
    ctx = ToolContext(work_dir=str(tmp_path), env={"GREETING": "hello"})
    t = BUILTIN_TOOLS
    assert (await t["Write"].ainvoke({"file_path": "a/b.txt", "content": "x\ny\nneedle\n"}, ctx)
            ).startswith("Wrote")
    assert await t["Read"].ainvoke({"file_path": "a/b.txt", "offset": 2, "limit": 1}, ctx) == "2\ty"
    assert "b.txt:3:needle" in await t["Grep"].ainvoke({"pattern": "needle"}, ctx)
    assert (await t["Glob"].ainvoke({"pattern": "**/*.txt"}, ctx)).endswith("a/b.txt")
    assert (await t["Bash"].ainvoke({"command": "echo $GREETING"}, ctx)).strip() == "hello"
    assert "[exit code: 3]" in await t["Bash"].ainvoke({"command": "exit 3"}, ctx)
    assert all(tool.claude_native for tool in t.values())
