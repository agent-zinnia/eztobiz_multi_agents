import pytest

from agent import graph

pytestmark = pytest.mark.anyio


@pytest.mark.langsmith
async def test_agent_simple_math() -> None:
    inputs = {"messages": [{"type": "human", "content": "What is 2 + 3?"}]}
    res = await graph.ainvoke(inputs)
    assert res is not None
    assert "messages" in res
    assert len(res["messages"]) > 0
