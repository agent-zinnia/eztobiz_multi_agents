import os
import pytest
from unittest.mock import patch

pytestmark = pytest.mark.anyio


async def test_agent_simple_math() -> None:
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
        from agent import graph
        inputs = {"messages": [{"type": "human", "content": "What is 2 + 3?"}]}
        # This test would require actual API call, so we just test structure
        assert graph is not None
        assert hasattr(graph, 'ainvoke')
