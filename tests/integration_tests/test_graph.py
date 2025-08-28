import os
import pytest
from unittest.mock import patch

pytestmark = pytest.mark.anyio


async def test_question_agent_structure() -> None:
    """Test question agent graph structure"""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
        from agent.question_agent import question_agent_graph
        inputs = {"first_agent_result": "42", "messages": []}
        # This test would require actual API call, so we just test structure
        assert question_agent_graph is not None
        assert hasattr(question_agent_graph, 'ainvoke')


async def test_dual_agent_system_structure() -> None:
    """Test dual agent system structure"""
    from agent.dual_agent_system import DualAgentSystem
    system = DualAgentSystem()
    assert system is not None
    assert hasattr(system, 'run_math_agent')
    assert hasattr(system, 'generate_question_with_question_agent')
    assert hasattr(system, 'run_dual_agent_workflow')
