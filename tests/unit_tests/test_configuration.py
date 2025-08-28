import os
from unittest.mock import patch
from langgraph.pregel import Pregel


def test_question_agent_is_compiled() -> None:
    """Test that the question agent graph is properly compiled"""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
        from agent.question_agent import question_agent_graph
        assert isinstance(question_agent_graph, Pregel)
    
def test_question_agent_has_nodes() -> None:
    """Test that the question agent graph has expected nodes"""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
        from agent.question_agent import question_agent_graph
        node_names = list(question_agent_graph.nodes.keys())
        assert "generate_question" in node_names
