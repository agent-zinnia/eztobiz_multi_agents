import os
from unittest.mock import patch
from langgraph.pregel import Pregel


def test_graph_is_compiled() -> None:
    """Test that the graph is properly compiled"""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
        from agent.math_agent import graph
        assert isinstance(graph, Pregel)
    
def test_graph_has_nodes() -> None:
    """Test that the graph has expected nodes"""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
        from agent.math_agent import graph
        node_names = list(graph.nodes.keys())
        assert "assistant" in node_names
        assert "tools" in node_names
