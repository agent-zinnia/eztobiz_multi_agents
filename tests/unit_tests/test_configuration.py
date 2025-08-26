from langgraph.pregel import Pregel

from agent.graph import graph


def test_graph_is_compiled() -> None:
    """Test that the graph is properly compiled"""
    assert isinstance(graph, Pregel)
    
def test_graph_has_nodes() -> None:
    """Test that the graph has expected nodes"""
    node_names = list(graph.nodes.keys())
    assert "assistant" in node_names
    assert "tools" in node_names
