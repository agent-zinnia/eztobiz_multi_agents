from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, MessagesState
from typing import TypedDict, List


class QuestionAgentState(TypedDict):
    """State for the question agent"""
    original_query: str
    first_agent_result: str
    messages: List


# Define LLM for question agent
question_llm = ChatOpenAI(model="gpt-4o-mini")

# System message for question agent
question_sys_msg = SystemMessage(
    content="""You are a critical thinking assistant that analyzes results from other agents. 
    Your role is to:
    1. Review the result from the first agent
    2. Ask thoughtful follow-up questions about the result
    3. Identify potential issues, edge cases, or areas that need clarification
    4. Provide constructive feedback and suggestions for improvement
    
    Be thorough but constructive in your analysis."""
)


def question_agent_node(state: QuestionAgentState):
    """Node that generates questions about the first agent's result"""
    
    # Prepare the prompt for the question agent
    analysis_prompt = f"""
    Original Query: {state['original_query']}
    
    First Agent Result: {state['first_agent_result']}
    
    Please analyze this result and provide:
    only one thoughtful question about the result
    """
    
    # Get response from question agent
    response = question_llm.invoke([
        question_sys_msg,
        HumanMessage(content=analysis_prompt)
    ])
    
    return {
        "messages": state.get("messages", []) + [response]
    }


# Build question agent graph
question_builder = StateGraph(QuestionAgentState)
question_builder.add_node("question_agent", question_agent_node)
question_builder.add_edge(START, "question_agent")

# Compile question agent graph
question_agent_graph = question_builder.compile()
