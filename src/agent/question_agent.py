from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from typing import TypedDict, List


class QuestionAgentState(TypedDict):
    """State for the question agent"""
    first_agent_result: str
    messages: List


# Define LLM for question agent
question_llm = ChatOpenAI(model="gpt-4o-mini")

# System message for question agent
question_sys_msg = SystemMessage(
    content="""You are a critical thinking assistant that analyzes mathematical results. 
    Your role is to:
    1. Review the mathematical result provided
    2. Generate ONE specific follow-up mathematical question based on the result
    3. Focus on expanding or exploring the result further through mathematical operations
    4. Create questions that can be answered using mathematical calculations
    
    Generate only mathematical questions that can be computed, not analytical or explanatory questions."""
)


def question_agent_node(state: QuestionAgentState):
    """Node that generates questions about the first agent's result"""
    
    analysis_prompt = f"""
    Math Result: {state['first_agent_result']}
    
    Please analyze this mathematical result and generate ONE specific follow-up mathematical question that can be asked to expand or explore this result further.
    
    Output format: Return ONLY the question itself, without any additional text, explanation, or formatting.
    
    Example: "What would be the result if we divided this by 2?"
    """
    
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
