from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from typing import TypedDict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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
    1. Review ONLY the mathematical result provided (not any original question or context)
    2. Generate ONE specific follow-up mathematical question based SOLELY on the result
    3. Focus on expanding or exploring the result further through mathematical operations
    4. Create questions that can be answered using mathematical calculations
    
    IMPORTANT: Base your question ONLY on the mathematical result provided, not on any original question or context.
    Generate only mathematical questions that can be computed, not analytical or explanatory questions."""
)


def question_agent_node(state: QuestionAgentState):
    """Node that generates questions about the first agent's result ONLY"""
    
    # Only use the first_agent_result, ignore any other messages to prevent original question leakage
    math_result = state.get('first_agent_result', '')
    
    analysis_prompt = f"""
    You are given a mathematical result: {math_result}
    
    CRITICAL INSTRUCTIONS:
    1. You must NOT know what the original question was
    2. You only see this result: {math_result}
    3. Create a NEW mathematical question that uses this result as a starting point
    4. Do NOT reference any specific numbers from the original calculation
    5. Focus on mathematical operations that can be applied to this result
    
    Generate ONE specific follow-up mathematical question that explores this result further.
    
    Good examples:
    - "What would be the result if we divided this by 2?"
    - "What is the square of this number?"
    - "What would happen if we added 10 to this result?"
    
    BAD examples (DO NOT do this):
    - Questions that reference original numbers from the calculation
    - Questions about the original problem
    
    Output format: Return ONLY the question itself, without any additional text, explanation, or formatting.
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
