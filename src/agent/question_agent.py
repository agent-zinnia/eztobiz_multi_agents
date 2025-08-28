import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from typing import TypedDict, List, Dict, Any
from .config import config


class QuestionAgentState(TypedDict):
    """State for the question agent"""
    math_result: str
    messages: List


class QuestionAgent:
    """Question agent that generates follow-up questions based on mathematical results"""
    
    def __init__(self, model_name: str = None):
        """Initialize the question agent
        
        Args:
            model_name: Name of the language model to use (defaults to config)
        """
        self.model_name = model_name or config.QUESTION_AGENT_MODEL
        self.llm = ChatOpenAI(model=self.model_name)
        self.logger = logging.getLogger(__name__)
        
        # System message for question agent
        self.system_message = SystemMessage(
            content="""You are a critical thinking assistant that analyzes mathematical results. 
            Your role is to:
            1. Review ONLY the mathematical result provided (not any original question or context)
            2. Generate ONE specific follow-up mathematical question based SOLELY on the result
            3. Focus on expanding or exploring the result further through mathematical operations
            4. Create questions that can be answered using mathematical calculations
            
            IMPORTANT: Base your question ONLY on the mathematical result provided, not on any original question or context.
            Generate only mathematical questions that can be computed, not analytical or explanatory questions."""
        )
    
    async def generate_question(self, math_result: str) -> str:
        """Generate a follow-up question based on a mathematical result
        
        Args:
            math_result: The mathematical result to base the question on
            
        Returns:
            Generated question as a string
        """
        if not math_result:
            self.logger.warning("No math result provided to question agent")
            return "Error: No mathematical result provided"
        
        self.logger.info(f"Generating question based on result: {math_result}")
        
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
        
        try:
            response = self.llm.invoke([
                self.system_message,
                HumanMessage(content=analysis_prompt)
            ])
            
            self.logger.info(f"Successfully generated question: {response.content}")
            return response.content
            
        except Exception as e:
            self.logger.error(f"Failed to generate question: {str(e)}", exc_info=True)
            return f"Error generating question: {str(e)}"


# Legacy support: Define LLM for backward compatibility
question_llm = ChatOpenAI(model=config.QUESTION_AGENT_MODEL)

# Legacy support: System message for backward compatibility
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


def question_agent_node(state: QuestionAgentState) -> Dict[str, Any]:
    """Legacy node function for backward compatibility
    
    Args:
        state: The current state containing the math result
        
    Returns:
        Updated state with the generated question message
    """
    logger = logging.getLogger(__name__)
    
    # Support both old and new state key names
    math_result = state.get('math_result', '') or state.get('first_agent_result', '')
    
    if not math_result:
        logger.warning("No math result provided to question agent")
        return {"messages": state.get("messages", [])}
    
    # Use the new QuestionAgent class
    agent = QuestionAgent()
    
    try:
        # Generate question using the new method
        import asyncio
        loop = asyncio.get_event_loop()
        generated_question = loop.run_until_complete(agent.generate_question(math_result))
        
        # Create response message
        response = HumanMessage(content=generated_question)
        
        return {
            "messages": state.get("messages", []) + [response]
        }
        
    except Exception as e:
        logger.error(f"Failed to generate question: {str(e)}", exc_info=True)
        # Return error message in the expected format
        error_response = HumanMessage(content=f"Error generating question: {str(e)}")
        return {
            "messages": state.get("messages", []) + [error_response]
        }


# Build question agent graph
question_builder = StateGraph(QuestionAgentState)
question_builder.add_node("question_agent", question_agent_node)
question_builder.add_edge(START, "question_agent")

# Compile question agent graph
question_agent_graph = question_builder.compile()
