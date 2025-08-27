import asyncio
import logging
from typing import Dict, Any, Optional
from .config import config
from .math_agent import MathAgent
from .question_agent import QuestionAgent


class DualAgentSystem:
    """System that coordinates math and question agents"""
    
    def __init__(self, platform_url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize the dual agent system
        
        Args:
            platform_url: URL of the LangGraph Platform deployment (defaults to config)
            api_key: API key for authentication (defaults to config)
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize the specialized agents
        self.math_agent = MathAgent(platform_url=platform_url, api_key=api_key)
        self.question_agent = QuestionAgent()
    
    async def run_math_agent(self, query: str, thread_id: str = None) -> Dict[str, Any]:
        """Run the math agent with the given query
        
        Args:
            query: The user's mathematical question/request
            thread_id: Optional existing thread ID to reuse. If None, creates a new thread.
            
        Returns:
            Dictionary containing the result from the math agent
        """
        return await self.math_agent.solve_math_problem(query, thread_id)
    
    async def generate_question_with_question_agent(self, math_result: str) -> str:
        """Use the question agent to generate a question based on the math result
        
        Args:
            math_result: Result from the math agent (ONLY the answer, not the original question)
            
        Returns:
            Generated question from the question agent
        """
        try:
            return await self.question_agent.generate_question(math_result)
        except Exception as e:
            self.logger.error(f"Error in question agent: {str(e)}", exc_info=True)
            return f"Error in question agent: {str(e)}"
    

    
    async def run_dual_agent_workflow(self, user_query: str, question_rounds: int = 1) -> Dict[str, Any]:
        """Run the complete dual agent workflow: Math -> Question (configurable times)
        
        Args:
            user_query: The user's original question
            question_rounds: Number of question rounds to run (default: 1)
            
        Returns:
            Dictionary containing results from all steps
        """
        print(f"🔄 Starting dual agent workflow for query: '{user_query}'")
        
        # Step 1: Run the first math agent
        print("📊 Step 1: Running first math agent...")
        first_result = await self.run_math_agent(user_query)
        
        if not first_result["success"]:
            print(f"❌ First math agent failed: {first_result['error']}")
            return {
                "error": f"First math agent failed: {first_result['error']}",
                "step1_math_result": None,
                "question_rounds": []
            }
        
        print(f"✅ Step 1 completed. Result: {first_result['result']}")
        
        # Initialize variables for multiple question rounds
        current_result = first_result["result"]
        thread_id = first_result["thread_id"]
        rounds_results = []
        
        # Run question agent for specified number of rounds
        for i in range(question_rounds):
            round_num = i + 2  # Starting from step 2
            print(f"🤔 Step {round_num}: Running question agent (round {i + 1}/{question_rounds}) to generate a follow-up question...")
            
            # Generate question based on current result
            generated_question = await self.generate_question_with_question_agent(current_result)
            
            if generated_question.startswith("Error"):
                print(f"❌ Question agent round {i + 1} failed: {generated_question}")
                rounds_results.append({
                    "round": i + 1,
                    "generated_question": None,
                    "answer": None,
                    "error": generated_question
                })
                break
            
            print(f"✅ Step {round_num} completed. Generated question: {generated_question}")
            
            # Answer the generated question
            answer_step = round_num + 3  # Math agent steps are 3, 6, 9
            print(f"🔢 Step {answer_step}: Reusing math agent to answer the generated question (round {i + 1}/{question_rounds})...")
            answer_result = await self.run_math_agent(generated_question, thread_id)
            
            if not answer_result["success"]:
                print(f"❌ Math agent round {i + 1} failed: {answer_result['error']}")
                rounds_results.append({
                    "round": i + 1,
                    "generated_question": generated_question,
                    "answer": None,
                    "error": answer_result["error"]
                })
                break
            
            print(f"✅ Step {answer_step} completed. Answer: {answer_result['result']}")
            
            # Store this round's results
            rounds_results.append({
                "round": i + 1,
                "generated_question": generated_question,
                "answer": answer_result["result"],
                "error": None
            })
            
            # Update current result for next round
            current_result = answer_result["result"]
        
        print("🎉 Dual agent workflow completed successfully!")
        
        return {
            "original_query": user_query,
            "step1_math_result": first_result["result"],
            "question_rounds": rounds_results,
            "workflow_metadata": {
                "step1_thread_id": first_result.get("thread_id"),
                "step1_run_id": first_result.get("run_id"),
                "total_question_rounds": len(rounds_results)
            }
        }
    



# Convenience function for easy usage
async def run_dual_agents(query: str, 
                         platform_url: str = None,
                         api_key: str = None,
                         question_rounds: int = 1) -> Dict[str, Any]:
    """Convenience function to run the dual agent system using LangGraph Platform
    
    Args:
        query: User's question
        platform_url: URL of the LangGraph Platform deployment (defaults to config)
        api_key: API key for authentication (defaults to config)
        question_rounds: Number of question rounds to run (default: 1)
        
    Returns:
        Results from both steps (Math -> Question)
    """
    system = DualAgentSystem(platform_url=platform_url, api_key=api_key)
    return await system.run_dual_agent_workflow(query, question_rounds)


# Example usage
if __name__ == "__main__":
    # Configure logging for example usage - file only, no console output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dual_agent_system.log')
        ]
    )
    
    async def main():
        logger = logging.getLogger(__name__)
        
        # Example query
        query = "What is 15 + 27, and then multiply the result by 3?"
        logger.info(f"Starting dual agent system with query: {query}")
        
        try:
            # Run the dual agent system
            result = await run_dual_agents(query)
            
            print("\n" + "="*60)
            print("DUAL AGENT SYSTEM RESULTS")
            print("="*60)
            print(f"Original Query: {result['original_query']}")
            print(f"\nStep 1 - Math Agent Result: {result['step1_math_result']}")
            
            # Display results for each question round
            for round_data in result.get('question_rounds', []):
                round_num = round_data['round']
                print(f"\nRound {round_num} - Generated Question: {round_data['generated_question']}")
                print(f"Round {round_num} - Answer: {round_data['answer']}")
            
            # Display summary
            total_rounds = len(result.get('question_rounds', []))
            print(f"\nSummary: Completed {total_rounds} question rounds")
            logger.info(f"Successfully completed dual agent workflow with {total_rounds} rounds")
            
        except Exception as e:
            logger.error(f"Failed to run dual agent system: {str(e)}", exc_info=True)
            print(f"Error: {str(e)}")
        
    # Run the example
    asyncio.run(main())
