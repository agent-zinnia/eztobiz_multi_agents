from langgraph_sdk import get_client
from langchain_core.messages import HumanMessage
import asyncio
from typing import Dict, Any


class DualAgentSystem:
    """System that coordinates two agents using langgraph_sdk"""
    
    def __init__(self, url: str = "http://127.0.0.1:2024"):
        """Initialize the dual agent system
        
        Args:
            url: URL of the local langgraph server
        """
        self.url = url
        self.client = get_client(url=url)
    
    def _extract_last_ai_message(self, messages: list) -> str:
        """Extract the last AI message from a list of messages
        
        Args:
            messages: List of messages from the agent state
            
        Returns:
            The content of the last AI message, or a fallback message
        """
        # Find the last AI message
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == "ai":
                return msg.content
            elif isinstance(msg, dict) and msg.get("type") == "ai":
                return msg.get("content", "")
        
        # Fallback: get any message content
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, 'content'):
                return last_msg.content
            elif isinstance(last_msg, dict):
                return last_msg.get("content", str(last_msg))
            else:
                return str(last_msg)
        
        return "No result generated"
    
    async def run_math_agent(self, query: str, thread_id: str = None) -> Dict[str, Any]:
        """Run the math agent with the given query
        
        Args:
            query: The user's question/request
            thread_id: Optional existing thread ID to reuse. If None, creates a new thread.
            
        Returns:
            Dictionary containing the result from the math agent
        """
        try:
            # Create a new thread if none provided, otherwise use existing one
            if thread_id is None:
                thread = await self.client.threads.create()
                thread_id = thread["thread_id"]
                is_new_thread = True
            else:
                is_new_thread = False
            
            # Run the math agent (using the graph name from langgraph.json)
            run = await self.client.runs.create(
                thread_id=thread_id,
                assistant_id="math_agent",  # This matches the graph name in langgraph.json
                input={"messages": [HumanMessage(content=query)]}
            )
            
            # Wait for completion
            await self.client.runs.join(thread_id=thread_id, run_id=run["run_id"])
            
            # Get the final state/messages
            state = await self.client.threads.get_state(thread_id=thread_id)
            
            # Extract messages from the state
            messages = state.get("values", {}).get("messages", [])
            last_ai_message = self._extract_last_ai_message(messages)
            
            return {
                "success": True,
                "result": last_ai_message,
                "thread_id": thread_id,
                "run_id": run["run_id"],
                "is_new_thread": is_new_thread
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": None,
                "thread_id": thread_id,
                "is_new_thread": thread_id is None
            }
    
    async def generate_question_with_question_agent(self, first_agent_result: str) -> str:
        """Use the question agent via langgraph server to generate a question based on the first agent's result
        
        Args:
            first_agent_result: Result from the first agent (ONLY the answer, not the original question)
            
        Returns:
            Generated question from the question agent
        """
        try:
            # Create a NEW thread specifically for the question agent to avoid context leakage
            thread = await self.client.threads.create()
            thread_id = thread["thread_id"]
            
            # Run the question agent via server - ONLY pass the first agent's result
            # Create a completely isolated input to prevent any context leakage
            isolated_prompt = f"Given this mathematical result: {first_agent_result}, generate a follow-up mathematical question."
            
            run = await self.client.runs.create(
                thread_id=thread_id,
                assistant_id="question_agent",  # This matches the graph name in langgraph.json
                input={
                    "first_agent_result": first_agent_result,  # Only the math result
                    "messages": [HumanMessage(content=isolated_prompt)]  # Direct prompt without context
                }
            )
            
            # Wait for completion
            await self.client.runs.join(thread_id=thread_id, run_id=run["run_id"])
            
            # Get the final state/messages
            state = await self.client.threads.get_state(thread_id=thread_id)
            
            # Extract messages from the state
            messages = state.get("values", {}).get("messages", [])
            generated_question = self._extract_last_ai_message(messages)
            
            return generated_question if generated_question != "No result generated" else "No question generated by question agent"
                
        except Exception as e:
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
async def run_dual_agents(query: str, server_url: str = "http://127.0.0.1:2024", question_rounds: int = 1) -> Dict[str, Any]:
    """Convenience function to run the dual agent system
    
    Args:
        query: User's question
        server_url: URL of the langgraph server
        question_rounds: Number of question rounds to run (default: 1)
        
    Returns:
        Results from both steps (Math -> Question)
    """
    system = DualAgentSystem(url=server_url)
    return await system.run_dual_agent_workflow(query, question_rounds)


# Example usage
if __name__ == "__main__":
    async def main():
        # Example query
        query = "What is 15 + 27, and then multiply the result by 3?"
        
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
        
    # Run the example
    asyncio.run(main())
