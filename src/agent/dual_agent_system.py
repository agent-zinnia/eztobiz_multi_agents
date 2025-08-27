import http.client
import json
import asyncio
from typing import Dict, Any
from .config import config


class DualAgentSystem:
    """System that coordinates two agents using LangGraph Platform API"""
    
    def __init__(self, platform_url: str = None, api_key: str = None):
        """Initialize the dual agent system
        
        Args:
            platform_url: URL of the LangGraph Platform deployment (defaults to config)
            api_key: API key for authentication (defaults to config)
        """
        self.platform_url = platform_url or config.LANGGRAPH_PLATFORM_URL
        self.api_key = api_key or config.LANGGRAPH_API_KEY
        self.headers = {
            'Content-Type': config.HTTP_CONTENT_TYPE,
            'x-api-key': self.api_key
        }
    
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
                content = msg.get("content", "")
                return content
        
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
    
    def _create_thread(self) -> Dict[str, Any]:
        """Create a new thread on LangGraph Platform
        
        Returns:
            Dictionary containing thread_id and creation status
        """
        try:
            conn = http.client.HTTPSConnection(self.platform_url)
            
            # Create thread payload - simplified version without complex metadata
            payload = json.dumps({
                "metadata": {},
                "if_exists": "raise"
            })
            
            conn.request("POST", "/threads", payload, self.headers)
            response = conn.getresponse()
            data = response.read()
            
            if response.status == 200:
                result = json.loads(data.decode("utf-8"))
                return {
                    "success": True,
                    "thread_id": result.get("thread_id"),
                    "response": result
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to create thread: {response.status} - {data.decode('utf-8')}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Exception creating thread: {str(e)}"
            }
        finally:
            try:
                conn.close()
            except:
                pass
    
    def _get_thread_state(self, thread_id: str) -> Dict[str, Any]:
        """Get the current state of a thread
        
        Args:
            thread_id: The thread ID to get state for
            
        Returns:
            Dictionary containing the thread state
        """
        try:
            conn = http.client.HTTPSConnection(self.platform_url)
            
            conn.request("GET", f"/threads/{thread_id}/state", headers=self.headers)
            response = conn.getresponse()
            data = response.read()
            
            if response.status == 200:
                result = json.loads(data.decode("utf-8"))
                return {
                    "success": True,
                    "result": result
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to get thread state: {response.status} - {data.decode('utf-8')}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Exception getting thread state: {str(e)}"
            }
        finally:
            try:
                conn.close()
            except:
                pass

    def _run_on_thread_stream(self, thread_id: str, input_data: Dict[str, Any], assistant_id: str = "agent") -> Dict[str, Any]:
        """Run an agent on a specific thread with streaming
        
        Args:
            thread_id: The thread ID to run on
            input_data: The input data for the agent
            assistant_id: The assistant/graph ID to run
            
        Returns:
            Dictionary containing the run result and final messages
        """
        try:
            conn = http.client.HTTPSConnection(self.platform_url)
            
            # Create run payload with streaming
            payload = json.dumps({
                "input": input_data,
                "assistant_id": assistant_id,
                "stream_mode": "updates"
            })
            
            conn.request("POST", f"/threads/{thread_id}/runs/stream", payload, self.headers)
            response = conn.getresponse()
            
            if response.status != 200:
                data = response.read()
                return {
                    "success": False,
                    "error": f"Failed to run stream on thread: {response.status} - {data.decode('utf-8')}"
                }
            
            # Read the stream data
            stream_data = []
            buffer = ""
            
            while True:
                chunk = response.read(config.AGENT_STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                    
                buffer += chunk.decode('utf-8')
                lines = buffer.split('\n')
                buffer = lines[-1]  # Keep incomplete line in buffer
                
                for line in lines[:-1]:
                    line = line.strip()
                    if line.startswith('data: '):
                        try:
                            data_str = line[6:]  # Remove 'data: ' prefix
                            if data_str and data_str != '[DONE]':
                                stream_item = json.loads(data_str)
                                stream_data.append(stream_item)
                                # print(f"DEBUG: Stream item: {stream_item}")  # Comment out for cleaner output
                        except json.JSONDecodeError:
                            continue
            
            if stream_data:
                # Get the last stream item which should contain the final state
                final_state = stream_data[-1]
                return {
                    "success": True,
                    "result": final_state,
                    "stream_data": stream_data
                }
            else:
                return {
                    "success": False,
                    "error": "No data received from stream"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Exception running stream on thread: {str(e)}"
            }
        finally:
            try:
                conn.close()
            except:
                pass

    def _run_on_thread(self, thread_id: str, input_data: Dict[str, Any], assistant_id: str = "agent") -> Dict[str, Any]:
        """Run an agent on a specific thread and wait for completion
        
        Args:
            thread_id: The thread ID to run on
            input_data: The input data for the agent
            assistant_id: The assistant/graph ID to run
            
        Returns:
            Dictionary containing the run result and final messages
        """
        # First try streaming approach
        stream_result = self._run_on_thread_stream(thread_id, input_data, assistant_id)
        if stream_result["success"]:
            return stream_result
        
        # print(f"DEBUG: Stream failed, falling back to polling: {stream_result['error']}")  # Comment out for cleaner output
        
        # Fallback to polling approach
        try:
            conn = http.client.HTTPSConnection(self.platform_url)
            
            # Create run payload with required assistant_id
            payload = json.dumps({
                "input": input_data,
                "assistant_id": assistant_id
            })
            
            conn.request("POST", f"/threads/{thread_id}/runs", payload, self.headers)
            response = conn.getresponse()
            data = response.read()
            
            if response.status != 200:
                return {
                    "success": False,
                    "error": f"Failed to run on thread: {response.status} - {data.decode('utf-8')}"
                }
            
            run_result = json.loads(data.decode("utf-8"))
            run_id = run_result.get("run_id")
            
            if not run_id:
                return {
                    "success": False,
                    "error": "No run_id returned from the API"
                }
            
            # Wait for completion by checking thread state
            import time
            max_wait = config.AGENT_MAX_WAIT_TIME  # Maximum wait time in seconds
            check_interval = config.AGENT_CHECK_INTERVAL  # Check interval in seconds
            waited = 0
            
            while waited < max_wait:
                state_result = self._get_thread_state(thread_id)
                if state_result["success"]:
                    state_data = state_result["result"]
                    # Check if there are any remaining tasks
                    if "next" in state_data and state_data["next"]:
                        # print(f"DEBUG: Still processing, next steps: {state_data['next']}")  # Comment out for cleaner output
                        time.sleep(check_interval)
                        waited += check_interval
                        continue
                    
                    # No more tasks, execution is complete
                    return {
                        "success": True,
                        "result": state_data,
                        "run_id": run_id,
                        "response_data": data.decode("utf-8")
                    }
                
                time.sleep(check_interval)
                waited += check_interval
            
            # Timeout - return what we have
            final_state = self._get_thread_state(thread_id)
            return {
                "success": True,
                "result": final_state.get("result", run_result),
                "run_id": run_id,
                "response_data": data.decode("utf-8"),
                "warning": "Timeout waiting for completion"
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Exception running on thread: {str(e)}"
            }
        finally:
            try:
                conn.close()
            except:
                pass
    
    async def run_math_agent(self, query: str, thread_id: str = None) -> Dict[str, Any]:
        """Run the math agent with the given query using LangGraph Platform
        
        Args:
            query: The user's question/request
            thread_id: Optional existing thread ID to reuse. If None, creates a new thread.
            
        Returns:
            Dictionary containing the result from the math agent
        """
        try:
            # Create a new thread if none provided
            if thread_id is None:
                thread_result = self._create_thread()
                if not thread_result["success"]:
                    return {
                        "success": False,
                        "error": f"Failed to create thread: {thread_result['error']}",
                        "result": None,
                        "thread_id": None,
                        "is_new_thread": True
                    }
                thread_id = thread_result["thread_id"]
                is_new_thread = True
            else:
                is_new_thread = False
            
            # Prepare input for the math agent
            input_data = {
                "messages": [
                    {
                        "type": "human",
                        "content": query
                    }
                ]
            }
            
            # Run the math agent on the thread
            run_result = self._run_on_thread(thread_id, input_data, assistant_id=config.MATH_AGENT_ASSISTANT_ID)
            
            if not run_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to run math agent: {run_result['error']}",
                    "result": None,
                    "thread_id": thread_id,
                    "is_new_thread": is_new_thread
                }
            
            # Extract the result from the response
            result_data = run_result["result"]
            # Debug information (can be removed for production)
            # print(f"DEBUG: Result data type: {type(result_data)}")
            # print(f"DEBUG: Result data keys: {result_data.keys() if isinstance(result_data, dict) else 'Not a dict'}")
            # print(f"DEBUG: Full result data: {json.dumps(result_data, indent=2) if isinstance(result_data, (dict, list)) else str(result_data)}")
            
            # Handle different response types
            if isinstance(result_data, dict):
                # Check for errors first
                if "error" in result_data:
                    error_msg = result_data.get("message", str(result_data))
                    last_ai_message = f"Platform error: {error_msg}"
                
                elif "values" in result_data:
                    # Get messages from the thread state
                    values = result_data["values"]
                    if "messages" in values:
                        messages = values["messages"]
                        last_ai_message = self._extract_last_ai_message(messages)
                    else:
                        last_ai_message = str(values)
                
                elif "assistant" in result_data and "messages" in result_data["assistant"]:
                    # Handle LangGraph Platform response format
                    messages = result_data["assistant"]["messages"]
                    last_ai_message = self._extract_last_ai_message(messages)
                
                else:
                    last_ai_message = str(result_data)
                    
            elif isinstance(result_data, list) and len(result_data) > 0:
                # Take the last item in the stream
                last_item = result_data[-1]
                if "messages" in last_item:
                    messages = last_item["messages"]
                    last_ai_message = self._extract_last_ai_message(messages)
                else:
                    last_ai_message = str(last_item)
            else:
                last_ai_message = str(result_data)
            
            return {
                "success": True,
                "result": last_ai_message,
                "thread_id": thread_id,
                "is_new_thread": is_new_thread,
                "full_response": result_data
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
        """Use the local question agent to generate a question based on the first agent's result
        
        Args:
            first_agent_result: Result from the first agent (ONLY the answer, not the original question)
            
        Returns:
            Generated question from the question agent
        """
        try:
            # Import the local question agent
            try:
                from agent.question_agent import question_agent_graph
            except ImportError:
                # Fallback for when running from different paths
                from question_agent import question_agent_graph
            
            # Create isolated input to prevent context leakage
            input_data = {
                "first_agent_result": first_agent_result,  # Only the math result
                "messages": []
            }
            
            # Run the local question agent
            result = await question_agent_graph.ainvoke(input_data)
            
            # Extract the generated question from the result
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    generated_question = last_message.content
                elif isinstance(last_message, dict):
                    generated_question = last_message.get('content', str(last_message))
                else:
                    generated_question = str(last_message)
            else:
                generated_question = "No question generated by question agent"
            
            return generated_question
                
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback_msg = traceback.format_exc()
            return f"Error in question agent: {error_msg}\nTraceback: {traceback_msg}"
    

    
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
