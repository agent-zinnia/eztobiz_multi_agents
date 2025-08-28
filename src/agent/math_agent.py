import http.client
import json
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Union
from .config import config


class MathAgent:
    """Math agent that handles mathematical operations using LangGraph Platform API"""
    
    def __init__(self, platform_url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize the math agent
        
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
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def _http_connection(self):
        """Context manager for HTTP connections to ensure proper cleanup"""
        conn = None
        try:
            conn = http.client.HTTPSConnection(self.platform_url)
            yield conn
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
    
    def _make_http_request(self, method: str, path: str, payload: Optional[str] = None) -> Dict[str, Any]:
        """Make HTTP request with standardized error handling
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path
            payload: JSON payload for POST requests
            
        Returns:
            Dictionary with success status and response data or error message
        """
        try:
            with self._http_connection() as conn:
                if payload:
                    conn.request(method, path, payload, self.headers)
                else:
                    conn.request(method, path, headers=self.headers)
                
                response = conn.getresponse()
                data = response.read()
                
                if response.status == config.HTTP_SUCCESS_STATUS:
                    result = json.loads(data.decode("utf-8"))
                    return {
                        "success": True,
                        "status_code": response.status,
                        "data": result
                    }
                else:
                    error_msg = f"HTTP {response.status}: {data.decode('utf-8')}"
                    self.logger.error(f"HTTP request failed - {method} {path}: {error_msg}")
                    return {
                        "success": False,
                        "status_code": response.status,
                        "error": error_msg
                    }
                    
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response: {str(e)}"
            self.logger.error(f"JSON decode error - {method} {path}: {error_msg}")
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            self.logger.error(f"Request exception - {method} {path}: {error_msg}")
            return {"success": False, "error": error_msg}
    
    def _extract_last_ai_message(self, messages: List[Union[Dict[str, Any], Any]]) -> str:
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
        payload = json.dumps({
            "metadata": {},
            "if_exists": "raise"
        })
        
        result = self._make_http_request("POST", "/threads", payload)
        
        if result["success"]:
            data = result["data"]
            return {
                "success": True,
                "thread_id": data.get("thread_id"),
                "response": data
            }
        else:
            return {
                "success": False,
                "error": f"Failed to create thread: {result['error']}"
            }
    
    def _get_thread_state(self, thread_id: str) -> Dict[str, Any]:
        """Get the current state of a thread
        
        Args:
            thread_id: The thread ID to get state for
            
        Returns:
            Dictionary containing the thread state
        """
        result = self._make_http_request("GET", f"/threads/{thread_id}/state")
        
        if result["success"]:
            return {
                "success": True,
                "result": result["data"]
            }
        else:
            return {
                "success": False,
                "error": f"Failed to get thread state: {result['error']}"
            }

    def _run_on_thread_stream(self, thread_id: str, input_data: Dict[str, Any], assistant_id: str = "agent") -> Dict[str, Any]:
        """Run an agent on a specific thread with streaming
        
        Args:
            thread_id: The thread ID to run on
            input_data: The input data for the agent
            assistant_id: The assistant/graph ID to run
            
        Returns:
            Dictionary containing the run result and final messages
        """
        payload = json.dumps({
            "input": input_data,
            "assistant_id": assistant_id,
            "stream_mode": "updates"
        })
        
        try:
            with self._http_connection() as conn:
                conn.request("POST", f"/threads/{thread_id}/runs/stream", payload, self.headers)
                response = conn.getresponse()
                
                if response.status != config.HTTP_SUCCESS_STATUS:
                    data = response.read()
                    error_msg = f"Stream request failed: {response.status} - {data.decode('utf-8')}"
                    self.logger.error(f"Stream failure for thread {thread_id}: {error_msg}")
                    return {"success": False, "error": error_msg}
                
                stream_data = self._process_stream_response(response)
                
                if stream_data:
                    return {
                        "success": True,
                        "result": stream_data[-1],
                        "stream_data": stream_data
                    }
                else:
                    return {"success": False, "error": "No data received from stream"}
                    
        except Exception as e:
            error_msg = f"Exception running stream on thread: {str(e)}"
            self.logger.error(f"Stream exception for thread {thread_id}: {error_msg}")
            return {"success": False, "error": error_msg}
    
    def _process_stream_response(self, response) -> List[Dict[str, Any]]:
        """Process streaming response and extract data items
        
        Args:
            response: HTTP response object with streaming data
            
        Returns:
            List of parsed stream data items
        """
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
                if line.startswith(config.STREAM_DATA_PREFIX):
                    try:
                        data_str = line[len(config.STREAM_DATA_PREFIX):]  # Remove 'data: ' prefix
                        if data_str and data_str != config.STREAM_DONE_MARKER:
                            stream_item = json.loads(data_str)
                            stream_data.append(stream_item)
                    except json.JSONDecodeError:
                        continue
        
        return stream_data

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
        
        self.logger.info(f"Stream failed for thread {thread_id}, falling back to polling: {stream_result['error']}")
        
        # Fallback to polling approach
        return self._run_on_thread_polling(thread_id, input_data, assistant_id)
    
    def _run_on_thread_polling(self, thread_id: str, input_data: Dict[str, Any], assistant_id: str) -> Dict[str, Any]:
        """Run agent with polling approach as fallback
        
        Args:
            thread_id: The thread ID to run on
            input_data: The input data for the agent
            assistant_id: The assistant/graph ID to run
            
        Returns:
            Dictionary containing the run result and final messages
        """
        payload = json.dumps({
            "input": input_data,
            "assistant_id": assistant_id
        })
        
        result = self._make_http_request("POST", f"/threads/{thread_id}/runs", payload)
        
        if not result["success"]:
            return {
                "success": False,
                "error": f"Failed to run on thread: {result['error']}"
            }
        
        run_data = result["data"]
        run_id = run_data.get("run_id")
        
        if not run_id:
            return {
                "success": False,
                "error": "No run_id returned from the API"
            }
        
        # Wait for completion by polling thread state
        return self._wait_for_completion(thread_id, run_id, run_data)
    
    def _wait_for_completion(self, thread_id: str, run_id: str, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Wait for thread execution to complete using polling
        
        Args:
            thread_id: The thread ID
            run_id: The run ID
            run_data: Initial run data
            
        Returns:
            Dictionary containing the final result
        """
        import time
        
        max_wait = config.AGENT_MAX_WAIT_TIME
        check_interval = config.AGENT_CHECK_INTERVAL
        waited = 0
        
        while waited < max_wait:
            state_result = self._get_thread_state(thread_id)
            if state_result["success"]:
                state_data = state_result["result"]
                # Check if there are any remaining tasks
                if "next" in state_data and state_data["next"]:
                    self.logger.debug(f"Thread {thread_id} still processing, next steps: {state_data['next']}")
                    time.sleep(check_interval)
                    waited += check_interval
                    continue
                
                # No more tasks, execution is complete
                return {
                    "success": True,
                    "result": state_data,
                    "run_id": run_id,
                    "response_data": json.dumps(run_data)
                }
            
            time.sleep(check_interval)
            waited += check_interval
        
        # Timeout - return what we have
        self.logger.warning(f"Timeout waiting for completion of thread {thread_id}")
        final_state = self._get_thread_state(thread_id)
        return {
            "success": True,
            "result": final_state.get("result", run_data),
            "run_id": run_id,
            "response_data": json.dumps(run_data),
            "warning": "Timeout waiting for completion"
        }
    
    async def solve_math_problem(self, query: str, thread_id: str = None) -> Dict[str, Any]:
        """Solve a mathematical problem using LangGraph Platform
        
        Args:
            query: The mathematical question/request
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
