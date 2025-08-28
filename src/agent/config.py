"""
Configuration module for the multi-agent system.
Centralizes all configuration values and environment variable handling.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class AgentConfig:
    """Configuration class for the multi-agent system"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # LangGraph Platform Configuration
    LANGGRAPH_PLATFORM_URL: str = "test-98b04f86355f58019a022f94c6ed1032.us.langgraph.app"
    LANGGRAPH_API_KEY: str = os.getenv("LANGGRAPH_API_KEY")
    
    # Math Agent Configuration
    MATH_AGENT_ASSISTANT_ID: str = "math_agent"
    MATH_AGENT_MODEL: str = "gpt-4o-mini"  # Model for standalone math operations
    
    # Question Agent Configuration  
    QUESTION_AGENT_MODEL: str = "gpt-4o-mini"
    
    # Local LangGraph Server Configuration
    LANGGRAPH_LOCAL_SERVER_URL: str = os.getenv("LANGGRAPH_LOCAL_SERVER_URL", "http://127.0.0.1:2024")
    
    # Agent Timeouts and Limits
    AGENT_MAX_WAIT_TIME: int = 30
    AGENT_CHECK_INTERVAL: int = 2
    AGENT_STREAM_CHUNK_SIZE: int = 1024
    
    # HTTP Configuration
    HTTP_CONTENT_TYPE: str = "application/json"
    HTTP_SUCCESS_STATUS: int = 200
    
    # Stream Processing Configuration
    STREAM_DATA_PREFIX: str = "data: "
    STREAM_DONE_MARKER: str = "[DONE]"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration values are set"""
        required_vars = [
            "OPENAI_API_KEY",
            "LANGGRAPH_API_KEY"
        ]
        
        missing_vars = []
        for var in required_vars:
            value = getattr(cls, var, "")
            if not value or value == "your_openai_api_key_here":
                missing_vars.append(var)
        
        if missing_vars:
            print(f"⚠️  Missing required environment variables: {', '.join(missing_vars)}")
            return False
        
        return True
    
    @classmethod
    def print_config_summary(cls) -> None:
        """Print a summary of current configuration (hiding sensitive values)"""
        print("🔧 Current Configuration:")
        print(f"  Platform URL: {cls.LANGGRAPH_PLATFORM_URL}")
        print(f"  Local Server: {cls.LANGGRAPH_LOCAL_SERVER_URL}")
        print(f"  Question Model: {cls.QUESTION_AGENT_MODEL}")
        print(f"  Math Agent ID: {cls.MATH_AGENT_ASSISTANT_ID}")
        print(f"  Math Agent Model: {cls.MATH_AGENT_MODEL}")
        print(f"  Max Wait Time: {cls.AGENT_MAX_WAIT_TIME}s")
        print(f"  Check Interval: {cls.AGENT_CHECK_INTERVAL}s")
        print(f"  Stream Chunk Size: {cls.AGENT_STREAM_CHUNK_SIZE}")
        
        # Check if sensitive values are configured
        has_openai_key = bool(cls.OPENAI_API_KEY and cls.OPENAI_API_KEY != "your_openai_api_key_here")
        has_api_key = bool(cls.LANGGRAPH_API_KEY and not cls.LANGGRAPH_API_KEY.startswith("your_"))
        
        print(f"  OpenAI API Key: {'✅ Configured' if has_openai_key else '❌ Not configured'}")
        print(f"  LangGraph API Key: {'✅ Configured' if has_api_key else '❌ Not configured'}")


# Global configuration instance
config = AgentConfig()

# Convenience functions for backward compatibility
def get_platform_url() -> str:
    """Get the LangGraph platform URL"""
    return config.LANGGRAPH_PLATFORM_URL

def get_api_key() -> str:
    """Get the LangGraph API key"""
    return config.LANGGRAPH_API_KEY

def get_question_model() -> str:
    """Get the question agent model name"""
    return config.QUESTION_AGENT_MODEL

def get_local_server_url() -> str:
    """Get the local LangGraph server URL"""
    return config.LANGGRAPH_LOCAL_SERVER_URL
