#!/usr/bin/env python3
"""
Dual Agent System Runner - Single Query Mode Only

This script runs two agents in sequence via langgraph server:
1. Math Agent: Performs calculations using tools
2. Question Agent: Generates follow-up questions based on the result

Usage:
    python run_dual_agents.py                      # Interactive input mode
    python run_dual_agents.py "your math question" # Command line argument mode
    
REQUIREMENTS:
- Make sure the langgraph server is running: langgraph dev
- Server should be available at http://127.0.0.1:2024
- Both math_agent and question_agent graphs must be deployed
"""

import asyncio
import sys
from dual_agent_system import run_dual_agents


def _print_server_requirements():
    """Print server requirements message"""
    print("Please make sure:")
    print("1. langgraph server is running: langgraph dev")
    print("2. Server is available at http://127.0.0.1:2024")
    print("3. Both math_agent and question_agent graphs are deployed")





async def single_query_mode(query: str):
    """Run a single query through the dual agent system"""
    print(f"🔄 Running dual agent system for: '{query}'")
    
    try:
        result = await run_dual_agents(query)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print("\n" + "="*60)
        print("DUAL AGENT SYSTEM RESULTS")
        print("="*60)
        print(f"Original Query: {result['original_query']}")
        
        print(f"\n📊 Step 1 - Math Agent Result:")
        print("-" * 30)
        print(result['step1_math_result'])
        
        print(f"\n🤔 Step 2 - Generated Question:")
        print("-" * 30)
        print(result['step2_generated_question'])
        
        print(f"\n🔢 Step 3 - Answer to Question:")
        print("-" * 30)
        print(result['step3_answer_to_question'])
        
    except Exception as e:
        print(f"❌ Error: {e}")
        _print_server_requirements()


def main():
    """Main function to handle command line arguments or user input"""
    if len(sys.argv) > 1:
        # Single query mode with command line argument
        query = " ".join(sys.argv[1:])
        asyncio.run(single_query_mode(query))
    else:
        # Single query mode with user input
        print("🤖 Dual Agent System - Single Query Mode")
        print("=" * 45)
        print("This system uses two agents via langgraph server:")
        print("1. Math Agent: Performs calculations using tools")
        print("2. Question Agent: Generates follow-up questions")
        print("\n⚠️  REQUIREMENTS:")
        print("- langgraph server must be running: langgraph dev")
        print("- Server at http://127.0.0.1:2024")
        print("- Both math_agent and question_agent graphs deployed")
        print("-" * 45)
        
        try:
            query = input("\n💬 Enter your math question: ").strip()
            
            if not query:
                print("❌ Please enter a valid question.")
                return
            
            asyncio.run(single_query_mode(query))
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            _print_server_requirements()


if __name__ == "__main__":
    main()
