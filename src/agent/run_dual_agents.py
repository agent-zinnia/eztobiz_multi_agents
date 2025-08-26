#!/usr/bin/env python3
"""
Dual Agent System Runner - Server Mode Only

This script runs two agents in sequence via langgraph server:
1. Math Agent: Performs calculations using tools
2. Question Agent: Analyzes the first agent's result and asks follow-up questions

Usage:
    python run_dual_agents.py
    
REQUIREMENTS:
- Make sure the langgraph server is running: langgraph dev
- Server should be available at http://127.0.0.1:2024
- Both math_agent and question_agent graphs must be deployed
"""

import asyncio
import sys
from dual_agent_system import run_dual_agents


async def interactive_mode():
    """Run the dual agent system in interactive mode"""
    print("🤖 Dual Agent System - Server Mode")
    print("==================================")
    print("This system uses two agents via langgraph server:")
    print("1. Math Agent: Performs calculations using tools")
    print("2. Question Agent: Analyzes results and asks follow-up questions")
    print("\n⚠️  REQUIREMENTS:")
    print("- langgraph server must be running: langgraph dev")
    print("- Server at http://127.0.0.1:2024")
    print("- Both math_agent and question_agent graphs deployed")
    print("\nType 'quit' or 'exit' to stop")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            query = input("\n💬 Enter your math question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not query:
                print("❌ Please enter a valid question.")
                continue
            
            print(f"\n🔄 Processing: '{query}'")
            print("-" * 30)
            
            # Run the dual agent system via server
            result = await run_dual_agents(query)
            
            # Check for errors
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            # Display results
            print("\n📊 FIRST AGENT (Math Calculation) RESULT:")
            print("-" * 40)
            print(result['first_agent_result'])
            
            print("\n🤔 SECOND AGENT (Question & Analysis) RESULT:")
            print("-" * 40)
            print(result['question_agent_analysis'])
            
            print("\n" + "="*50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
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
        
        print(f"\n📊 First Agent (Math) Result:")
        print("-" * 30)
        print(result['first_agent_result'])
        
        print(f"\n🤔 Second Agent (Question & Analysis) Result:")
        print("-" * 30)
        print(result['question_agent_analysis'])
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please make sure:")
        print("1. langgraph server is running: langgraph dev")
        print("2. Server is available at http://127.0.0.1:2024")
        print("3. Both math_agent and question_agent graphs are deployed")


def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        asyncio.run(single_query_mode(query))
    else:
        # Interactive mode
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
