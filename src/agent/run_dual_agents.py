#!/usr/bin/env python3
"""
Triple Agent System Runner - Server Mode Only

This script runs three agents in sequence via langgraph server:
1. Math Agent: Performs calculations using tools
2. Question Agent: Generates follow-up questions based on the result
3. Math Agent: Answers the generated question

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
    """Run the triple agent system in interactive mode"""
    print("🤖 Triple Agent System - Server Mode")
    print("===================================")
    print("This system uses three agents via langgraph server:")
    print("1. Math Agent: Performs calculations using tools")
    print("2. Question Agent: Generates follow-up questions")
    print("3. Math Agent: Answers the generated question")
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
            print("\n📊 STEP 1 - MATH AGENT RESULT:")
            print("-" * 40)
            print(result['step1_math_result'])
            
            print("\n🤔 STEP 2 - GENERATED QUESTION:")
            print("-" * 40)
            print(result['step2_generated_question'])
            
            print("\n🔢 STEP 3 - ANSWER TO QUESTION:")
            print("-" * 40)
            print(result['step3_answer_to_question'])
            
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
    """Run a single query through the triple agent system"""
    print(f"🔄 Running triple agent system for: '{query}'")
    
    try:
        result = await run_dual_agents(query)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print("\n" + "="*60)
        print("TRIPLE AGENT SYSTEM RESULTS")
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
