#!/usr/bin/env python3
"""
Dual Agent System Runner - Configurable Question Rounds

This script runs two agents in sequence:
1. Math Operations: Performs calculations via LangGraph Platform API
2. Question Agent: Generates follow-up questions based on the result

Usage:
    python run_dual_agents.py                                    # Interactive input mode
    python run_dual_agents.py "your math question"               # 1 question round (default)
    python run_dual_agents.py "your math question" -r 3          # 3 question rounds
    python run_dual_agents.py --rounds 5 "Calculate 10 * 20"     # 5 question rounds
    
REQUIREMENTS:
- Make sure the langgraph server is running: langgraph dev  
- Server should be available at configured URL (check env.example)
- question_agent graph must be deployed locally
"""

import asyncio
import sys
import argparse
import os

# Add the src directory to Python path for proper imports
src_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, src_dir)

from agent.dual_agent_system import run_dual_agents
from agent.config import config


def _print_server_requirements():
    """Print server requirements message"""
    print("Please make sure:")
    print("1. langgraph server is running: langgraph dev")
    print(f"2. Server is available at {config.LANGGRAPH_LOCAL_SERVER_URL}")
    print("3. question_agent graph is deployed locally")





async def single_query_mode(query: str, question_rounds: int = 1):
    """Run a single query through the dual agent system"""
    print(f"🔄 Running dual agent system for: '{query}'")
    
    try:
        result = await run_dual_agents(query, question_rounds=question_rounds)
        
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
        
        # Display results for each question round
        for round_data in result.get('question_rounds', []):
            round_num = round_data['round']
            print(f"\n🤔 Round {round_num} - Generated Question:")
            print("-" * 30)
            if round_data['generated_question']:
                print(round_data['generated_question'])
            else:
                print(f"❌ Error: {round_data.get('error', 'Unknown error')}")
            
            print(f"\n🔢 Round {round_num} - Answer:")
            print("-" * 30)
            if round_data['answer']:
                print(round_data['answer'])
            else:
                print(f"❌ Error: {round_data.get('error', 'Unknown error')}")
        
        # Display summary
        total_rounds = len(result.get('question_rounds', []))
        print(f"\n📊 Summary: Completed {total_rounds}/{question_rounds} question rounds")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        _print_server_requirements()


def main():
    """Main function to handle command line arguments or user input"""
    # Validate configuration before proceeding
    if not config.validate_config():
        print("\n❌ Configuration validation failed!")
        print("Please check your .env file and ensure all required variables are set.")
        print("See env.example for reference.")
        return
    
    parser = argparse.ArgumentParser(
        description="Dual Agent System - Run math and question agents sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_dual_agents.py                                    # Interactive mode
  python run_dual_agents.py "What is 15 + 27?"               # Direct query (1 round)
  python run_dual_agents.py "What is 15 + 27?" -r 3          # Direct query (3 rounds)
  python run_dual_agents.py --rounds 5 "Calculate 10 * 20"   # Direct query (5 rounds)

REQUIREMENTS:
- langgraph server must be running: langgraph dev
- Server at configured URL (check env.example)
- question_agent graph deployed locally
        """
    )
    
    parser.add_argument(
        'query', 
        nargs='?', 
        help='Math question to process (if not provided, interactive mode will be used)'
    )
    parser.add_argument(
        '-r', '--rounds', 
        type=int, 
        default=1, 
        help='Number of question rounds to run (default: 1)'
    )
    
    args = parser.parse_args()
    
    if args.query:
        # Single query mode with command line argument
        asyncio.run(single_query_mode(args.query, args.rounds))
    else:
        # Single query mode with user input
        print("🤖 Dual Agent System - Single Query Mode")
        print("=" * 45)
        print("This system uses two agents via langgraph server:")
        print("1. Math Operations: Via LangGraph Platform API")
        print("2. Question Agent: Generates follow-up questions")
        print("\n⚠️  REQUIREMENTS:")
        print("- langgraph server must be running: langgraph dev")
        print(f"- Server at {config.LANGGRAPH_LOCAL_SERVER_URL}")
        print("- question_agent graph deployed locally")
        print("-" * 45)
        
        try:
            query = input("\n💬 Enter your math question: ").strip()
            
            if not query:
                print("❌ Please enter a valid question.")
                return
            
            # Ask for number of rounds in interactive mode
            while True:
                try:
                    rounds_input = input(f"🔄 How many question rounds? (default: 1): ").strip()
                    if not rounds_input:
                        question_rounds = 1
                        break
                    question_rounds = int(rounds_input)
                    if question_rounds < 1:
                        print("❌ Number of rounds must be at least 1.")
                        continue
                    break
                except ValueError:
                    print("❌ Please enter a valid number.")
                    continue
            
            asyncio.run(single_query_mode(query, question_rounds))
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            _print_server_requirements()


if __name__ == "__main__":
    main()
