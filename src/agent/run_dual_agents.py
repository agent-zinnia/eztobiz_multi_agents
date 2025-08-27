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

import argparse
import asyncio
import logging
import os
import sys
from typing import Dict, Any, Optional

# Add the src directory to Python path for proper imports
src_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, src_dir)

from agent.config import config
from agent.dual_agent_system import run_dual_agents

# Constants
DEFAULT_QUESTION_ROUNDS = 1
MIN_QUESTION_ROUNDS = 1
SEPARATOR_LENGTH = 60
SUB_SEPARATOR_LENGTH = 30
HEADER_SEPARATOR_LENGTH = 45

# UI Messages
UI_MESSAGES = {
    'title': '🤖 Dual Agent System - Single Query Mode',
    'description': 'This system uses two agents via langgraph server:',
    'math_agent_desc': '1. Math Operations: Via LangGraph Platform API',
    'question_agent_desc': '2. Question Agent: Generates follow-up questions',
    'requirements_header': '⚠️  REQUIREMENTS:',
    'server_requirement': 'langgraph server must be running: langgraph dev',
    'running_prompt': '🔄 Running dual agent system for: \'{}\'',
    'results_header': 'DUAL AGENT SYSTEM RESULTS',
    'original_query': 'Original Query: {}',
    'math_result_header': '📊 Step 1 - Math Agent Result:',
    'question_header': '🤔 Round {} - Generated Question:',
    'answer_header': '🔢 Round {} - Answer:',
    'summary': '📊 Summary: Completed {}/{} question rounds',
    'input_prompt': '💬 Enter your math question: ',
    'rounds_prompt': '🔄 How many question rounds? (default: 1): ',
    'goodbye': '👋 Interrupted by user. Goodbye!'
}

ERROR_MESSAGES = {
    'config_failed': '❌ Configuration validation failed!',
    'check_env': 'Please check your .env file and ensure all required variables are set.',
    'see_example': 'See env.example for reference.',
    'invalid_query': '❌ Please enter a valid question.',
    'invalid_rounds': '❌ Number of rounds must be at least 1.',
    'invalid_number': '❌ Please enter a valid number.',
    'general_error': '❌ An error occurred: {}',
    'result_error': '❌ Error: {}',
    'unknown_error': 'Unknown error'
}


def _print_server_requirements() -> None:
    """Print server requirements message."""
    print("Please make sure:")
    print("1. langgraph server is running: langgraph dev")
    print(f"2. Server is available at {config.LANGGRAPH_LOCAL_SERVER_URL}")
    print("3. question_agent graph is deployed locally")





def _print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * SEPARATOR_LENGTH)
    print(title)
    print("=" * SEPARATOR_LENGTH)


def _print_subsection_header(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n{title}")
    print("-" * SUB_SEPARATOR_LENGTH)


def _display_math_result(result: Dict[str, Any]) -> None:
    """Display the math agent result."""
    _print_subsection_header(UI_MESSAGES['math_result_header'])
    print(result['step1_math_result'])


def _display_question_round(round_data: Dict[str, Any]) -> None:
    """Display a single question round result."""
    round_num = round_data['round']
    
    # Display generated question
    _print_subsection_header(UI_MESSAGES['question_header'].format(round_num))
    if round_data['generated_question']:
        print(round_data['generated_question'])
    else:
        error_msg = round_data.get('error', ERROR_MESSAGES['unknown_error'])
        print(ERROR_MESSAGES['result_error'].format(error_msg))
    
    # Display answer
    _print_subsection_header(UI_MESSAGES['answer_header'].format(round_num))
    if round_data['answer']:
        print(round_data['answer'])
    else:
        error_msg = round_data.get('error', ERROR_MESSAGES['unknown_error'])
        print(ERROR_MESSAGES['result_error'].format(error_msg))


def _display_results(result: Dict[str, Any], question_rounds: int) -> None:
    """Display the complete dual agent system results."""
    _print_section_header(UI_MESSAGES['results_header'])
    print(UI_MESSAGES['original_query'].format(result['original_query']))
    
    # Display math result
    _display_math_result(result)
    
    # Display results for each question round
    for round_data in result.get('question_rounds', []):
        _display_question_round(round_data)
    
    # Display summary
    total_rounds = len(result.get('question_rounds', []))
    print(f"\n{UI_MESSAGES['summary'].format(total_rounds, question_rounds)}")


async def single_query_mode(query: str, question_rounds: int = DEFAULT_QUESTION_ROUNDS) -> None:
    """Run a single query through the dual agent system.
    
    Args:
        query: The math question to process
        question_rounds: Number of follow-up question rounds to run
    """
    print(UI_MESSAGES['running_prompt'].format(query))
    
    try:
        result = await run_dual_agents(query, question_rounds=question_rounds)
        
        if "error" in result:
            print(ERROR_MESSAGES['result_error'].format(result['error']))
            return
        
        _display_results(result, question_rounds)
        
    except Exception as e:
        print(ERROR_MESSAGES['general_error'].format(e))
        _print_server_requirements()


def _validate_configuration() -> bool:
    """Validate configuration and print error messages if needed.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    if not config.validate_config():
        print(f"\n{ERROR_MESSAGES['config_failed']}")
        print(ERROR_MESSAGES['check_env'])
        print(ERROR_MESSAGES['see_example'])
        return False
    return True


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
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
        default=DEFAULT_QUESTION_ROUNDS, 
        help=f'Number of question rounds to run (default: {DEFAULT_QUESTION_ROUNDS})'
    )
    
    return parser


def _print_interactive_header() -> None:
    """Print the interactive mode header."""
    print(UI_MESSAGES['title'])
    print("=" * HEADER_SEPARATOR_LENGTH)
    print(UI_MESSAGES['description'])
    print(UI_MESSAGES['math_agent_desc'])
    print(UI_MESSAGES['question_agent_desc'])
    print(f"\n{UI_MESSAGES['requirements_header']}")
    print(f"- {UI_MESSAGES['server_requirement']}")
    print(f"- Server at {config.LANGGRAPH_LOCAL_SERVER_URL}")
    print("- question_agent graph deployed locally")
    print("-" * HEADER_SEPARATOR_LENGTH)


def _get_user_input() -> Optional[str]:
    """Get and validate user input for the math question.
    
    Returns:
        User's query string if valid, None if invalid
    """
    query = input(f"\n{UI_MESSAGES['input_prompt']}").strip()
    
    if not query:
        print(ERROR_MESSAGES['invalid_query'])
        return None
    
    return query


def _get_question_rounds() -> int:
    """Get and validate the number of question rounds from user.
    
    Returns:
        Number of question rounds
    """
    while True:
        try:
            rounds_input = input(UI_MESSAGES['rounds_prompt']).strip()
            if not rounds_input:
                return DEFAULT_QUESTION_ROUNDS
            
            question_rounds = int(rounds_input)
            if question_rounds < MIN_QUESTION_ROUNDS:
                print(ERROR_MESSAGES['invalid_rounds'])
                continue
            
            return question_rounds
            
        except ValueError:
            print(ERROR_MESSAGES['invalid_number'])
            continue


def _run_interactive_mode() -> None:
    """Run the interactive mode for user input."""
    _print_interactive_header()
    
    try:
        query = _get_user_input()
        if not query:
            return
        
        question_rounds = _get_question_rounds()
        asyncio.run(single_query_mode(query, question_rounds))
        
    except KeyboardInterrupt:
        print(f"\n\n{UI_MESSAGES['goodbye']}")
    except Exception as e:
        print(ERROR_MESSAGES['general_error'].format(e))
        _print_server_requirements()


def main() -> None:
    """Main function to handle command line arguments or user input."""
    # Configure logging - file only, no console output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dual_agent_runner.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Dual Agent System Runner")
    
    # Validate configuration before proceeding
    if not _validate_configuration():
        logger.error("Configuration validation failed")
        return
    
    parser = _create_argument_parser()
    args = parser.parse_args()
    
    try:
        if args.query:
            # Single query mode with command line argument
            logger.info(f"Running single query mode: {args.query} with {args.rounds} rounds")
            asyncio.run(single_query_mode(args.query, args.rounds))
        else:
            # Interactive mode
            logger.info("Running interactive mode")
            _run_interactive_mode()
    except Exception as e:
        logger.error(f"Application failed: {str(e)}", exc_info=True)
        print(f"Application failed: {str(e)}")


if __name__ == "__main__":
    main()
