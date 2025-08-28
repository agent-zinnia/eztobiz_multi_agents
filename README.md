# EztoBiz Multi-Agent System

A sophisticated dual-agent system that combines mathematical computation with intelligent question generation using LangGraph Platform and OpenAI GPT models.

## Overview

This project implements a multi-turn conversation system featuring two specialized agents:

1. **Math Agent**: Performs mathematical calculations and problem-solving via LangGraph Platform API
2. **Question Agent**: Generates intelligent follow-up questions based on mathematical results

The system supports configurable question rounds, allowing for extended mathematical conversations and explorations.

## Architecture

```
User Query → Math Agent → Question Agent → Math Agent → ... (configurable rounds)
```

The system maintains conversation context across multiple rounds, enabling deep mathematical exploration.

## Installation

### Prerequisites

- Python 3.11 or higher
- OpenAI API key
- LangGraph API key

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd eztobiz_multi_agents
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Configure environment**:
   ```bash
   cp env.example .env
   ```

4. **Edit `.env` file** with your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   LANGGRAPH_API_KEY=your_langgraph_api_key_here
   ```

## Usage

### Interactive Mode

Run the system in interactive mode for a user-friendly experience:

```bash
python src/agent/run_dual_agents.py
```

### Command Line Mode

Execute direct queries with customizable question rounds:

```bash
# Single round (default)
python src/agent/run_dual_agents.py "What is 15 + 27?"

# Multiple rounds
python src/agent/run_dual_agents.py "Calculate the area of a circle with radius 5" --rounds 3

# Using short flag
python src/agent/run_dual_agents.py "What is 2^10?" -r 2
```


## Examples

### Basic Mathematical Query

```bash
python src/agent/run_dual_agents.py "What is 15 × 27 + 42?"
```

**Output**:
```
🔄 Running dual agent system for: 'What is 15 × 27 + 42?'

📊 Math Agent Result:
447

🤔 Generated Question:
What would be the result if we divided this answer by 3?

🔢 Answer:
149
```

### Multi-Round Exploration

```bash
python src/agent/run_dual_agents.py "Calculate the volume of a sphere with radius 3" -r 2
```

This generates progressively deeper questions about the mathematical result.

## Configuration

The system uses a centralized configuration in `src/agent/config.py`:

- **Models**: GPT-4o-mini for both agents
- **Platform**: LangGraph Platform integration
- **Timeouts**: 30-second maximum wait time
- **Streaming**: 1024-byte chunk processing

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API authentication | Yes |
| `LANGGRAPH_API_KEY` | LangGraph Platform API key | Yes |
| `LANGGRAPH_LOCAL_SERVER_URL` | Local server URL (optional) | No |

## Project Structure

```
eztobiz_multi_agents/
├── src/agent/
│   ├── config.py              # Configuration management
│   ├── dual_agent_system.py   # Core system orchestration
│   ├── math_agent.py          # Mathematical computation agent
│   ├── question_agent.py      # Question generation agent
│   └── run_dual_agents.py     # CLI interface
├── tests/
│   ├── unit_tests/            # Unit test suite
│   └── integration_tests/     # Integration test suite
├── pyproject.toml             # Project dependencies
├── env.example                # Environment template
└── README.md                  # This file
```

## Logging

The system generates comprehensive logs in:
- `dual_agent_system.log` - Core system operations
- `dual_agent_runner.log` - CLI interface operations

## Author

**zinnia-agsn** - zinnia@agsn.ai