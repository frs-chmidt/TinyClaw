# Setup

You can use a local LLM - install ollama qwen3.5:4b

Or use a free hosted LLM - from https://openrouter.ai
Do this by 1st creating an account. 2nd click 'Get API Key'. 3rd set Credit limit to 0. copy API Key. create .env in same dir as main and add API Keys as displayed in .env.example

Either way update PROVIDER and MODEL in config.py (choose a currently free model from https://openrouter.ai if this option is used)


Start TinyClaw by

uv run main.py

Start with -d flag for more detailed debug output

# Improvement ideas

- Check dependency-aware execution (e.g. tools that depend on eachother)
- right now they are all executed simultaneously

# Improvements
i would like plugins which can interact with docker. to be more precise i want to be able to list all docker images, stop them and start new ones



