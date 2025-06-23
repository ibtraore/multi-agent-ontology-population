#!/bin/bash

# Install dependencies if not already installed
if [ ! -d ".venv" ]; then
  echo "📦 Creating virtual environment..."
  crewai install
fi

# Activate the virtual environment
source .venv/bin/activate

# Launch your app
echo "🚀 Launching the CrewAI application..."
python src/ontology_population_project/gradio_interface.py
