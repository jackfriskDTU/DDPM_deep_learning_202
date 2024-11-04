from pathlib import Path

# Define the project root directory
PROJECT_ROOT = Path(__file__).resolve()

def set_project_root():
    global PROJECT_ROOT
    PROJECT_ROOT = PROJECT_ROOT

    while PROJECT_ROOT.name != 'DDPM_deep_learning_202':
        PROJECT_ROOT = PROJECT_ROOT.parent
