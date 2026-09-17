"""Compatibility entry point; use main.py for new runs."""
from src.data_preparation import CleanFeatures, load_data, make_pipeline
from main import main

if __name__ == "__main__":
    main()
