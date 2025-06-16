import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.train_model import AnomalyModelTrainer
from loguru import logger

def main():
    """Main function to train and register the model."""
    try:
        logger.info("Starting model training...")
        trainer = AnomalyModelTrainer()
        trainer.train_and_evaluate()
        logger.info("Model training completed successfully!")
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 