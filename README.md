# TANGO

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/mabonmn/TANGO.git
   cd TANGO
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Code

### Jupyter Notebooks

1. Launch Jupyter Notebook:
   ```sh
   jupyter notebook
   ```

2. Open the desired notebook (e.g., `Dev_Basic.ipynb`, `scraper.ipynb`) and run the cells.

### Python Scripts

1. Run the Python script directly:
   ```sh
   python scraper.py
   ```

## Main Scripts and Functionalities

### `Dev_Basic.ipynb`

- **Purpose**: This notebook demonstrates basic development and testing of the core functionalities.
- **Sections**:
  - Data Loading
  - Data Preprocessing
  - Model Training
  - Evaluation

### `scraper.ipynb`

- **Purpose**: This notebook contains code to scrape and preprocess data from various sources.
- **Sections**:
  - Data Scraping
  - Data Cleaning
  - Data Export

### `scraper.py`

- **Purpose**: This script is used to scrape data from the Wikipedia English corpus and save it as a CSV file.
- **Usage**:
  ```sh
  python scraper.py
  ```

## Dataset Files

### `dataset_Small/dataset_aug_train_all_new_clean.csv`

- **Purpose**: This file contains augmented training data for the model.
- **Usage**: Load this CSV file into your data processing pipeline to train the model with augmented data.

### `dataset_Small/dataset_aug_train_all_new.csv`

- **Purpose**: This file contains the original augmented training data.
- **Usage**: Similar to the clean version, but may contain raw and unprocessed entries.

### `dataset_Small/dataset_train.csv`

- **Purpose**: This file contains the original training data.
- **Usage**: Use this file for initial training and testing of the model.
