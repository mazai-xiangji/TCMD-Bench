## Setup

1.  **Clone the repository:**
    ```bash
    git clone 
    cd TCMD-Bench
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Defaults (Optional but Recommended):**
    * Open the `config.py` file.
    * Locate the `Default Configuration Values` section near the top.
    * Modify the default values for API keys (`DEFAULT_SIM_API_KEY`, `DEFAULT_EXPERT_API_KEY`), base URLs, model names, and file paths as needed.
    

## Usage

Run the main script from the project root directory:

```bash
python main.py

Example Overrides:

Specify a different test model and output file:


python main.py --test_model_name MyFineTunedModel --output_json_path ./results/MyFineTunedModel_eval.json
```