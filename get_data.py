import os
import pandas as pd
from dotenv import load_dotenv

def main():
    # Load the Hugging Face token from the .env file
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please set HF_TOKEN in your .env file.")

    # Set the token as an environment variable for Hugging Face authentication
    os.environ["HF_HUB_TOKEN"] = hf_token

    # Ensure the data directory exists
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    # Define the datasets and their corresponding filenames
    datasets = {
        "demand.parquet": "hf://datasets/EDS-lab/electricity-demand/data/demand.parquet",
        "metadata.parquet": "hf://datasets/EDS-lab/electricity-demand/data/metadata.parquet",
        "weather.parquet": "hf://datasets/EDS-lab/electricity-demand/data/weather.parquet",
    }

    # Download and save each dataset
    for filename, url in datasets.items():
        print(f"Downloading {filename} from {url}...")
        df = pd.read_parquet(url)
        save_path = os.path.join(data_dir, filename)
        df.to_parquet(save_path)
        print(f"Saved {filename} to {save_path}")

    print("All datasets have been downloaded and saved.")

if __name__ == "__main__":
    main()