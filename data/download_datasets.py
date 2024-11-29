import os

def download_dataset(dataset_name, target_dir="raw"):
    try:
        os.makedirs(target_dir, exist_ok=True)

        # IMPORTANT: You need to set your credentials as environment variables before using Kaggle API
        # os.environ["KAGGLE_USERNAME"] = "your_kaggle_username"
        # os.environ["KAGGLE_KEY"] = "your_kaggle_api_key"

        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        print("Authenticating...")
        api.authenticate()

        print(f"Downloading {dataset_name}...")
        api.dataset_download_files(
            dataset_name,
            path=target_dir,
            unzip=True
        )
        print(f"Dataset downloaded successfully to {target_dir}")

    except Exception as e:
        print(f"Error downloading dataset: {e}")


if __name__ == "__main__":
    datasets = [
        "imranbukhari/comprehensive-btcusd-1m-data",
        "imranbukhari/comprehensive-ethusd-1m-data",
    ]

    for dataset in datasets:
        target_dir = os.path.join("raw", dataset.split("/")[1])
        download_dataset(dataset, target_dir)
