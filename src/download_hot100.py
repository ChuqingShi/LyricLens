from pathlib import Path
import requests

HOT100_URL = "https://raw.githubusercontent.com/utdata/rwd-billboard-data/refs/heads/main/data-out/hot-100-current.csv"
HOT100_OUTPUT = Path("data/raw/hot-100-current.csv")


def download_hot100() -> None:
    # download the Billboard Hot 100 dataset and put it in data/raw/.
    url = HOT100_URL
    output = HOT100_OUTPUT

    output.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    output.write_bytes(response.content)

    print(f"Downloaded hot-100 data to {output}.")


def main():
    download_hot100()


if __name__ == "__main__":
    main()
