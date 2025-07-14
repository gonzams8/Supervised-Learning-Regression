from pathlib import Path
import pandas as pd

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "gonzams8/Data/main/Supervised-Learning-Regression/"
)

def load_csv(filename: str) -> pd.DataFrame:
    """
    Busca <filename> en datasets/.
    Si no existe, lo descarga desde GitHub y lo guarda allí.
    Retorna un DataFrame.
    """
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    local_file = datasets_dir / filename
    remote_url = BASE_URL + filename

    if local_file.is_file():
        print(f"✅ Loaded local file: {local_file}")
        return pd.read_csv(local_file)

    print(f"🌐 Local file not found. Downloading from GitHub...")
    df = pd.read_csv(remote_url)
    df.to_csv(local_file, index=False)
    print(f"💾 Saved locally at: {local_file}")
    return df
