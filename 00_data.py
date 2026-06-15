# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import os
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
os.environ["KAGGLEHUB_CACHE"] = "./data"
file_path = "WELFake_Dataset.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "saurabhshahane/fake-news-classification",
  file_path,
  # Provide any additional arguments like
  # sql_query or pandas_kwargs. See the
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())
print(df.dtypes)
print(df.describe())