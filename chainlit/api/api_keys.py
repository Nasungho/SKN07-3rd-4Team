import os
from dotenv import load_dotenv
from pathlib import Path

# print(f'[info] .env path: {Path(__file__).parent}')
# load_dotenv(Path(__file__).parent / '.env')
load_dotenv('./api/.env')

# print(f'[info] MILVUS_HOST: {os.getenv("MILVUS_HOST")}, MILVUS_PORT: {os.getenv("MILVUS_PORT")}')

if __name__ == "__main__":
    ...