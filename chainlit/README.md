# How to run

## Prerequiste

- run `pip install -r requirements.txt`

- add openapi key unber `./api/.env`
    
    ```
    OPENAI_API_KEY=...
    ```

## Getting started

- Chainlit
    ```
    # run chainlit
    chainlit run app.py -w --host [ip_addr] --port 8501
    ``` 

- Milvus (vector DB)
    ```
    # Download the configuration file
    $ wget https://github.com/milvus-io/milvus/releases/download/v2.5.4/milvus-standalone-docker-compose.yml -O docker-compose.yml

    # Start Milvus
    $ sudo docker compose up -d

    # Stop Milvus
    $ sudo docker compose down

    # Delete service data
    $ sudo rm -rf volumes
    ```

    설치 후, Milvus WebUI(`http://localhost:9091/webui/`)에 액세스 가능하다.