from pymilvus import connections, utility 
from pymilvus import MilvusClient, Collection, FieldSchema, CollectionSchema
from pymilvus import DataType

# Milvus 연결 설정
connections.connect("default", host="ii578293.iptime.org", port="19530")  # Milvus 서버 설정 (기본값은 localhost:19530)

# Milvus Collection 스키마 정의
fields = [
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),  # 예: OpenAI 임베딩 차원
    FieldSchema(name="message", dtype=DataType.STRING, is_primary=True),
    FieldSchema(name="timestamp", dtype=DataType.INT64)
]

schema = CollectionSchema(fields, description="Chat history collection")
collection_name = "chat_history"
if collection_name not in connections.get_connection("default").list_collections():
    collection = Collection(name=collection_name, schema=schema)
else:
    collection = Collection(collection_name)

def build_FieldSchema(cfg: dict):
    """
    """
    field_name = {
            "file_id" : {
            "dtype" : DataType.INT64, 
            "is_primary" : True, 
            "auto_id" : True
        },
        "filename" : {
            "dtype" : DataType.VARCHAR, 
            "max_length" : 20
        },
        "date" : {
            "dtype" : DataType.VARCHAR,
            "max_length" : 10
        },
        "Title" : {
            "dtype" : DataType.VARCHAR, 
            "max_length" : 1000
        },
        "Reviews" : {
            "dtype" : DataType.VARCHAR, 
            "max_length" : 1000
        },
        "Embeddings" : {
            "dtype" : DataType.FLOAT_VECTOR, 
            "dim" : 768
        },
        cfg["EMBEDDING_FIELD_NAME"] : {
            "dtype" : DataType.FLOAT_VECTOR,
            "dim" : cfg["DIMENSION"]
        }
    }
    
    if utility.has_collection(cfg["COLLECTION_NAME"]):
        utility.drop_collection(cfg["COLLECTION_NAME"])
    
    fields = [
        FieldSchema(name = n, **args) for n, args in field_name.items()
    ]
    schema = CollectionSchema(fields=fields)
    collection = Collection(
        name=cfg["COLLECTION_NAME"], 
        schema=schema
    )
    # 인덱스 부분
    index_params = {
        'metric_type':'L2',
        'index_type': cfg["INDEX_TYPE"],
        'params':{"nlist":128}
    }

    # colletion에서 색인 부분을 생성하는 부분을 명시하고 부르겠다는 뜻
    collection.create_index(
        field_name=cfg["EMBEDDING_FIELD_NAME"], 
        index_params=index_params
    )
    collection.load()

class db_connection:
    def __init__(self, uri, token, database):
        self.client = MilvusClient(uri=uri, db_name=database, token=token)

    def get_client(self):
        return self.client

    def create_schema(self, auto_id=False, enable_dynamic_field=True):
        schema = self.client.create_schema(
            auto_id=auto_id,
            enable_dynamic_field=enable_dynamic_field,
        )

        return schema

    def insert_data(self, collection_name, data):
        res = self.client.insert(
            collection_name=collection_name,
            data=data
        )
        
        return res