from pymilvus import connections, db , Role
from data_processing import db_connection, get_sentence_embedding, change_float_vector

_URI = "http://localhost:19530"
_TOKEN = "root:Milvus"

conn = connections.connect(uri=_URI,
        token=_TOKEN)

database = db.create_database("project4")

connect = db_connection(_URI, _TOKEN, 'project4')

schema = connect.create_schema()

# 3.2. Add fields to schema
schema.add_field(field_name="review_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="review_vector", datatype=DataType.FLOAT_VECTOR, dim=384)
schema.add_field(field_name="review_varchar", datatype=DataType.VARCHAR, max_length=8192)
schema.add_field(field_name="review_product", datatype=DataType.VARCHAR, max_length=50)

# 3.3. Prepare index parameters
index_params = client.prepare_index_params()

# 3.4. Add indexes
index_params.add_index(
    field_name="review_id",
    index_type="STL_SORT"
)

index_params.add_index(
    field_name="review_vector", 
    index_type="AUTOINDEX",
    metric_type="COSINE"
)

# 3.5. Create a collection with the index loaded simultaneously
client.create_collection(
    collection_name="review_data",
    schema=schema,
    index_params=index_params
)

res = client.get_load_state(
    collection_name="review_data"
)

print(res)

import pandas as pd

milk_df = pd.read_csv('./data/milk.csv', delimiter=",")

insert_data = []

for index , (star, review) in tqdm(milk_df.iterrows()):
    insert_data.append({'review_id' : index, 
                        'review_vector' : change_float_vector(get_sentence_embedding(review)),
                        'review_varchar' : review,
                        'review_product' : '서울우유'})


re_data2 = []

for _data in insert_data:
    _data123 = {}
    _data123['review_vector'] = _data['review_vector'][0]
    _data123['review_varchar'] = _data['review_varchar']
    _data123['review_id'] = _data['review_id']
    _data123['review_product'] = _data['review_product']

    re_data2.append(_data123)

res = connect.get_client().insert( collection_name="review_data",data=re_data2)