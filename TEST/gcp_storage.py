import os
from pprint import pprint
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "aiffel-gn-3-c8c200820331.json"

storage_client = storage.Client()

buckets = list(storage_client.list_buckets())
print(buckets)


my_bucket = storage_client.get_bucket("images-original")

print(my_bucket)
