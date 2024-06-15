# import asyncio

import os, sys

os.environ['HF_ENDPOINT']='https://hf-mirror.com'

from dotenv import dotenv_values
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.llms import OpenAILike as OpenAI
from qdrant_client import models
from tqdm.asyncio import tqdm

from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.qa import read_jsonl, save_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval



config = dotenv_values(".env")

# 初始化 LLM 嵌入模型 和 Reranker
llm = OpenAI(
    api_key=config["GLM_KEY"],
    model="glm-4",
    api_base="https://open.bigmodel.cn/api/paas/v4/",
    is_chat_model=True,
)
embeding = HuggingFaceEmbedding(
    # model_name="BAAI/bge-big-zh-v1.5",
    model_name="BAAI/bge-base-zh-v1.5",
    cache_folder="./",
    embed_batch_size=128,
)
Settings.embed_model = embeding

# 初始化 数据ingestion pipeline 和 vector store
client, vector_store = build_vector_store(config, reindex=False)

collection_info = client.get_collection(
    config["COLLECTION_NAME"] or "aiops24"
)

print(collection_info.points_count)

