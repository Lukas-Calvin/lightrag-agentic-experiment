import os
import asyncio
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_complete, openai_embed
from lightrag.utils import logger, set_verbose_debug
import traceback

from functools import partial
# from lightrag.rerank import jina_rerank
from lightrag.rerank import ali_rerank

WORKING_DIR = "./dickens"

# rerank_model_func = partial(
#     jina_rerank,
#     model=os.getenv("RERANK_MODEL", "jina-reranker-v3"),
#     api_key=os.getenv("RERANK_BINDING_API_KEY"),
#     base_url=os.getenv("RERANK_BINDING_HOST", "https://api.jina.ai/v1/rerank"),
# )

rerank_model_func = partial(
    ali_rerank,
    model=os.getenv("RERANK_MODEL", "gte-rerank-v2"),
    api_key=os.getenv("RERANK_BINDING_API_KEY"),
    base_url=os.getenv("RERANK_BINDING_HOST", "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"),
)

def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_demo.log"))

    print(f"\nLightRAG demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_complete,
        rerank_model_func=rerank_model_func,
    )

    await rag.initialize_storages()  # Auto-initializes pipeline_status

    return rag


async def main():
    # Check if OPENAI_API_KEY environment variable exists
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is not set. Please set this variable before running the program."
        )
        print("You can set the environment variable by running:")
        print("  export OPENAI_API_KEY='your-openai-api-key'")
        return  # Exit the async function

    # 自定义添加query_only参数，默认为false
    # 注意：query_only=true 时应该复用 working_dir 里的既有索引/缓存文件，否则会查询不到内容
    query_only = os.getenv("LIGHTRAG_QUERY_ONLY", "false").lower() == "true"

    rag = None
    try:
        if not query_only:
            # Clear old data files (only in ingest mode)
            files_to_delete = [
                "graph_chunk_entity_relation.graphml",
                "kv_store_doc_status.json",
                "kv_store_full_docs.json",
                "kv_store_text_chunks.json",
                "vdb_chunks.json",
                "vdb_entities.json",
                "vdb_relationships.json",
            ]

            for file in files_to_delete:
                file_path = os.path.join(WORKING_DIR, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleting old file:: {file_path}")

        # Initialize RAG instance
        rag = await initialize_rag()

        if not query_only:
            # Test embedding function
            test_text = ["This is a test string for embedding."]
            embedding = await rag.embedding_func(test_text)
            embedding_dim = embedding.shape[1]
            print("\n=======================")
            print("Test embedding function")
            print("========================")
            print(f"Test dict: {test_text}")
            print(f"Detected embedding dimension: {embedding_dim}\n\n")

            with open("./book.txt", "r", encoding="utf-8") as f:
                await rag.ainsert(f.read())
            # 其中的f.read()作用是一次性读取整个文件内容并返回一个字符串，再将该字符串作为参数传递给rag.ainsert()方法进行处理。
            #   这种方式适用于文件较小的情况，可以方便地将整个文本内容作为一个整体进行处理和存储，如果是大文件的话，可以使用aiofiles等库进行异步逐行读取和处理，以避免一次性加载过大的文件内容导致内存问题。
            # 另外其处理方式具体为异步处理。

        # Perform naive search
        print("\n=====================")
        print("Query mode: naive")
        print("=====================")
        query1 = "What are the top themes in this story?"
        print(f"\n Test query: '{query1}'\n")
        print(
            await rag.aquery(
                query1, 
                param=QueryParam(
                    mode="naive",
                    top_k=20,
                    chunk_top_k=10,
                    enable_rerank=True,  # Enable reranking for this query)
                ),
            )
        )

        # Perform local search
        print("\n=====================")
        print("Query mode: local")
        print("=====================")
        print(f"\n Test query: '{query1}'\n")
        print(
            await rag.aquery(
                query1,
                param=QueryParam(
                    mode="local",
                    top_k=20,
                    chunk_top_k=10,
                    enable_rerank=True,  # Enable reranking for this query)
                ),
            )
        )

        # Perform global search
        print("\n=====================")
        print("Query mode: global")
        print("=====================")
        print(
            await rag.aquery(
                query1,
                param=QueryParam(
                    mode="global",
                    top_k=20,
                    chunk_top_k=10,
                    enable_rerank=True,  # Enable reranking for this query)
                ),
            )
        )

        # Perform hybrid search
        print("\n=====================")
        print("Query mode: hybrid")
        print("=====================")
        print(
            await rag.aquery(
                query1,
                param=QueryParam(
                    mode="hybrid",
                    top_k=20,
                    chunk_top_k=10,
                    enable_rerank=True,  # Enable reranking for this query)
                    ),
            )
        )
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    except Exception:
        traceback.print_exc()   
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
