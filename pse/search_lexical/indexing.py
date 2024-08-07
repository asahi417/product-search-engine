import os

from tqdm import tqdm
from typing import Optional, List, Dict, Any
from opensearchpy import OpenSearch
from datasets import load_dataset

from ..util import get_logger

logger = get_logger(__name__)


def indexing_hf_datasets(
        dataset_path: str,
        dataset_name: Optional[str] = None,
        dataset_split: Optional[str] = None,
        dataset_column_names: Optional[List[str]] = None,
        host: str = "localhost",
        port: int = 9200,
        settings: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None):

    logger.info("setup config")
    dataset = load_dataset(dataset_path, dataset_name, split=dataset_split)
    dataset_column_names = dataset.column_names if dataset_column_names is None else dataset_column_names
    if any(c not in dataset.column_names for c in dataset_column_names):
        raise ValueError(f"{dataset_column_names} contains invalid columns: {dataset.column_names}")
    if index_name is None:
        index_name = f"dataset_path={os.path.basename(dataset_path)}.dataset_name={dataset_name}.dataset_split={dataset_split}"
    if settings is None:
        properties = {c: {"type": "text" if "id" in c else "keyword"} for c in dataset_column_names}
        settings = {"mappings": {"properties": properties}}
    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=("admin", "admin"),
        use_ssl=True,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=settings)

    logger.info("start ingestion")
    for n, row in tqdm(enumerate(dataset)):
        client.index(
            index=index_name,
            body=row,
            id=n + 1
        )
