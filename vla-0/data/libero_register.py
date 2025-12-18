# libero_dataset.py
from email.mime import image
import random
from typing import Any, Dict, Optional

from swift.llm import DatasetMeta, ResponsePreprocessor, register_dataset, load_dataset

def random_masking(response: str, masking_ratio: float) -> str:
    pass

class VLA0LiberoPreprocessor(ResponsePreprocessor):

    def __init__(self, masking_ratio: float = 0.25, is_training: bool = True):
        """
        Args:
            masking_ratio: 动作文本的掩码比例。
            is_training: 只有在训练时才应用掩码。
        """
        super().__init__()
        self.masking_ratio = masking_ratio
        self.is_training = is_training

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        system = row.get('system')
        query = row.get('query')
        response = row.get('response')
        images = row.get('images', [])

        if not system or not query or not response or not images:
            raise ValueError(f"Missing required fields in the dataset row. {row}")
            
        return super().preprocess({
            'system': system,
            'query': query,
            'response': response,
            'images': images,
        })
    
    
    def prepare_dataset(self, dataset):
        self.prefix_path = '/home/yuquan002/ssd/xyq_ws/libero_vl_dataset/'
        return super().prepare_dataset(dataset)

register_dataset(
    DatasetMeta(
        dataset_name='libero-vla0',
        ms_dataset_id='libero_vla0',
        hf_dataset_id=None,
        dataset_path='/home/yuquan002/ssd/xyq_ws/libero_vl_dataset/train.jsonl',
        preprocess_func=VLA0LiberoPreprocessor(masking_ratio=0.25, is_training=True),
    ))

register_dataset(
    DatasetMeta(
        dataset_name='libero-spatial-vla0',
        ms_dataset_id='libero_spatial_vla0',
        hf_dataset_id=None,
        dataset_path='/home/yuquan002/ssd/xyq_ws/libero_vl_dataset/libero_spatial_vla/train_chunked.jsonl',
        preprocess_func=VLA0LiberoPreprocessor(masking_ratio=0.25, is_training=False),
    ))

register_dataset(
    DatasetMeta(
        dataset_name='libero-spatial-chunked-vla0',
        ms_dataset_id='libero_spatial_chunked_vla0',
        hf_dataset_id=None,
        dataset_path='/home/yuquan002/ssd/xyq_ws/libero_vl_dataset/libero_spatial_vla/train_chunk.jsonl',
        preprocess_func=VLA0LiberoPreprocessor(masking_ratio=0.25, is_training=False),
    ))

register_dataset(
    DatasetMeta(
        dataset_name='vla0-debug',
        dataset_path='/home/yuquan002/ssd/xyq_ws/libero_vl_dataset/libero_spatial_vla/train_1k.jsonl',
        preprocess_func=VLA0LiberoPreprocessor(masking_ratio=0.25, is_training=False),
    ))

register_dataset(
    DatasetMeta(
        dataset_name='libero-spatial-vla',
        dataset_path='/home/yuquan002/ssd/xyq_ws/libero_vl_dataset/libero_spatial_vla/libero_raw.jsonl',
        preprocess_func=VLA0LiberoPreprocessor(masking_ratio=0.25, is_training=False),
    ))


if __name__ == '__main__':
    dataset = load_dataset(['vla0-debug'])[0]
    print(f'dataset: {dataset}')
    print(f'dataset[0]: {dataset[0]}')