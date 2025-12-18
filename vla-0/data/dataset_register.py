import random
from typing import Any, Dict, Optional
from swift.llm import DatasetMeta, ResponsePreprocessor, register_dataset, load_dataset
from swift.utils import get_logger

logger = get_logger()

def random_masking(response: str, masking_ratio: float) -> str:
    """
    vla0 核心增强逻辑：
    对字符串中的每个非空格字符，以 masking_ratio 的概率替换为 '?'。
    例如: "123 456" -> "1?3 ?56"
    """
    if not response:
        return response
    
    masked_chars = []
    for char in response:
        # 只有非空格字符才参与掩码
        if char != ' ' and random.random() < masking_ratio:
            masked_chars.append('?')
        else:
            masked_chars.append(char)
            
    return "".join(masked_chars)

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

        if not response:
             raise ValueError(f"Missing 'response' field in the dataset row. {row}")

        # 核心改动：在训练阶段对 response 进行动态掩码
        if self.is_training and self.masking_ratio > 0:
            response = random_masking(response, self.masking_ratio)

        # 调用父类的标准处理流程
        return super().preprocess({
            'system': system,
            'query': query,
            'response': response,
            'images': images,
        })

    def prepare_dataset(self, dataset):
        # 如果你的图片路径是相对路径，可以在这里统一处理 prefix
        # 注意：ms-swift 内部会自动处理图片加载，确保 dataset 里的 images 路径正确
        return super().prepare_dataset(dataset)

# --- 数据集注册部分 ---

# 训练集：启用动态掩码 (masking_ratio=0.25)
register_dataset(
    DatasetMeta(
        dataset_name='libero-vla0',
        ms_dataset_id='libero_vla0',
        hf_dataset_id=None,
        dataset_path='/home/yuquan002/ssd/xyq_ws/libero_vl_dataset/libero_omni_swift/train.jsonl',
        preprocess_func=VLA0LiberoPreprocessor(masking_ratio=0.4, is_training=True),
    ))
