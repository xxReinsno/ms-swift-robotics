# vla0_plugin.py
import torch
from swift.plugin import LossScaleRegistry, LossRegistry
from swift.utils import get_logger

logger = get_logger()

# 逻辑 1: 针对 '?' 进行 Loss Masking
@LossScaleRegistry.register('vla0_mask')
def vla0_loss_scale(labels, tokenizer, **kwargs):
    # 获取 '?' 的 ID。注意：在某些 tokenizer 中，'?' 可能有前导空格 ID
    # 这里我们获取包含 '?' 字符的所有可能 token ID
    question_mark_id = tokenizer.convert_tokens_to_ids('?')
    
    weights = torch.ones_like(labels, dtype=torch.float32)
    weights[labels == question_mark_id] = 0.0
    weights[labels == -100] = 0.0
    return weights

# 逻辑 2: 输出强制约束为 0~9, 空格, EOS
@LossRegistry.register('vla0_constrained_loss')
class VLA0Loss:
    def __call__(self, model_output, labels, **kwargs):
        logits = model_output.logits
        tokenizer = kwargs.get('tokenizer')
        
        # 1. 找到所有合法的 token ID
        # vla0 限制输出为：数字 0-9, 空格 ' ', 和 终止符 EOS
        allowed_chars = [str(i) for i in range(10)] + [' ']
        allowed_ids = set()
        for char in allowed_chars:
            # 加入字符本身的 ID 和可能带前导空格的 ID
            allowed_ids.add(tokenizer.convert_tokens_to_ids(char))
        allowed_ids.add(tokenizer.eos_token_id)
        if tokenizer.pad_token_id is not None:
            allowed_ids.add(tokenizer.pad_token_id)
        
        allowed_ids = list(allowed_ids)
        
        # 2. 构造 Mask (除了允许的 ID，其余全部设为负无穷)
        mask = torch.full((logits.size(-1),), float('-inf'), device=logits.device)
        mask[allowed_ids] = 0
        
        # 3. 应用约束并计算交叉熵
        # 注意：这里我们手动应用约束，防止模型在训练时学习到非法 token
        constrained_logits = logits + mask
        
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = constrained_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))