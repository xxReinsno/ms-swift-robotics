# robotics_template.py  
import random  
from swift.llm.template import Template, register_template, TemplateMeta
from swift.llm.template.template_inputs import StdTemplateInputs  
  
class QwenRoboticsTemplate(Template):  
    def __init__(self, *args, action_mask_aug_per=0.1, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.action_mask_aug_per = action_mask_aug_per  
      
    def _encode(self, inputs: StdTemplateInputs):  
        encoded = super()._encode(inputs)  
          
        # 应用随机掩码增强（保留原始逻辑）  
        if self.is_training and random.random() > 0.1:  # 10% 概率不增强  
            encoded = self._apply_action_mask_augmentation(encoded)  
          
        return encoded  
      
    def _apply_action_mask_augmentation(self, encoded):  
        """实现随机掩码增强，与原始代码逻辑一致"""  
        input_ids = encoded['input_ids']  
        labels = encoded.get('labels', input_ids.clone())  
          
        for i in range(input_ids.shape[0]):  
            # 找到assistant响应中的动作序列  
            assistant_start = self._find_assistant_start(input_ids[i])  
            if assistant_start > 0:
                action_tokens = self._find_action_tokens(input_ids[i], assistant_start)  
                if action_tokens:  
                    # 计算掩码长度（与原始逻辑一致）  
                    mask_len = int(len(action_tokens) * self.action_mask_aug_per)  
                    if mask_len > 0:  
                        mask_indices = random.sample(action_tokens, mask_len)  
                        labels[i, mask_indices] = -100  # 不计算损失  
                        input_ids[i, mask_indices] = 30  # 替换为 '?' token  
          
        encoded['input_ids'] = input_ids  
        encoded['labels'] = labels  
        return encoded  
      
    def _find_assistant_start(self, input_ids):  
        """找到assistant响应的开始位置"""  
        # 实现查找assistant token的逻辑  
        pass  
      
    def _find_action_tokens(self, input_ids, start_pos):  
        """找到动作序列的token位置"""  
        # 实现查找数字token的逻辑  
        pass  
  
# 注册模板  
register_template(  
    TemplateMeta(
        template_type='qwen_robotics',  
        template_cls=QwenRoboticsTemplate,  
        default_system='Analyze the input image and predict robot actions...'  
    )  
)