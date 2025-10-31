import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd
from typing import List, Dict, Union
import re
from tqdm import tqdm
import time


class DeepSeekBatchTextClassifier:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        """初始化DeepSeek文本分类器"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")

        # 加载模型和分词器
        print("正在加载DeepSeek-R1模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("模型加载完成！")

    def classify_batch(self, texts: List[str], batch_size: int = 8,
                       system_prompt: str = "", labels_list: List[str] = None) -> List[Dict[str, Union[str, float]]]:
        """批量分类文本"""
        results = []

        # 分批处理
        for i in tqdm(range(0, len(texts), batch_size), desc="批量分类进度"):
            batch_texts = texts[i:i + batch_size]
            batch_results = self._process_batch(batch_texts, system_prompt, labels_list)
            results.extend(batch_results)

            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def _process_batch(self, texts: List[str], system_prompt: str, labels_list: List[str]) -> List[Dict]:
        """处理一个批次的文本"""
        # 构建批量消息
        all_messages = []
        for text in texts:
            user_prompt = f"Spanish text：{text}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            all_messages.append(messages)

        # 批量应用chat模板
        batch_texts = []
        for messages in all_messages:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_texts.append(text)

        # 批量tokenize
        model_inputs = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)

        # 批量生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=64,  # 减少生成长度提高速度
                do_sample=False,  # 关闭采样提高一致性
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=1,  # 使用贪婪搜索
                early_stopping=True
            )

        # 解码结果
        batch_results = []
        for i, (input_ids, output_ids) in enumerate(zip(model_inputs.input_ids, generated_ids)):
            # 提取新生成的部分
            new_tokens = output_ids[len(input_ids):]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # 提取分类标签
            label = self._extract_deepseek_classification(response, labels_list)

            batch_results.append({
                "text": texts[i],
                "raw_output": response,
                "label": label
            })

        return batch_results

    def classify_csv(self, csv_path: str, text_column: str = "text", id_column: str = "id",
                     output_path: str = None, batch_size: int = 8, system_prompt: str = "",
                     labels_list: List[str] = None, save_interval: int = 100) -> pd.DataFrame:
        """直接处理CSV文件，输出只包含id和label的结果"""
        print(f"正在读取CSV文件: {csv_path}")
        df = pd.read_csv(csv_path)

        # 检查必要列是否存在
        if text_column not in df.columns:
            raise ValueError(f"列 '{text_column}' 不存在于CSV文件中")
        if id_column not in df.columns:
            raise ValueError(f"列 '{id_column}' 不存在于CSV文件中")

        texts = df[text_column].astype(str).tolist()
        ids = df[id_column].tolist()
        print(f"共需要分类 {len(texts)} 条文本")

        # 创建结果DataFrame，只包含id和label
        result_df = pd.DataFrame({
            'id': ids,
            'label': [''] * len(ids)
        })

        # 分批处理并定期保存
        processed_count = 0

        for i in tqdm(range(0, len(texts), batch_size), desc="处理CSV"):
            batch_texts = texts[i:i + batch_size]
            batch_results = self._process_batch(batch_texts, system_prompt, labels_list)

            # 更新结果DataFrame
            for j, result in enumerate(batch_results):
                idx = i + j
                result_df.loc[idx, 'label'] = result['label']

            processed_count += len(batch_results)

            # 定期保存
            if output_path and processed_count % save_interval == 0:
                result_df.to_csv(output_path, index=False)
                print(f"已保存进度: {processed_count}/{len(texts)}")

            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 最终保存
        if output_path:
            result_df.to_csv(output_path, index=False)
            print(f"分类完成！结果已保存到: {output_path}")

        return result_df

    def _extract_deepseek_classification(self, response: str, valid_categories: List[str]) -> str:
        """从响应中提取分类结果"""
        if not response:
            return valid_categories[0] if valid_categories else "unknown"

        response = response.strip()

        # DeepSeek-R1特有的思考标签处理
        if response.find("</think>") != -1:
            response = response.split("</think>")[-1].strip()

        # 直接匹配类别
        if valid_categories:
            for category in valid_categories:
                if category.lower() in response.lower():
                    return category

        # 如果没有找到匹配，返回第一个有效类别
        return valid_categories[0] if valid_categories else "unknown"

    def get_memory_usage(self):
        """获取显存使用情况"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 ** 3  # GB
        return 0


# 使用示例
def main():
    # 初始化分类器
    classifier = DeepSeekBatchTextClassifier()

    # 定义系统提示和标签
    system_prompt = """你是一个专业的文本分类器。请对给定的西班牙语文本进行主题分类。
    可选类别：政治、体育、科技、娱乐、经济
    请直接输出最合适的类别名称。"""

    labels_list = ["政治", "体育", "科技", "娱乐", "经济"]

    # 处理test.csv文件，输出只包含id和label的结果
    result_df = classifier.classify_csv(
        csv_path="test.csv",
        text_column="text",  # 用于分类的文本列
        id_column="id",  # ID列名
        output_path="results.csv",  # 输出文件，只包含id和label两列
        batch_size=8,  # 根据显存调整
        system_prompt=system_prompt,
        labels_list=labels_list,
        save_interval=50  # 每50条保存一次
    )

    print("分类完成！")
    print(f"输出文件包含 {len(result_df)} 条记录")
    print(f"输出文件列名: {list(result_df.columns)}")
    print(f"显存使用: {classifier.get_memory_usage():.2f} GB")


if __name__ == "__main__":
    main()