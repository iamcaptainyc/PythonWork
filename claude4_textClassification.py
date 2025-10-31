import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from typing import List, Dict, Union
import re


class DeepSeekTextClassifier:
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

    def classify_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """情感分类：积极、消极、中性"""
        prompt = f"""请分析以下文本的情感倾向，只能从以下选项中选择一个：积极、消极、中性

文本：{text}

情感分类："""

        result = self._generate_response(prompt, max_new_tokens=10)

        # 提取分类结果
        sentiment = self._extract_classification(result, ["积极", "消极", "中性"])
        confidence = self._calculate_confidence(result, sentiment)

        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "raw_output": result
        }

    def classify_topic(self, text: str, topics: List[str]) -> Dict[str, Union[str, float]]:
        """主题分类"""
        topics_str = "、".join(topics)

        prompt = f"""请将以下文本分类到最合适的主题类别中，只能从以下选项中选择一个：{topics_str}

文本：{text}

主题分类："""

        result = self._generate_response(prompt, max_new_tokens=20)

        # 提取分类结果
        topic = self._extract_classification(result, topics)
        confidence = self._calculate_confidence(result, topic)

        return {
            "text": text,
            "topic": topic,
            "confidence": confidence,
            "available_topics": topics,
            "raw_output": result
        }

    def classify_intent(self, text: str) -> Dict[str, Union[str, float]]:
        """意图分类：问询、投诉、建议、赞扬、其他"""
        intents = ["问询", "投诉", "建议", "赞扬", "其他"]

        prompt = f"""请分析以下文本的用户意图，只能从以下选项中选择一个：问询、投诉、建议、赞扬、其他

文本：{text}

意图分类："""

        result = self._generate_response(prompt, max_new_tokens=10)

        # 提取分类结果
        intent = self._extract_classification(result, intents)
        confidence = self._calculate_confidence(result, intent)

        return {
            "text": text,
            "intent": intent,
            "confidence": confidence,
            "raw_output": result
        }

    def classify_spam(self, text: str) -> Dict[str, Union[str, bool, float]]:
        """垃圾信息检测"""
        prompt = f"""请判断以下文本是否为垃圾信息（广告、诈骗、垃圾邮件等），只能回答：是 或 否

文本：{text}

是否为垃圾信息："""

        result = self._generate_response(prompt, max_new_tokens=5)

        # 提取分类结果
        is_spam_str = self._extract_classification(result, ["是", "否"])
        is_spam = is_spam_str == "是"
        confidence = self._calculate_confidence(result, is_spam_str)

        return {
            "text": text,
            "is_spam": is_spam,
            "confidence": confidence,
            "raw_output": result
        }

    def custom_classify(self, text: str, categories: List[str], instruction: str = None) -> Dict[
        str, Union[str, float]]:
        """自定义分类"""
        categories_str = "、".join(categories)

        if instruction:
            prompt = f"""{instruction}

只能从以下选项中选择一个：{categories_str}

文本：{text}

分类结果："""
        else:
            prompt = f"""请将以下文本分类到最合适的类别中，只能从以下选项中选择一个：{categories_str}

文本：{text}

分类结果："""

        result = self._generate_response(prompt, max_new_tokens=20)

        # 提取分类结果
        category = self._extract_classification(result, categories)
        confidence = self._calculate_confidence(result, category)

        return {
            "text": text,
            "category": category,
            "confidence": confidence,
            "available_categories": categories,
            "raw_output": result
        }

    def batch_classify(self, texts: List[str], classify_type: str = "sentiment", **kwargs) -> List[Dict]:
        """批量分类"""
        results = []

        for text in texts:
            if classify_type == "sentiment":
                result = self.classify_sentiment(text)
            elif classify_type == "topic":
                result = self.classify_topic(text, kwargs.get("topics", []))
            elif classify_type == "intent":
                result = self.classify_intent(text)
            elif classify_type == "spam":
                result = self.classify_spam(text)
            elif classify_type == "custom":
                result = self.custom_classify(
                    text,
                    kwargs.get("categories", []),
                    kwargs.get("instruction")
                )
            else:
                raise ValueError(f"不支持的分类类型: {classify_type}")

            results.append(result)

        return results

    def _generate_response(self, prompt: str, max_new_tokens: int = 50) -> str:
        """生成模型响应"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 解码响应
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取新生成的部分
        response = full_response[len(prompt):].strip()

        return response

    def generate_model_response(self, system_prompt, user_prompt):

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    def _extract_classification(self, response: str, valid_categories: List[str]) -> str:
        """从响应中提取分类结果"""
        response = response.strip()

        # 直接匹配
        for category in valid_categories:
            if category in response:
                return category

        # 如果没有找到匹配，返回第一个有效类别
        return valid_categories[0] if valid_categories else "未知"

    def _calculate_confidence(self, response: str, predicted_category: str) -> float:
        """计算置信度（简单的启发式方法）"""
        if not response or not predicted_category:
            return 0.5

        # 如果响应很短且包含预测类别，置信度较高
        if len(response) <= 10 and predicted_category in response:
            return 0.9

        # 如果响应包含预测类别
        if predicted_category in response:
            return 0.8

        # 默认置信度
        return 0.6


# 使用示例
def main():
    # 初始化分类器
    print("初始化DeepSeek文本分类器...")
    classifier = DeepSeekTextClassifier()

    # 测试文本
    test_texts = [
        "这个产品真的很棒，我非常满意！",
        "客服态度太差了，完全不解决问题",
        "请问你们的营业时间是什么？",
        "恭喜您中奖了！请点击链接领取奖品",
        "今天天气不错，适合出门走走"
    ]

    print("\n=== 情感分类测试 ===")
    for text in test_texts[:3]:
        result = classifier.classify_sentiment(text)
        print(f"文本: {result['text']}")
        print(f"情感: {result['sentiment']} (置信度: {result['confidence']:.2f})")
        print()

    print("=== 意图分类测试 ===")
    for text in test_texts[:3]:
        result = classifier.classify_intent(text)
        print(f"文本: {result['text']}")
        print(f"意图: {result['intent']} (置信度: {result['confidence']:.2f})")
        print()

    print("=== 垃圾信息检测测试 ===")
    for text in test_texts[3:5]:
        result = classifier.classify_spam(text)
        print(f"文本: {result['text']}")
        print(f"是否垃圾信息: {result['is_spam']} (置信度: {result['confidence']:.2f})")
        print()

    print("=== 主题分类测试 ===")
    topics = ["科技", "娱乐", "体育", "新闻", "生活"]
    result = classifier.classify_topic("苹果发布了新款iPhone", topics)
    print(f"文本: {result['text']}")
    print(f"主题: {result['topic']} (置信度: {result['confidence']:.2f})")
    print()

    print("=== 自定义分类测试 ===")
    categories = ["正式", "非正式"]
    instruction = "请判断以下文本的语言风格是正式还是非正式"
    result = classifier.custom_classify("您好，请问有什么可以帮助您的吗？", categories, instruction)
    print(f"文本: {result['text']}")
    print(f"语言风格: {result['category']} (置信度: {result['confidence']:.2f})")
    print()

    print("=== 批量分类测试 ===")
    batch_results = classifier.batch_classify(test_texts[:3], "sentiment")
    for result in batch_results:
        print(f"{result['text']} -> {result['sentiment']}")


if __name__ == "__main__":
    main()