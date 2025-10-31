import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# 让我们用一个具体的例子来说明 predictions 是如何构造的

def explain_causal_lm_predictions():
    """
    详细解释 causal LM 中 predictions 的构造过程
    """
    
    # 假设我们有一个简单的例子
    print("=== Causal Language Model Predictions 构造过程 ===\n")
    
    # 1. 输入文本示例
    input_text = "The cat is sitting on the"
    target_text = "The cat is sitting on the mat"
    
    print(f"输入文本: '{input_text}'")
    print(f"目标文本: '{target_text}'")
    print()
    
    # 模拟 tokenizer 的处理
    # 假设我们的词汇表很小，只有这些 tokens
    vocab = {
        "<pad>": 0, "<eos>": 1, "The": 2, "cat": 3, "is": 4, 
        "sitting": 5, "on": 6, "the": 7, "mat": 8, "dog": 9, "chair": 10
    }
    id2token = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    
    # 2. Token化过程
    input_tokens = ["The", "cat", "is", "sitting", "on", "the"]
    target_tokens = ["The", "cat", "is", "sitting", "on", "the", "mat"]
    
    input_ids = [vocab[token] for token in input_tokens]
    target_ids = [vocab[token] for token in target_tokens]
    
    print(f"输入 tokens: {input_tokens}")
    print(f"输入 token_ids: {input_ids}")
    print(f"目标 tokens: {target_tokens}")
    print(f"目标 token_ids: {target_ids}")
    print()
    
    # 3. 模拟模型的前向传播过程
    seq_len = len(target_ids)  # 7
    batch_size = 1
    
    print(f"序列长度: {seq_len}")
    print(f"词汇表大小: {vocab_size}")
    print(f"Batch 大小: {batch_size}")
    print()
    
    # 4. 模拟模型输出 logits
    # 对于每个位置，模型输出整个词汇表上的概率分布
    print("=== 模型预测过程 (位置级别) ===")
    
    # 模拟 logits: (batch_size, seq_len, vocab_size)
    # 这里我们手动构造一些合理的 logits
    logits = np.random.randn(batch_size, seq_len, vocab_size)
    
    # 为了演示，我们让正确答案的 logit 更高
    for pos in range(seq_len):
        if pos < len(target_ids):
            correct_token_id = target_ids[pos]
            logits[0, pos, correct_token_id] += 3.0  # 增加正确token的logit
    
    print(f"Logits shape: {logits.shape}")
    print(f"这表示: batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}")
    print()
    
    # 5. 详细解释每个位置的预测
    print("=== 每个位置的预测详情 ===")
    
    for pos in range(seq_len):
        print(f"\n位置 {pos}:")
        
        # 当前位置的 logits
        pos_logits = logits[0, pos, :]  # shape: (vocab_size,)
        
        # 转换为概率
        pos_probs = torch.softmax(torch.tensor(pos_logits), dim=0)
        
        # 获取top-3预测
        top3_indices = torch.topk(pos_probs, 3).indices
        top3_probs = torch.topk(pos_probs, 3).values
        
        print(f"  输入上下文: {' '.join(target_tokens[:pos]) if pos > 0 else '<start>'}")
        print(f"  真实下一个token: {target_tokens[pos] if pos < len(target_tokens) else '<eos>'}")
        print(f"  Top-3 预测:")
        
        for i, (idx, prob) in enumerate(zip(top3_indices, top3_probs)):
            token = id2token[idx.item()]
            print(f"    {i+1}. {token}: {prob:.3f}")
    
    print("\n" + "="*60)
    
    # 6. 解释 eval_preds.predictions 的构造
    print("=== eval_preds.predictions 的构造 ===")
    
    print(f"""
在 Hugging Face Trainer 的评估过程中：

1. **模型前向传播**:
   - 输入: token_ids shape = (batch_size, seq_len)
   - 输出: logits shape = (batch_size, seq_len, vocab_size)

2. **predictions 就是这些 logits**:
   - eval_preds.predictions = logits
   - Shape: (batch_size, seq_len, vocab_size)
   - 每个位置包含对整个词汇表的未归一化分数

3. **具体含义**:
   - predictions[i, j, k] = 第i个样本，第j个位置，第k个token的logit分数
   - 通过 softmax 可以转换为概率分布
   - 通过 argmax 可以得到最可能的token
    """)
    
    # 7. 演示如何从 predictions 中提取有用信息
    print("\n=== 从 predictions 中提取信息 ===")
    
    # 转换为概率
    probs = torch.softmax(torch.tensor(logits), dim=-1)
    print(f"概率分布 shape: {probs.shape}")
    
    # 获取每个位置最可能的token
    predicted_ids = torch.argmax(probs, dim=-1)
    print(f"预测的token IDs: {predicted_ids[0].tolist()}")
    
    # 转换为tokens
    predicted_tokens = [id2token[id.item()] for id in predicted_ids[0]]
    print(f"预测的tokens: {predicted_tokens}")
    
    # 8. 在文本分类任务中的特殊处理
    print("\n=== 文本分类任务中的处理 ===")
    
    print(f"""
对于你的文本分类任务：

1. **训练数据格式**:
   输入: "System: You are a classifier... User: [Spanish text] Assistant: 5"
   
2. **模型需要学会**:
   在 "Assistant: " 之后生成正确的数字(0-42)

3. **predictions 包含**:
   - 整个序列每个位置的预测
   - 包括系统提示、用户输入、助手回复的所有token位置

4. **评估时的挑战**:
   - 只关心 "Assistant: " 后面的数字预测
   - 需要从完整的 predictions 中提取相关部分
   - 需要将token概率转换为分类结果
    """)

# 运行演示
explain_causal_lm_predictions()

# 补充：实际的 compute_metrics 实现思路
def improved_compute_metrics_explanation():
    """
    改进的 compute_metrics 实现思路
    """
    print("\n" + "="*60)
    print("=== 改进的 compute_metrics 实现思路 ===")
    
    print("""
实际实现中，我们需要：

1. **定位助手回复部分**:
   - 找到 "assistant" token 的位置
   - 提取该位置之后的预测

2. **处理数字预测**:
   - 数字可能被分解为多个 token (如 "1", "2" 或 "42")
   - 需要处理 tokenizer 的具体行为

3. **更精确的方法**:
   - 预先记录每个样本中助手回复的起始位置
   - 直接从该位置提取预测
   - 避免字符串解析的不准确性
    """)

improved_compute_metrics_explanation()

# 最佳实践的 compute_metrics 实现
def better_compute_metrics(eval_preds):
    """
    更好的 compute_metrics 实现
    """
    predictions = eval_preds.predictions  # (batch_size, seq_len, vocab_size)
    labels = eval_preds.label_ids         # (batch_size, seq_len)
    
    predicted_labels = []
    true_labels = []
    
    for batch_idx in range(predictions.shape[0]):
        # 1. 找到标签中非-100的部分（这是我们要预测的部分）
        sample_labels = labels[batch_idx]
        valid_positions = sample_labels != -100
        
        if not valid_positions.any():
            continue
            
        # 2. 获取对应位置的预测
        sample_predictions = predictions[batch_idx]  # (seq_len, vocab_size)
        
        # 3. 在有效位置上获取最可能的token
        valid_predictions = sample_predictions[valid_positions]  # (valid_len, vocab_size)
        predicted_token_ids = torch.argmax(torch.tensor(valid_predictions), dim=-1)
        
        # 4. 将token转换为文本并提取数字
        predicted_text = tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
        true_text = tokenizer.decode(sample_labels[valid_positions], skip_special_tokens=True)
        
        # 5. 提取数字（这里需要根据你的具体格式调整）
        import re
        pred_numbers = re.findall(r'\d+', predicted_text)
        true_numbers = re.findall(r'\d+', true_text)
        
        if pred_numbers and true_numbers:
            try:
                predicted_labels.append(int(pred_numbers[0]))
                true_labels.append(int(true_numbers[0]))
            except:
                pass
    
    # 计算指标
    if len(predicted_labels) > 0:
        accuracy = sum(p == t for p, t in zip(predicted_labels, true_labels)) / len(predicted_labels)
        return {'accuracy': accuracy}
    else:
        return {'accuracy': 0.0}

print(f"""

=== 总结 ===

1. **predictions 的本质**:
   - 是模型对每个位置、每个可能token的未归一化分数
   - Shape: (batch_size, seq_len, vocab_size)

2. **关键挑战**:
   - 从序列级预测中提取分类结果
   - 处理tokenization的复杂性
   - 准确定位目标预测位置

3. **建议**:
   - 使用更精确的位置定位
   - 考虑数字token的具体表示
   - 添加充分的错误处理
""")