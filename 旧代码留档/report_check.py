import os
import re
import math
import pkg_resources
import torch
import jieba
import time
import chardet
import hashlib

import pandas as pd
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

STOPWORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', 
    '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '样', '看', '过', '用', '她', '他', '我们', '这', '那', 
    '中', '下', '为', '来', '还', '又', '对', '能', '可以', '吧', '啊', '吗', '呢', '啦', '给', '等', '与', '跟', '让', 
    '被', '把', '而', '且', '或', '如果', '因为', '所以', '但是', '然后', '虽然', '即使', '并且', '或者', '因此', '例如', 
    '比如', '首先', '其次', '最后', '结果', '实验', '数据', '分析', '步骤', '方法', '目的', '要求', '内容', '原理', '仪器', 
    '设备', '操作', '记录', '结果', '讨论', '结论', '报告', '小组', '成员'
])


class ReportGroup:
    def __init__(self, group_dir):
        self.group_name = os.path.basename(group_dir)
        self.report_dir = os.path.join(group_dir, "实验报告", "文字")
        self.report_path = self._find_report_text()
        self.content = ""
        self.length = 0
        self.hash = ""
        self._load_report()


    def _find_report_text(self):
        if not os.path.exists(self.report_dir):
            return None
        
        text_path = os.path.join(self.report_dir, "完整文本.txt")
        if os.path.exists(text_path):
            return text_path
        
        return None

    def _load_report(self):
        if not self.report_path:
            return 

        try:
            with open(self.report_path, 'rb') as f:
                raw_data = f.read(1024)
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'gbk'

            with open(self.report_path, 'r', encoding=encoding, errors='replace') as f:
                self.content = f.read()
                self.length = len(self.content)
                self.hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()
        except Exception as e:
            print(f"加载失败 {self.report_path}：{str(e)}")

    def preprocess(self, for_winnowing=False):
        if not self.content:
            return ""


        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', self.content)
        if for_winnowing:
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        
        
        # 移除特殊字符和数字
        text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # 中文分词
        words = jieba.lcut(text)
        
        # 去除停用词和单字词
        filtered_words = [
            word for word in words 
            if len(word) > 1 and word not in STOPWORDS
        ]
        
        return " ".join(filtered_words[-4096:])


class ReportDataLoader:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.groups = self._load_groups()

    def _load_groups(self):
        groups = {}
        valid_groups = 0

        for group_name in os.listdir(self.root_dir):
            group_dir = os.path.join(self.root_dir, group_name)
            if os.path.isdir(group_dir):
                group_report = ReportGroup(group_dir)
                if group_report.content:
                    groups[group_report.group_name] = group_report
                    valid_groups += 1
        
        print(f"加载了 {valid_groups} 个小组的报告数据")
        return groups


class ReportSimilarityAnalyzer:
    def __init__(self, dataloader):
        self.groups = dataloader.groups
        self.group_names = list(self.groups.keys())
        self.results = []

    def analyze(self):
        raise NotImplementedError("子类需要实现此方法")

    def get_similar_pairs(self, threshold=0.7):
        return [pair for pair in self.results if pair["similarity"] >= threshold]
    
    def get_top_pairs(self, top_n=10):
        """获取相似度最高的top_n对"""
        sorted_results = sorted(self.results, key=lambda x: x["similarity"], reverse=True)
        return sorted_results[:top_n]


# ===============================================================
# ===============================================================
# 基础TFIDF方法

# class report_TFIDF(ReportSimilarityAnalyzer):
#     def __init__(self, dataloader):
#         super().__init__(dataloader)
#         self.vectorizer = TfidfVectorizer(
#             ngram_range=(2,4),
#             min_df=2,
#             max_features=100000
#         )

#     def analyze(self):
#         texts = [group.preprocess() for group in self.groups.values()]

#         tfidf_matrix = self.vectorizer.fit_transform(texts)

#         similarity_matrix = cosine_similarity(tfidf_matrix)

#         self.results = []
#         for i in range(len(self.group_names)):
#             for j in range(i+1, len(self.group_names)):
#                 self.results.append({
#                     "method": "tfidf",
#                     "group1": self.group_names[i],
#                     "group2": self.group_names[j],
#                     "similarity": similarity_matrix[i][j]
#                 })
#         return self
    
# ===============================================================
# ===============================================================
# 基于bert的方法

# class report_BERT(ReportSimilarityAnalyzer):
#     def __init__(self, dataloader):
#         super().__init__(dataloader)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self._init_model()

#     def _init_model(self):
#         """初始化BERT模型"""
#         self.model = SentenceTransformer("/data_new/scy/paraphrase-multilingual-MiniLM-L12-v2").to(self.device)
        
#     def analyze(self):
#         """执行BERT相似度分析"""
#         texts = [group.preprocess() for group in self.groups.values()]
        
#         # 获取嵌入
#         embeddings = self.model.encode(
#             texts, 
#             batch_size=8,
#             max_length = 20000,
#             show_progress_bar=True,
#             convert_to_tensor=True
#         )
        
#         # 计算余弦相似度
#         similarity_matrix = cosine_similarity(embeddings.cpu().numpy())
        
#         # 收集结果
#         self.results = []
#         for i in range(len(self.group_names)):
#             for j in range(i+1, len(self.group_names)):
#                 self.results.append({
#                     "method": "bert",
#                     "group1": self.group_names[i],
#                     "group2": self.group_names[j],
#                     "similarity": similarity_matrix[i][j]
#                 })
        
#         return self

# ===============================================================
# ===============================================================
# 大模型方法

# class report_LLM(ReportSimilarityAnalyzer):
#     def __init__(self, dataloader):
#         super().__init__(dataloader)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self._init_model()
    
#     def _init_model(self):
#         self.tokenizer = AutoTokenizer.from_pretrained("/data_new/scy/longformer-chinese-base-4096")
#         self.model = AutoModel.from_pretrained("/data_new/scy/longformer-chinese-base-4096").to(self.device)

#     def get_embedding(self, text):
#         inputs = self.tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             max_length=4096,
#             padding=True
#         ).to(self.device)

#         with torch.no_grad():
#             output = self.model(**inputs)

#         return output.last_hidden_state.mean(dim=1).cpu().numpy()
    

#     def analyze(self):
#         embeddings = {}
        
#         for group_name, group in tqdm(self.groups.items(), desc="计算嵌入"):
#             text = group.preprocess()
#             text = text[:4000:-1]
#             # print(len(text))
#             embeddings[group_name] = self.get_embedding(text)
#         # exit(0)
#         self.results = []
#         group_names = list(embeddings.keys())

#         for i in range(len(group_names)):
#             for j in range(i+1, len(group_names)):
#                 emb1 = embeddings[group_names[i]]
#                 emb2 = embeddings[group_names[j]]

#                 similarity = cosine_similarity(emb1, emb2)[0][0]

#                 self.results.append({
#                     "method": "bert",
#                     "group1": self.group_names[i],
#                     "group2": self.group_names[j],
#                     "similarity": similarity
#                 })
        
#         return self

# ===============================================================
# ===============================================================
# 基于winnowing的方法

class Winnowing:
    def __init__(self, k=5, w=10, robust=True):
        self.k = k
        self.w = w
        self.t = w + k - 1
        self.robust = robust

    def _compute_hashes(self, text):
        n = len(text)
        hashes = []
        base = 257
        mod = (1 << 64) - 1

        if n < self.k:
            return []
        
        current_hash = 0
        for i in range(self.k):
            current_hash = (current_hash * base + ord(text[i])) % mod
        hashes.append(current_hash)

        high_power = pow(base, self.k - 1, mod)
        for i in range(self.k, n):
            current_hash = (current_hash - ord(text[i - self.k]) * high_power) % mod
            current_hash = (current_hash * base + ord(text[i])) % mod
            hashes.append(current_hash)

        return hashes


    def figerprint(self, text):
        if len(text) < self.k:
            return set()
        
        hashes = self._compute_hashes(text)
        n = len(hashes)
        fingerprints = set()

        if n < self.w:
            min_hash = min(hashes) if hashes else 0
            return {min_hash}
        
        min_index = -1

        for start in range(n - self.w + 1):
            end = start + self.w - 1

            if min_index < start:
                min_index = start
                for j in range(start + 1, end + 1):
                    if self.robust:
                        if hashes[j] < hashes[min_index]:
                            min_index = j
                    else:
                        if hashes[j] <= hashes[min_index]:
                            min_index = j
            else:
                if self.robust:
                    if hashes[end] < hashes[min_index]:
                        min_index = end
                else:
                    if hashes[end] <= hashes[min_index]:
                        min_index = end
            
            fingerprints.add(hashes[min_index])

        return fingerprints

# class TFIDFWinnowing(Winnowing):
#     def __init__(self, k=5, w=10, robust=True, corpus=None):
#         super().__init__(k, w, robust)
#         self.corpus=corpus
#         self.kgram_idf={}

#     def build_tfidf_model(self):
#         if not self.corpus:
#             raise ValueError("词库未提供")
        
#         all_kgrams = set()
#         for doc in self.corpus:
#             for i in range(len(doc) - self.k+1):
#                 all_kgrams.add(doc[i:i+self.k])

#         df = defaultdict(int)
#         for kg in all_kgrams:
#             for doc in self.corpus:
#                 if kg in doc:
#                     df[kg] += 1

#         N = len(self.corpus)
#         self.kgram_idf = {kg: math.log((N+1)/(df[kg]+1))+1 for kg in df}    

#     def weighted_figerprint(self, text):
#         if not self.kgram_idf:
#             self.build_tfidf_model()

#         hashes = self._compute_hashes(text)
#         n= len(hashes)
#         if n < self.w:
#             return set()

#         # 计算每个k-gram的TF-IDF权重
#         weights = []
#         for i in range(len(text) - self.k + 1):
#             kgram = text[i:i+self.k]
#             # 简化TF计算：只考虑当前文档中出现的次数
#             tf = text.count(kgram)  
#             weights.append(tf * self.kgram_idf.get(kgram, 1.0))
        
#         fingerprints = set()
        
#         # 滑动窗口选择加权最小哈希
#         for start in range(n - self.w + 1):
#             end = start + self.w - 1
#             window_hashes = hashes[start:end+1]
#             window_weights = weights[start:end+1]
            
#             # 计算加权哈希值
#             weighted_hashes = [h * (1.0 / (w + 1e-5)) for h, w in zip(window_hashes, window_weights)]
            
#             # 选择加权最小哈希
#             min_index = start + np.argmin(weighted_hashes)
#             fingerprints.add(hashes[min_index])
        
#         return fingerprints
    
# class ContextAwareWinnowing(Winnowing):
#     def __init__(self, k=5, w=10, robust=True, model_name="/data_new/Qwen/Qwen2-7B"):
#         super().__init__(k, w, robust)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model_name = model_name
#         self._init_model()
        
#         # 哈希层
#         self.hash_layer = nn.Sequential(
#             nn.Linear(self.model.config.hidden_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, 32),
#             nn.Tanh()
#         ).to(self.device)
        
#         # 初始化哈希层权重
#         for layer in self.hash_layer:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_uniform_(layer.weight)
#                 nn.init.zeros_(layer.bias)
    
#     def _init_model(self):
#         """初始化 Qwen 模型"""
#         print(f"加载 Qwen 模型: {self.model_name}")
        
#         # 特殊处理 Qwen 模型
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.model_name,
#             trust_remote_code=True
#         )
        
#         # 使用 bfloat16 精度节省显存
#         self.model = AutoModel.from_pretrained(
#             self.model_name,
#             device_map="auto",
#             torch_dtype=torch.bfloat16,
#             trust_remote_code=True
#         ).eval()
        
#         # 获取模型配置
#         self.model_config = self.model.config
#         print(f"模型加载完成，隐藏层大小: {self.model_config.hidden_size}, 最大长度: {self.model_config.max_position_embeddings}")
    
#     def _compute_contextual_hashes(self, text):
#         """使用 Qwen 模型生成上下文感知哈希"""
#         if not text:
#             return []
        
#         n = len(text)
#         hashes = []
        
#         # 处理整个文本获取上下文嵌入
#         try:
#             inputs = self.tokenizer(
#                 text, 
#                 return_tensors="pt", 
#                 truncation=True, 
#                 max_length=self.model_config.max_position_embeddings,
#                 padding=True,
#                 return_offsets_mapping=True  # 获取偏移映射
#             ).to(self.device)
#         except Exception as e:
#             print(f"Tokenizer 错误: {str(e)}")
#             return []
        
#         with torch.no_grad():
#             try:
#                 outputs = self.model(**inputs)
#                 embeddings = outputs.last_hidden_state[0].float()  # 转换为 float32 计算
#             except Exception as e:
#                 print(f"模型推理错误: {str(e)}")
#                 return []

#         获取偏移映射
#         offset_mapping = inputs['offset_mapping'][0].cpu().numpy()
        
#         # 为每个 k-gram 生成哈希
#         for start in range(n - self.k + 1):
#             end = start + self.k - 1
            
#             # 找到包含当前字符的token
#             token_indices = set()
#             for char_idx in range(start, end + 1):
#                 # 找到包含该字符的token
#                 for token_idx, (token_start, token_end) in enumerate(offset_mapping):
#                     if token_start <= char_idx < token_end:
#                         token_indices.add(token_idx)
#                         break
            
#             if not token_indices:
#                 # 使用整个文档的平均嵌入作为后备
#                 kg_embedding = embeddings.mean(dim=0)
#             else:
#                 # 获取对应的嵌入
#                 token_indices = sorted(token_indices)
#                 token_embeddings = embeddings[token_indices]
#                 kg_embedding = token_embeddings.mean(dim=0)
            
#             # 生成哈希 - 使用更复杂的方法
#             with torch.no_grad():
#                 hash_vec = self.hash_layer(kg_embedding.unsqueeze(0))
#                 hash_value = hash_vec.sum().item()
#                 # 将浮点数转换为整数哈希
#                 hash_bytes = hash_vec.cpu().numpy().tobytes()
#                 hash_int = int.from_bytes(hash_bytes, byteorder='big') % (2**64)
            
#             hashes.append(hash_int)
        
#         return hashes

#     def contextual_fingerprint(self, text):
#         """使用上下文感知哈希生成指纹，支持长文档分段处理"""
#         if not text:
#             return set()
        
#         # 如果文本太长，使用滑动窗口处理
#         max_length = self.model_config.max_position_embeddings
#         if len(text) <= max_length:
#             return self._generate_fingerprint(text)
        
#         # 长文档处理：分段处理然后合并指纹
#         fingerprints = set()
#         segment_size = max_length - self.k  # 保留重叠区域
#         overlap = self.w + self.k - 1  # 确保窗口重叠
        
#         for start in range(0, len(text), segment_size):
#             end = min(start + max_length, len(text))
#             segment = text[start:end]
            
#             # 确保重叠区域
#             if start > 0:
#                 segment = text[start - overlap:end]
            
#             segment_fp = self._generate_fingerprint(segment)
#             fingerprints.update(segment_fp)
        
#         return fingerprints
    
#     def _generate_fingerprint(self, text):
#         """生成单个文本片段的指纹"""
#         hashes = self._compute_contextual_hashes(text)
#         n = len(hashes)
        
#         # 打印调试信息
#         print(f"文本长度: {len(text)}, k-gram数量: {n}")
#         if n > 0:
#             print(f"前5个哈希值: {hashes[:5]}")
        
#         if n < self.w:
#             # 如果哈希值数量小于窗口大小，返回所有哈希值
#             if n == 0:
#                 return set()
#             min_hash = min(hashes)
#             return {min_hash}
        
#         fingerprints = set()
#         min_index = -1
        
#         # 使用原始Winnowing算法选择指纹
#         for start in range(n - self.w + 1):
#             end = start + self.w - 1
            
#             if min_index < start:
#                 min_index = start
#                 for j in range(start + 1, end + 1):
#                     if self.robust:
#                         if hashes[j] < hashes[min_index]:
#                             min_index = j
#                     else:
#                         if hashes[j] <= hashes[min_index]:
#                             min_index = j
#             else:
#                 if self.robust:
#                     if hashes[end] < hashes[min_index]:
#                         min_index = end
#                 else:
#                     if hashes[end] <= hashes[min_index]:
#                         min_index = end
            
#             fingerprints.add(hashes[min_index])
        
#         print(f"生成的指纹数量: {len(fingerprints)}")
#         return fingerprints


# class ContextAwareWinnowing(Winnowing):
#     def __init__(self, k=5, w=10, robust=True, model_name="/data_new/Qwen/Qwen2-7B", batch_size=128):
#         """
#         优化版的上下文感知Winnowing算法
        
#         参数:
#             k: k-gram长度
#             w: 窗口大小
#             robust: 是否使用鲁棒模式
#             model_name: 模型名称 (推荐使用较小模型)
#             batch_size: k-gram批处理大小
#         """
#         super().__init__(k, w, robust)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model_name = model_name
#         self.batch_size = batch_size
#         self._init_model()
        
#         # 更轻量的哈希层
#         self.hash_layer = nn.Sequential(
#             nn.Linear(self.model.config.hidden_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, 32),
#             nn.Tanh()
#         ).to(self.device)
        
#         # 初始化哈希层权重
#         for layer in self.hash_layer:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_uniform_(layer.weight)
#                 nn.init.zeros_(layer.bias)
    
#     def _init_model(self):
#         """初始化优化后的模型"""
#         print(f"加载优化模型: {self.model_name}")
        
#         # 使用更小的模型
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.model_name,
#             trust_remote_code=True
#         )
        
#         # 使用量化技术加速推理
#         self.model = AutoModel.from_pretrained(
#             self.model_name,
#             device_map="auto",
#             torch_dtype=torch.bfloat16,
#             trust_remote_code=True
#         ).eval()
        
#         # 获取模型配置
#         self.model_config = self.model.config
#         print(f"模型加载完成，隐藏层大小: {self.model_config.hidden_size}, 最大长度: {self.model_config.max_position_embeddings}")
    
#     def _compute_contextual_hashes(self, text):
#         """使用批处理和向量化优化哈希计算"""
#         if not text or len(text) < self.k:
#             return []
        
#         n = len(text)
        
#         # 1. 对整个文本进行编码
#         start_time = time.time()
#         try:
#             inputs = self.tokenizer(
#                 text, 
#                 return_tensors="pt", 
#                 truncation=True, 
#                 max_length=self.model_config.max_position_embeddings,
#                 padding=True,
#                 return_offsets_mapping=True
#             ).to(self.device)
#         except Exception as e:
#             print(f"Tokenizer 错误: {str(e)}")
#             return []
        
#         with torch.no_grad():
#             try:
#                 outputs = self.model(**inputs)
#                 embeddings = outputs.last_hidden_state[0].float()  # 转换为 float32 计算
#             except Exception as e:
#                 print(f"模型推理错误: {str(e)}")
#                 return []
        
#         token_time = time.time() - start_time
#         # print(f"模型推理时间: {token_time:.2f}秒")
        
#         # 2. 构建字符到token的映射
#         offset_mapping = inputs['offset_mapping'][0].cpu().numpy()
#         char_to_token = np.full(len(text), -1, dtype=np.int32)
        
#         for token_idx, (start, end) in enumerate(offset_mapping):
#             start, end = int(start), int(end)
#             if start < len(char_to_token):
#                 char_to_token[start:min(end, len(char_to_token))] = token_idx
        
#         # 3. 准备批处理数据
#         num_kg = n - self.k + 1
#         kg_indices = []
        
#         for start in range(num_kg):
#             end = start + self.k - 1
#             token_indices = char_to_token[start:end+1]
#             token_indices = token_indices[token_indices != -1]  # 移除无效索引
#             kg_indices.append(token_indices)
        
#         # 4. 批处理计算k-gram嵌入
#         kg_embeddings = []
#         start_time = time.time()
        
#         # 使用批处理计算平均嵌入
#         for i in range(0, num_kg, self.batch_size):
#             batch_indices = kg_indices[i:i+self.batch_size]
#             batch_embeds = []
            
#             for indices in batch_indices:
#                 if len(indices) == 0:
#                     # 使用全局平均作为后备
#                     batch_embeds.append(embeddings.mean(dim=0))
#                 else:
#                     # 获取唯一的token索引并排序
#                     unique_indices = np.unique(indices)
#                     token_embeds = embeddings[unique_indices]
#                     batch_embeds.append(token_embeds.mean(dim=0))
            
#             # 堆叠所有嵌入
#             batch_embeds = torch.stack(batch_embeds)
#             kg_embeddings.append(batch_embeds)
        
#         if kg_embeddings:
#             kg_embeddings = torch.cat(kg_embeddings, dim=0)
#         else:
#             return []
        
#         embed_time = time.time() - start_time
#         # print(f"k-gram嵌入计算时间: {embed_time:.2f}秒, 处理 {num_kg} 个k-gram")
        
#         # 5. 批处理计算哈希值
#         start_time = time.time()
#         with torch.no_grad():
#             hash_vecs = self.hash_layer(kg_embeddings)
        
#         # 向量化哈希计算
#         hash_vecs_np = hash_vecs.cpu().numpy()
#         hashes = []
        
#         # 使用numpy向量化计算哈希值
#         for vec in hash_vecs_np:
#             # 更高效的哈希计算方法
#             hash_bytes = vec.tobytes()
#             hash_int = int.from_bytes(hash_bytes, byteorder='big', signed=False) % (2**64)
#             hashes.append(hash_int)
        
#         hash_time = time.time() - start_time
#         # print(f"哈希计算时间: {hash_time:.2f}秒")
        
#         return hashes

#     def contextual_fingerprint(self, text):
#         """使用上下文感知哈希生成指纹，支持长文档分段处理"""
#         if not text:
#             return set()
        
#         # 如果文本太长，使用滑动窗口处理
#         max_length = min(self.model_config.max_position_embeddings, 4096)  # 限制最大长度
#         if len(text) <= max_length:
#             return self._generate_fingerprint(text)
        
#         # 长文档处理：并行分段处理
#         fingerprints = set()
#         segment_size = max_length - self.k
#         overlap = self.w + self.k - 1  # 确保窗口重叠
#         segments = []
        
#         # 准备分段
#         for start in range(0, len(text), segment_size):
#             end = min(start + max_length, len(text))
#             if start > 0:
#                 seg_start = max(0, start - overlap)
#                 segments.append(text[seg_start:end])
#             else:
#                 segments.append(text[start:end])
        
#         # 并行处理分段
#         with ThreadPoolExecutor(max_workers=4) as executor:
#             results = list(executor.map(self._generate_fingerprint, segments))
        
#         # 合并结果
#         for fp in results:
#             fingerprints.update(fp)
        
#         return fingerprints
    
#     def _generate_fingerprint(self, text):
#         """生成单个文本片段的指纹"""
#         hashes = self._compute_contextual_hashes(text)
#         n = len(hashes)
        
#         if n < self.w:
#             # 如果哈希值数量小于窗口大小，返回所有哈希值
#             if n == 0:
#                 return set()
#             min_hash = min(hashes)
#             return {min_hash}
        
#         fingerprints = set()
#         min_index = -1
        
#         # 使用原始Winnowing算法选择指纹
#         for start in range(n - self.w + 1):
#             end = start + self.w - 1
            
#             if min_index < start:
#                 min_index = start
#                 for j in range(start + 1, end + 1):
#                     if self.robust:
#                         if hashes[j] < hashes[min_index]:
#                             min_index = j
#                     else:
#                         if hashes[j] <= hashes[min_index]:
#                             min_index = j
#             else:
#                 if self.robust:
#                     if hashes[end] < hashes[min_index]:
#                         min_index = end
#                 else:
#                     if hashes[end] <= hashes[min_index]:
#                         min_index = end
            
#             fingerprints.add(hashes[min_index])
        
#         return fingerprints


# class SemanticWinnowing:
#     def __init__(self, k=3, w=5, model_name="/data_new/scy/paraphrase-multilingual-MiniLM-L12-v2", 
#                  hash_bits=64, device="cuda", robust=True):
#         """
#         基于语义的Winnowing算法
#         :param k: k-gram中句子数量
#         :param w: 窗口大小
#         :param model_name: 语义嵌入模型
#         :param hash_bits: 哈希位数
#         :param device: 计算设备
#         :param robust: 是否使用鲁棒模式
#         """
#         self.k = k  # 句子数量
#         self.w = w  # 窗口大小
#         self.hash_bits = hash_bits
#         self.robust = robust
#         self.device = device if torch.cuda.is_available() else "cpu"
        
#         # 加载语义模型
#         self.model = SentenceTransformer(model_name).to(self.device)
#         self.model.eval()
        
#         # 初始化随机投影矩阵
#         self.random_vectors = None
        
#     def _init_random_vectors(self, embedding_dim):
#         """初始化随机投影矩阵"""
#         np.random.seed(42)  # 固定随机种子保证可复现性
#         self.random_vectors = np.random.randn(self.hash_bits, embedding_dim)
        
#     def _embedding_to_hash(self, embedding):
#         """将嵌入向量转换为哈希值"""
#         if self.random_vectors is None:
#             self._init_random_vectors(embedding_dim=embedding.shape[0])
        
#         # 计算投影
#         projection = np.dot(self.random_vectors, embedding)
        
#         # 生成二进制哈希
#         binary_hash = (projection > 0).astype(int)
        
#         # 转换为整数哈希
#         hash_int = 0
#         for bit in binary_hash:
#             hash_int = (hash_int << 1) | bit
            
#         return hash_int
    
#     def _generate_semantic_hashes(self, sentences):
#         """为句子列表生成语义哈希"""
#         if not sentences:
#             return []
        
#         # 批量生成句子嵌入
#         embeddings = self.model.encode(
#             sentences, 
#             batch_size=32,
#             show_progress_bar=False,
#             convert_to_tensor=True
#         ).cpu().numpy()
        
#         # 为每个句子生成哈希
#         return [self._embedding_to_hash(emb) for emb in embeddings]
    
#     def _create_kgram_hashes(self, sentence_hashes):
#         """创建k-gram哈希序列"""
#         if len(sentence_hashes) < self.k:
#             return []
        
#         # 滑动窗口生成k-gram哈希
#         kgram_hashes = []
#         for i in range(len(sentence_hashes) - self.k + 1):
#             # 组合k个句子的哈希
#             combined_hash = 0
#             for j in range(self.k):
#                 combined_hash ^= sentence_hashes[i + j]  # 使用异或组合
            
#             kgram_hashes.append(combined_hash)
        
#         return kgram_hashes
    
#     def fingerprint(self, text):
#         """生成文档的语义指纹"""
#         # 将文本分割成句子
#         sentences = re.split(r'(?<=[。！？；;])', text)
#         sentences = [s.strip() for s in sentences if s.strip()]
        
#         if len(sentences) < self.k:
#             return set()
        
#         # 生成句子级语义哈希
#         sentence_hashes = self._generate_semantic_hashes(sentences)
        
#         # 生成k-gram哈希序列
#         kgram_hashes = self._create_kgram_hashes(sentence_hashes)
        
#         if not kgram_hashes:
#             return set()
        
#         n = len(kgram_hashes)
#         fingerprints = set()
        
#         # 如果哈希序列长度小于窗口大小，返回所有哈希
#         if n < self.w:
#             min_hash = min(kgram_hashes)
#             return {min_hash}
        
#         min_index = -1
        
#         # 应用Winnowing算法
#         for start in range(n - self.w + 1):
#             end = start + self.w - 1
            
#             # 如果最小值已移出窗口，重新扫描
#             if min_index < start:
#                 min_index = start
#                 # 从右向左扫描确保选择最右侧最小值
#                 for j in range(start + 1, end + 1):
#                     if self.robust:
#                         if kgram_hashes[j] < kgram_hashes[min_index]:
#                             min_index = j
#                     else:
#                         if kgram_hashes[j] <= kgram_hashes[min_index]:
#                             min_index = j
            
#             # 检查新加入的右侧元素
#             else:
#                 if self.robust:
#                     if kgram_hashes[end] < kgram_hashes[min_index]:
#                         min_index = end
#                 else:
#                     if kgram_hashes[end] <= kgram_hashes[min_index]:
#                         min_index = end
            
#             fingerprints.add(kgram_hashes[min_index])
        
#         return fingerprints




class report_Winnowing(ReportSimilarityAnalyzer):
    def __init__(self, dataloader, k=10, w=15, robust=True, mode="basic",
                 semantic_model="/data_new/scy/paraphrase-multilingual-MiniLM-L12-v2"):
        super().__init__(dataloader)
        self.k = k
        self.w = w
        self.robust = robust
        self.mode = mode
        self.winnowing = Winnowing(k=k, w=w, robust=robust)
        self.semantic_model = semantic_model
        self.fingerprints = {}
        self.corpus = self._get_corpus()
        self._init_winnowing()

    def _init_winnowing(self):
        if self.mode == "basic":
            self.winnowing = Winnowing(k=self.k, w=self.w, robust=self.robust)
        # elif self.mode == "tfidf":
        #     self.winnowing = TFIDFWinnowing(k=self.k, w=self.w, robust=self.robust, corpus=self.corpus)
        # elif self.mode == "contextual":
            # self.winnowing = ContextAwareWinnowing(k=self.k, w=self.w, robust=self.robust)
        # elif self.mode == "semantic":
        #     self.winnowing = SemanticWinnowing(
        #         k=3,  # 3个句子组成一个k-gram
        #         w=5,  # 窗口大小为5个k-gram
        #         model_name=self.semantic_model,
        #         robust=self.robust
        #     )
        else:
            raise ValueError(f"未知的模式：{self.mode}")

    def _get_corpus(self):
        return [group.preprocess(for_winnowing=True) for group in self.groups.values()]

    def analyze(self):
        for group_name, group in tqdm(self.groups.items(), desc="生成指纹"):
            text = group.preprocess(for_winnowing=True)

            if self.mode == "tfidf":
                self.fingerprints[group_name] = self.winnowing.weighted_figerprint(text)
            elif self.mode == "contextual":
                self.fingerprints[group_name] = self.winnowing.contextual_fingerprint(text)
            elif self.mode == "semantic":
                self.fingerprints[group_name] = self.winnowing.fingerprint(group.content)
            else:
                self.fingerprints[group_name] = self.winnowing.figerprint(text)

        group_names = list(self.fingerprints.keys())
        n = len(group_names)

        self.results = []
        for i in range(n):
            for j in range(i+1, n):
                fp1 = self.fingerprints[group_names[i]]
                fp2 = self.fingerprints[group_names[j]]

                intersection = len(fp1 & fp2)
                union = len(fp1 | fp2)

                similarity = intersection / union if union >0 else 0.0

                self.results.append({
                    "method": f"winnowing(k={self.k},w={self.w})",
                    "group1": group_names[i],
                    "group2": group_names[j],
                    "similarity": similarity,
                    "shared_fingerprints": intersection,
                    "total_fingerprints": union
                })

        return self




def main():

    root_dir = "/home/scy/similarity_detect/code/test_total_files"
    dataloader = ReportDataLoader(root_dir)

    if not dataloader.groups:
        print("无可用的数据报告")

    # analyzer = report_TFIDF
    # analyzer = report_BERT
    # analyzer = report_LLM
    # analyzer = report_Winnowing
    # test = analyzer(dataloader)
    # winnowing_test = analyzer(dataloader, k=10, w=15, robust=True)
    # test.analyze()

    # # df = pd.DataFrame(test.results)
    # df = pd.DataFrame(winnowing_test.results)
    # sorted_df = df.sort_values(by='similarity', ascending=False)
    # filtered_df = sorted_df.query('similarity>= 0.5')
    # # results = test.get_similar_pairs()
    # results = winnowing_test.get_similar_pairs()
    # results.sort(key=lambda x:x['similarity'], reverse=True)

    # # print(results[0])
    # # print(filtered_df)
    # for i, pair  in enumerate(results):
    #     group1 = pair['group1']
    #     group2 = pair['group2']
    #     similarity = pair['similarity']
    #     print(f"{i+1} 小组：{group1} 和 {group2} 相似度：{similarity}")
    # analyzers = {
        # "TF-IDF": report_TFIDF(dataloader),
        # "BERT": report_BERT(dataloader),
        # "Winnowing (基础)": report_Winnowing(dataloader, k=10, w=15, robust=False),
        # "Winnowing (鲁棒)": report_Winnowing(dataloader, k=10, w=15, robust=True)
    # }

    winnowing_modes = {
        "basic": "基础模型"
        # "semantic": "语义"
        # "tfidf": "TFIDF加权"
        # "contextual": "qwen模型"
    }

    analyzers = {}
    for mode, name in winnowing_modes.items():
        print(f"初始化 {name} 分析器...")
        analyzers[name] = report_Winnowing(
            dataloader, 
            k=10, 
            w=15, 
            robust=True,
            mode=mode
        )

    for name, analyzer in analyzers.items():
        print(f"\n{'='*50}")
        print(f"开始 {name} 相似度分析...")
        analyzer.analyze()
        print(f"完成 {name} 分析，共计算 {len(analyzer.results)} 组对比")
        
        # 获取并展示最相似的前10对
        top_pairs = analyzer.get_top_pairs(10)
        print(f"\n{name} 方法 - 最相似的10对小组:")
        for i, pair in enumerate(top_pairs):
            sim_percent = pair['similarity'] * 100
            if 'shared_fingerprints' in pair:
                extra_info = f", 共享指纹: {pair['shared_fingerprints']}/{pair['total_fingerprints']}"
            else:
                extra_info = ""
                
            print(f"{i+1}. {pair['group1']} vs {pair['group2']}: "
                  f"相似度 {sim_percent:.2f}%{extra_info}")
    

if __name__ == "__main__":
    main()