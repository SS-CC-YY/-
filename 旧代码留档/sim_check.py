import os
import re
import torch
import csv
import torch
import pandas as pd
import numpy as np
import hashlib
import chardet
import tempfile

from pycparser import c_parser, c_ast, c_generator, parse_file
from pathlib import Path
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoModel, AutoTokenizer


CONFIG = {
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # "CODE_DIR": 'processed_code_data',
    # "CODE_DIR": '2024processed_code_data',
    # "CODE_DIR": 'testpdf_processed_code_data',
    # "CODE_DIR": '2024testpdf_processed_code_data',
    # "CODE_DIR": 'total_files',
    "CODE_DIR": "test_total_files",
    "THRESHOLD": 0.8,
    "MAX_LENGTH": 4096,
    "MODEL_PATH": "/data_new/scy/Qwen2.5-Coder-1.5B-Instruct",
    # "MODEL_PATH": "/data_new/scy/Qwen2.5-Coder-7B",
    # "MODEL_PATH": "/data_new/scy/Qwen2.5-Coder-3B-Instruct",
    "USE_GPU": False,
    "EXTENSIONS": ('.c', '.h'),
    # "MODEL_PATH": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "ast_weight":0.6,
    "sw_weight":0.4,
    "gcc_path":"gcc",
    "cpp_args": r"~Iutils/fake_libc_include",
    "tfidf_weight": 0.5,
    "llm_weight": 0.5
}

# ===============================================================
# ===============================================================

class GroupMetaData:
    def __init__(self, group_dir):
        self.group_name = os.path.basename(group_dir)
        self.metadata_path = os.path.join(group_dir, 'metadata.csv')
        self.code_dir = os.path.join(group_dir, '代码')
        self.members = set()
        self.files = []
        self.md5s = set()
        self.md5_to_files = {}
        self._load_metadata()


    def _load_metadata(self):
        if not os.path.exists(self.metadata_path):
            print(f"错误： {self.group_name} 没有元数据文件")
            return

        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                members = [tuple(m.split('_')) for m in row['成员'].split(' ')]
                self.members.update(members)
                
                file_path = os.path.join(self.code_dir, os.path.basename(row['目标文件路径']))
                if os.path.exists(file_path):
                    self.files.append(file_path)
                    md5_val = row['MD5']
                    # self.md5s.add(row['md5'])

                    if md5_val not in self.md5_to_files:
                        self.md5_to_files[md5_val] = []
                    self.md5_to_files[md5_val].append(os.path.basename(file_path))
                else:
                    print(f"警告: 文件不存在 {file_path}")
                

# ===============================================================
# ===============================================================


class DataLoader():
    # 加载所有小组的数据，元数据以及代码
    def __init__(self):
        self.groups = self.load_all_groups()

    def load_all_groups(self):
        groups = {}
        base_dir = CONFIG['CODE_DIR']

        for group_dir in os.listdir(base_dir):
            full_path = os.path.join(base_dir, group_dir)
            if os.path.isdir(full_path):
                group_meta = GroupMetaData(full_path)
                groups[group_dir] = group_meta
            
        print(f"加载了{len(groups)}个小组的数据")
        return groups


# ===============================================================
# ===============================================================
class CodePre_process:
    @staticmethod
    def pre_process(code):
        code = re.sub(r'//.*', '', code)  # 移除单行注释
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # 多行注释
        code = re.sub(r'\s+', ' ', code).strip()
        return code.lower()

    @classmethod
    def detect_file_encoding(cls, file_path):
        """检测文件编码"""
        try:
            # 先尝试读取前1024字节来检测编码
            with open(file_path, 'rb') as f:
                raw_data = f.read(1024)
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'gbk'  # 默认使用gbk
                confidence = result['confidence']
                
                # 如果置信度低于90%，尝试常见中文编码
                if confidence < 0.9:
                    try:
                        raw_data.decode('gbk')
                        return 'gbk'
                    except:
                        pass
                    
                    try:
                        raw_data.decode('gb2312')
                        return 'gb2312'
                    except:
                        pass
                    
                    try:
                        raw_data.decode('gb18030')
                        return 'gb18030'
                    except:
                        pass
                
                return encoding
        except Exception as e:
            print(f"编码检测失败 {file_path}: {str(e)}")
            return 'gbk' 


    @classmethod
    def read_group_code(cls, group_meta):
        """读取小组所有代码"""
        combined = []
        for file_path in group_meta.files:
            try:
                encoding = cls.detect_file_encoding(file_path)
                
                # 使用检测到的编码打开文件
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    code = f.read()
                    processed = cls.pre_process(code)
                    # combined.append(processed[:2000])
                    combined.append(processed)
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='gbk', errors='replace') as f:
                        code = f.read()
                        processed = cls.pre_process(code)
                        combined.append(processed[:2000])
                except:
                    print(f"最终读取失败 {file_path}")
            except Exception as e:
                print(f"读取文件错误 {file_path}: {str(e)}")
        return ' '.join(combined)[:CONFIG['MAX_LENGTH']]

# ===============================================================
# ===============================================================
# 方法1：
# 暴力使用tf-idf方法，遍历所有的代码文件，简单比较计算余弦相似度，感觉结果出来还行，但是就是比较的直观
# tf-idf: TF(t)*idf(t), tf=\frac{count(t)}{nums_of_words(doc)}, idf(t) = log(\frac{nums of docs}{1+DF(t)}), DF(t) 包含t的文档数量
class Traditional_Method:
    def __init__(self, dataloader):
        self.groups = dataloader.groups
        self.vectorizer = TfidfVectorizer(
            ngram_range = (3, 5),
            analyzer = 'char',
            min_df=2,
            max_features =10000
        )

    def _find_md5_duplicate(self):
        md5_map = {}
        for group_name, group_meta in self.groups.items():
            for md5_val, files in group_meta.md5_to_files.items():
                if md5_val not in md5_map:
                    md5_map[md5_val] = []
                
                # 存储组名和对应的文件名
                for file_name in files:
                    md5_map[md5_val].append((group_name, file_name))
        
        return {md5: groups_files for md5, groups_files in md5_map.items() if len(groups_files) > 1}


    def run(self):
        md5_duplicates = self._find_md5_duplicate()
        # print("文件MD5对比检查")
        # for md5, groups in md5_duplicates.items():
        #     print(f" - {md5[:6]}...: {', '.join(groups)}")

        group_names = list(self.groups.keys())
        contents = {}
        for group_name in group_names:
            contents[group_name] = CodePre_process.read_group_code(self.groups[group_name])

        corpus = [contents[group] for group in group_names]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        sim_matrix = cosine_similarity(tfidf_matrix)

        results = []
        for i in range(len(group_names)):
            for j in range(i+1, len(group_names)):
                results.append({
                    'group1': group_names[i],
                    'group2': group_names[j],
                    'similarity': sim_matrix[i][j]
                })

        return {
            'md5_duplicates': md5_duplicates,
            'similarity_matrix': sim_matrix,
            'group_order': group_names,
            'sim_results': sorted(results, key= lambda x: x['similarity'], reverse=True)
        }

# ===============================================================
# ===============================================================
# 方法二，介于服务器好像有人在跑实验，暂时没有卡能用，因此先暂停了大模型方法的开发
# 尝试开发使用AST语法树来处理，通过处理代码的逻辑而非直接比较对比来对更换函数名等进行检测

# log 6.3 主要c语言代码使用单片机的C51处理器来进行的处理，导致常规的gcc编译器等无法正确的编译相关的c语言代码
# 并且由于在服务器上安装相关的C51处理器存在一定的困难，因此使用抽象语义库来的方法暂时暂停

# class ASTProcessor:
#     # 解析AST语法树
#     def __init__(self):
#         self.parser = c_parser.CParser()
#         self.generator = c_generator.CGenerator()

#     def pre_process_code(self, code):
#         with tempfile.NamedTemporaryFile(suffix='.c', delete=False) as tmp_file:
#             tmp_file.write(code.encode('utf-8'))
#             tmp_file_path = tmp_file.name

#         with tempfile.NamedTemporaryFile(suffix='.c', delete=False) as tmp_out:
#             tmp_out_path = tmp_out.name


#         try:

#     def parse_to_ast(self, code):
#         try:
#             pre_process = self.pre_process_code(code)

# class ASTComparator:
#     def __init__(self):
#         self.ast_processor = ASTProcessor()
#         self.sw_algorithm = SmithWaterman()

#     def read_file(self, file_path):
#         chinese_encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'cp936', 'big5']
        
#         for encoding in chinese_encodings:
#             try:
#                 with open(file_path, 'r', encoding=encoding, errors='replace') as f:
#                     return f.read()
#             except UnicodeDecodeError:
#                 continue
#             except Exception as e:
#                 print(f"读取文件错误 {file_path}: {str(e)}")
        
#         print(f"无法读取文件 {file_path}，跳过")
#         return ""


#     def calculate_file_similarity(self, file1, file2):
#         content1 = self.read_file(file1)
#         content2 = self.read_file(file2)

#         if not content1 or not content2:
#             return 0.0
        
#         ast1 = self.ast_processor


# class AST_method:
#     def __init__(self, dataloader):
#         self.groups = dataloader.groups
#         self.comparator = ASTComparator()

#     def find_best_match(self, files1, files2):
#         best_sim = 0.0
#         best_pair = (None, None)

#         for file1 in files1:
#             for file2 in files2:
#                 if not file1.endswith(CONFIG['EXTENSIONS']) or not files2.endswith(CONFIG['EXTENSIONS']):
#                     continue
                
#                 similarity = self.comparator.calculate_file_similarity(file1, file2)

#     def calculate_group_sim(self, group1, group2):
#         files1 = self.groups[group1].files
#         files2 = self.groups[group2].files

#         if not files1 or not files2:
#             return 0.0

        
#         best_pair, bset_sim = self.find_best_match(files1, files2)


#     def run(self):

#         group_names = list(self.groups.keys())
#         num_group = len(group_names)
#         sim_matrix = np.zeros(num_group, num_group)

#         with tqdm(total=num_group*(num_group-1)//2, desc="AST计算相似度") as pbar:
#             for i in range(num_group):
#                 for j in range(i+1, num_group):
#                     sim = self.
# ===============================================================
# ===============================================================
# 方法三：尝试使用语言模型来进行查重检验 （Qwen-Coder2.5、）

# class LLM_method:
#     def __init__(self, dataloader):
#         self.groups = dataloader.groups
#         self.device = torch.device("cuda" if CONFIG['USE_GPU'] else "cpu")
#         self._init_model()

#     def _init_model(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_PATH'], trust_remote_code=True)
#         self.model = AutoModel.from_pretrained(CONFIG['MODEL_PATH'], trust_remote_code=True).to(self.device)

#     def get_embedding(self, content):
#         inputs = self.tokenizer(
#             content,
#             return_tensors="pt",
#             truncation=True,
#             max_length=CONFIG['MAX_LENGTH'],
#             padding=True
#         ).to(self.device)

#         with torch.no_grad():
#             output = self.model(**inputs)
#         return output.last_hidden_state.mean(dim=1).cpu().numpy()


#     def run(self):
#         group_names = list(self.groups.keys())
#         embeddings = {}

#         for group_name in tqdm(group_names, desc="处理小组"):
#             content = CodePre_process.read_group_code(self.groups[group_name])
#             embeddings[group_name] = self.get_embedding(content)

#         sim_matrix = np.zeros((len(group_names), len(group_names)))
#         for i in range(len(group_names)):
#             for j in range(i+1, len(group_names)):
#                 sim = cosine_similarity(
#                     embeddings[group_names[i]],
#                     embeddings[group_names[j]]
#                 )[0][0]
#                 sim_matrix[i][j] = sim
        
#         return {
#             'similarity_matrix': sim_matrix,
#             'group_order': group_names,
#             'embeddings': embeddings
#         }

# ===============================================================
# ===============================================================

# class Hybrid_method:
#     def __init__(self, dataloader, tra_result, llm_result):
#         self.groups = dataloader.groups
#         self.tra_result = tra_result
#         self.llm_result = llm_result

#     def run(self):
#         tra_groups = self.tra_result['group_order']
#         llm_groups = self.llm_result['group_order']

#         if tra_groups != llm_groups:
#             raise ValueError("组不一致")

#         group_names = tra_groups
#         num_group = len(group_names)

#         tra_matrix = self.tra_result['similarity_matrix']
#         llm_matrix = self.llm_result['similarity_matrix']

#         hybrid_matrix = np.zeros((num_group, num_group))
#         hybrid_results = []

#         for i in range(num_group):
#             for j in range(i+1, num_group):
#                 tra_sim = tra_matrix[i][j]
#                 llm_sim = llm_matrix[i][j]

#                 hybrid_sim = (
#                                 CONFIG['tfidf_weight'] * tra_sim + 
#                                 CONFIG['llm_weight'] * llm_sim
#                 )

#                 hybrid_matrix[i][j] = hybrid_sim
#                 hybrid_matrix[j][i] = hybrid_sim

#                 hybrid_results.append({
#                     'group1': group_names[i],
#                     'group2': group_names[j],
#                     'similarity': hybrid_sim,
#                     'tfidf_sim': tra_sim,
#                     'llm_sim': llm_sim
#                 })

#         hybrid_results.sort(key=lambda x:x['similarity'], reverse=True)

#         return {
#             'similarity_matrix': hybrid_matrix,
#             'group_order': group_names,
#             'hybrid_results': hybrid_results
#             # 'md5_duplicates': self.tra_result.get('md5_duplicates', {})
#         }


class SmithWaterman:
    """Smith-Waterman算法实现代码相似性检测"""
    def __init__(self, match_score=2, mismatch_penalty=-1, gap_penalty=-1):
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty
    
    def align(self, seq1, seq2):
        """执行序列比对并返回最高得分和位置"""
        m, n = len(seq1), len(seq2)
        # 初始化得分矩阵
        score_matrix = [[0] * (n+1) for _ in range(m+1)]
        max_score = 0
        max_i, max_j = 0, 0
        
        # 填充得分矩阵
        for i in range(1, m+1):
            for j in range(1, n+1):
                if seq1[i-1] == seq2[j-1]:
                    match = score_matrix[i-1][j-1] + self.match_score
                else:
                    match = score_matrix[i-1][j-1] + self.mismatch_penalty
                
                delete = score_matrix[i-1][j] + self.gap_penalty
                insert = score_matrix[i][j-1] + self.gap_penalty
                
                score_matrix[i][j] = max(0, match, delete, insert)
                
                if score_matrix[i][j] > max_score:
                    max_score = score_matrix[i][j]
                    max_i, max_j = i, j
        
        return max_score, max_i, max_j
    
    def traceback(self, seq1, seq2, score_matrix, i, j):
        """回溯找出最佳匹配序列"""
        alignment1, alignment2 = [], []
        
        while i > 0 and j > 0 and score_matrix[i][j] > 0:
            current_score = score_matrix[i][j]
            diagonal = score_matrix[i-1][j-1]
            up = score_matrix[i][j-1]
            left = score_matrix[i-1][j]
            
            if current_score == diagonal + (self.match_score if seq1[i-1] == seq2[j-1] else self.mismatch_penalty):
                alignment1.append(seq1[i-1])
                alignment2.append(seq2[j-1])
                i -= 1
                j -= 1
            elif current_score == left + self.gap_penalty:
                alignment1.append(seq1[i-1])
                alignment2.append('-')
                i -= 1
            elif current_score == up + self.gap_penalty:
                alignment1.append('-')
                alignment2.append(seq2[j-1])
                j -= 1
        
        return ''.join(reversed(alignment1)), ''.join(reversed(alignment2))
    
    def similarity(self, seq1, seq2):
        """计算两个序列的相似度得分"""
        if not seq1 or not seq2:
            return 0.0
        
        # 计算最长公共子序列得分
        max_score, i, j = self.align(seq1, seq2)
        
        # 归一化得分
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        return max_score / (min_len * self.match_score)
    
    def detailed_compare(self, seq1, seq2):
        """返回详细的比对结果"""
        m, n = len(seq1), len(seq2)
        score_matrix = [[0] * (n+1) for _ in range(m+1)]
        max_score = 0
        max_i, max_j = 0, 0
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                if seq1[i-1] == seq2[j-1]:
                    match = score_matrix[i-1][j-1] + self.match_score
                else:
                    match = score_matrix[i-1][j-1] + self.mismatch_penalty
                
                delete = score_matrix[i-1][j] + self.gap_penalty
                insert = score_matrix[i][j-1] + self.gap_penalty
                
                score_matrix[i][j] = max(0, match, delete, insert)
                
                if score_matrix[i][j] > max_score:
                    max_score = score_matrix[i][j]
                    max_i, max_j = i, j
        
        aln1, aln2 = self.traceback(seq1, seq2, score_matrix, max_i, max_j)
        return {
            'score': max_score,
            'alignment1': aln1,
            'alignment2': aln2,
            'similarity': max_score / (min(len(seq1), len(seq2)) * self.match_score) if min(len(seq1), len(seq2)) > 0 else 0
        }


class SmithWatermanMethod:
    """使用Smith-Waterman算法计算代码相似度"""
    def __init__(self, dataloader, max_length):
        self.groups = dataloader.groups
        self.sw = SmithWaterman()
        self.max_length = max_length  # 限制比对的代码长度
    
    def tokenize_code(self, code):
        """将代码转换为token序列，提高比对效率"""
        # 简单的token化：保留关键字、标识符和数字
        tokens = re.findall(r'[a-zA-Z_]\w*|\d+|\S', code)
        return tokens
    
    def run(self):
        group_names = list(self.groups.keys())
        num_groups = len(group_names)
        group_codes = {}
        
        # 预处理每个小组的代码
        for group_name in tqdm(group_names, desc="预处理代码"):
            content = CodePre_process.read_group_code(self.groups[group_name])
            tokens = self.tokenize_code(content)[:self.max_length]
            group_codes[group_name] = tokens
        
        # 初始化相似度矩阵
        sim_matrix = np.zeros((num_groups, num_groups))
        results = []
        
        # 两两计算相似度
        with tqdm(total=num_groups*(num_groups-1)//2, desc="Smith-Waterman计算相似度") as pbar:
            for i in range(num_groups):
                for j in range(i+1, num_groups):
                    sim = self.sw.similarity(group_codes[group_names[i]], group_codes[group_names[j]])
                    sim_matrix[i][j] = sim
                    sim_matrix[j][i] = sim
                    results.append({
                        'group1': group_names[i],
                        'group2': group_names[j],
                        'similarity': sim
                    })
                    pbar.update(1)
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return {
            'similarity_matrix': sim_matrix,
            'group_order': group_names,
            'sim_results': results,
            'group_codes': group_codes
        }
    
    def find_similar_fragments(self, group1, group2, group_codes, min_length=10):
        """找出两个小组代码中最相似的片段"""
        code1 = group_codes[group1]
        code2 = group_codes[group2]
        
        # 执行详细比对
        result = self.sw.detailed_compare(code1, code2)
        
        # 提取相似片段
        alignment1 = result['alignment1']
        alignment2 = result['alignment2']
        
        # 找出连续匹配的片段
        similar_fragments = []
        current_fragment1 = []
        current_fragment2 = []
        in_match = False
        
        for i in range(len(alignment1)):
            if alignment1[i] == alignment2[i] and alignment1[i] != '-':
                if not in_match:
                    in_match = True
                current_fragment1.append(alignment1[i])
                current_fragment2.append(alignment2[i])
            else:
                if in_match:
                    if len(current_fragment1) >= min_length:
                        fragment_str = ''.join(current_fragment1)
                        similar_fragments.append({
                            'fragment': fragment_str,
                            'length': len(current_fragment1),
                            'score': len(current_fragment1) * self.sw.match_score
                        })
                    current_fragment1 = []
                    current_fragment2 = []
                    in_match = False
        
        # 处理最后一个片段
        if in_match and len(current_fragment1) >= min_length:
            fragment_str = ''.join(current_fragment1)
            similar_fragments.append({
                'fragment': fragment_str,
                'length': len(current_fragment1),
                'score': len(current_fragment1) * self.sw.match_score
            })
        
        # 按长度排序
        similar_fragments.sort(key=lambda x: x['length'], reverse=True)
        
        return {
            'total_score': result['score'],
            'similarity': result['similarity'],
            'fragments': similar_fragments[:5]  # 返回前5个最长片段
        }





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


class WinnowingMethod:
    """使用Winnowing算法计算代码相似度"""
    def __init__(self, dataloader, k=5, w=10, robust=True):
        self.groups = dataloader.groups
        self.winnowing = Winnowing(k=k, w=w, robust=robust)
        
    def run(self):
        group_names = list(self.groups.keys())
        num_groups = len(group_names)
        fingerprints = {}
        
        # 为每个小组生成指纹
        for group_name in tqdm(group_names, desc="生成Winnowing指纹"):
            content = CodePre_process.read_group_code(self.groups[group_name])
            fingerprints[group_name] = self.winnowing.figerprint(content)
        
        # 计算相似度矩阵
        sim_matrix = np.zeros((num_groups, num_groups))
        results = []
        
        for i in range(num_groups):
            for j in range(i+1, num_groups):
                set1 = fingerprints[group_names[i]]
                set2 = fingerprints[group_names[j]]
                
                if not set1 or not set2:
                    similarity = 0.0
                else:
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    similarity = intersection / union if union > 0 else 0.0
                
                sim_matrix[i][j] = similarity
                sim_matrix[j][i] = similarity
                
                results.append({
                    'group1': group_names[i],
                    'group2': group_names[j],
                    'similarity': similarity
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'similarity_matrix': sim_matrix,
            'group_order': group_names,
            'sim_results': results,
            'fingerprints': fingerprints
        }


# ===============================================================
# ===============================================================
def print_results(result, name):
    if 'md5_duplicates' in result:
        md5_duplicates = result['md5_duplicates']
        print(f"\n文件MD5对比检, 共出现({len(md5_duplicates)})个完全相同文件")
        for md5, groups in md5_duplicates.items():
            group_files_map = {}
            for group_name, file_name in groups:
                if group_name not in group_files_map:
                    group_files_map[group_name] = []
                group_files_map[group_name].append(file_name)
            
            print(f"\nMD5: {md5[:6]}...{md5[-6:]}")
            for group_name, files in group_files_map.items():
                print(f"  小组: {group_name}")
                for file_name in files:
                    print(f"    - {file_name}")
            # print(f" - {md5[:6]}...: {', '.join(groups)}")

    print(f"\n{name}方法，代码相似度:")
    if 'sim_results' in result:
        results = result['sim_results']
    elif 'hybrid_results' in result:
        results = result['hybrid_results']
    elif 'similarity_matrix' in result:
        groups = result['group_order']
        matrix = result['similarity_matrix']
        results = []
        n = len(groups)
        for i in range(n):
            for j in range(i + 1, n):
                results.append({
                    'group1': groups[i],
                    'group2': groups[j],
                    'similarity': matrix[i][j]
                })
        results.sort(key=lambda x: x['similarity'], reverse=True)

    # print("\nTop 10 最相似小组对:")
    for i, item in enumerate(results[:20]):
    # for i, item in enumerate(results):
        extra_info = ""
        if 'tfidf_sim' in item and 'llm_sim' in item:
            extra_info = f" (TF-IDF: {item['tfidf_sim']:.4f}, LLM: {item['llm_sim']:.4f})"
        if item['similarity'] > CONFIG['THRESHOLD']:
            print(f"{i+1}. {item['group1']} vs {item['group2']}:\n {item['similarity']:.4f}{extra_info}")
    
    # 检查高相似度
    if results:
        top_sim = results[0]['similarity']
        if top_sim > CONFIG['THRESHOLD']:
            print(f"\n警告: 检测到高相似度 ({top_sim:.2f})，可能存在代码抄袭！")
    
    # 打印可疑组对（超过阈值的）
    # suspicious_pairs = [p for p in results if p['similarity'] > CONFIG['THRESHOLD']]
    # if suspicious_pairs:
    #     print(f"\n发现 {len(suspicious_pairs)} 对可疑抄袭组对 (相似度 > {CONFIG['THRESHOLD']}):")
    #     for i, pair in enumerate(suspicious_pairs):
    #         print(f"{i+1}. {pair['group1']} vs {pair['group2']}: {pair['similarity']:.4f}")

# ===============================================================
# ===============================================================



if __name__ == "__main__":

    print(f"正在加载数据")
    dataloader = DataLoader()

    traditional = Traditional_Method(dataloader)
    tra_result = traditional.run()
    print_results(tra_result, "tf-idf")

    # SW_Ast = AST_method(dataloader)
    # ast_result = SW_Ast.run()    

    # llm = LLM_method(dataloader)
    # llm_result = llm.run()
    # print_results(llm_result, "llm")

    # # 加权平均
    # hybrid = Hybrid_method(dataloader, tra_result, llm_result)
    # hybrid_result = hybrid.run()
    # print_results(hybrid_result, "hybrid")

    winnowing = WinnowingMethod(dataloader, k=5, w=10)
    winnowing_result = winnowing.run()
    print_results(winnowing_result, "Winnowing")


    sw_method = SmithWatermanMethod(dataloader, max_length=4000)
    sw_result = sw_method.run()
    print_results(sw_result, "Smith-Waterman")