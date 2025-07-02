from utils.utils import *
from check_algorithm.code_pre_pro import *

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


class CODE_SmithWaterman:
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