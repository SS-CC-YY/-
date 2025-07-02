from utils.utils import *
from check_algorithm.code_pre_pro import *
from data_loader.data_loader import *

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

class CODE_Winnowing:
    """使用Winnowing算法计算代码相似度"""
    def __init__(self, dataloader, k=5, w=10, robust=True, target_group=None):
        self.groups = dataloader.groups
        self.target_group = target_group  # 新增目标小组参数
        self.winnowing = Winnowing(k=k, w=w, robust=robust)
        
    def run(self):
        group_names = list(self.groups.keys())
        num_groups = len(group_names)
        fingerprints = {}
        
        # 为每个小组生成指纹
        for group_name in tqdm(group_names, desc="生成代码指纹"):
            content = CodePre_process.read_group_code(self.groups[group_name])
            fingerprints[group_name] = self.winnowing.figerprint(content)
        
        # 计算相似度矩阵
        sim_matrix = np.zeros((num_groups, num_groups))
        results = []
        
        # 计算总进度
        total_pairs = num_groups * (num_groups - 1) // 2
        if self.target_group:
            # 如果是目标小组模式，只计算目标小组与其他小组的相似度
            total_pairs = num_groups - 1
        
        with tqdm(total=total_pairs, desc="Winnowing计算代码相似度") as pbar:
            if self.target_group:
                # 目标小组模式：只计算目标小组与其他小组的相似度
                target_index = group_names.index(self.target_group)
                for j in range(num_groups):
                    if j == target_index:
                        continue
                    
                    set1 = fingerprints[group_names[target_index]]
                    set2 = fingerprints[group_names[j]]
                    
                    if not set1 or not set2:
                        similarity = 0.0
                    else:
                        intersection = len(set1 & set2)
                        union = len(set1 | set2)
                        similarity = intersection / union if union > 0 else 0.0
                    
                    sim_matrix[target_index][j] = similarity
                    sim_matrix[j][target_index] = similarity
                    
                    results.append({
                        'group1': group_names[target_index],
                        'group2': group_names[j],
                        'similarity': similarity
                    })
                    pbar.update(1)
            else:
                # 全量模式：计算所有小组两两之间的相似度
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
                        pbar.update(1)
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'similarity_matrix': sim_matrix,
            'group_order': group_names,
            'sim_results': results,
            'fingerprints': fingerprints
        }

class REPORT_Winnowing(ReportSimilarityAnalyzer):
    def __init__(self, dataloader, k=10, w=15, robust=True, target_group=None):
        super().__init__(dataloader)
        self.k = k
        self.w = w
        self.robust = robust
        self.target_group = target_group  # 新增目标小组参数
        self.winnowing = Winnowing(k=k, w=w, robust=robust)
        self.fingerprints = {}
        self.corpus = self._get_corpus()
        self._init_winnowing()

    def _init_winnowing(self):
        self.winnowing = Winnowing(k=self.k, w=self.w, robust=self.robust)

    def _get_corpus(self):
        return [group.preprocess(for_winnowing=True) for group in self.groups.values()]

    def analyze(self):
        for group_name, group in tqdm(self.groups.items(), desc="报告生成指纹"):
            text = group.preprocess(for_winnowing=True)
            self.fingerprints[group_name] = self.winnowing.figerprint(text)

        group_names = list(self.fingerprints.keys())
        n = len(group_names)

        self.results = []
        
        # 计算总进度
        total_pairs = n * (n - 1) // 2
        if self.target_group:
            total_pairs = n - 1
        
        with tqdm(total=total_pairs, desc="Winnowing计算报告相似度") as pbar:
            if self.target_group:
                # 目标小组模式：只计算目标小组与其他小组的相似度
                for j in range(n):
                    if group_names[j] == self.target_group:
                        continue
                    
                    fp1 = self.fingerprints[self.target_group]
                    fp2 = self.fingerprints[group_names[j]]

                    intersection = len(fp1 & fp2)
                    union = len(fp1 | fp2)

                    similarity = intersection / union if union >0 else 0.0

                    self.results.append({
                        "method": f"winnowing(k={self.k},w={self.w})",
                        "group1": self.target_group,
                        "group2": group_names[j],
                        "similarity": similarity,
                        "shared_fingerprints": intersection,
                        "total_fingerprints": union
                    })
                    pbar.update(1)
            else:
                # 全量模式：计算所有小组两两之间的相似度
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
                        pbar.update(1)

        # 按相似度排序
        self.results.sort(key=lambda x: x["similarity"], reverse=True)
        return self