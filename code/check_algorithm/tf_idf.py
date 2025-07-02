from utils.utils import *
from check_algorithm.code_pre_pro import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class CODE_TFIDF:
    def __init__(self, dataloader, target_group=None):
        self.groups = dataloader.groups
        self.target_group = target_group  # 新增目标小组参数
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

        group_names = list(self.groups.keys())
        contents = {}
        for group_name in group_names:
            contents[group_name] = CodePre_process.read_group_code(self.groups[group_name])

        corpus = [contents[group] for group in group_names]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        sim_matrix = cosine_similarity(tfidf_matrix)

        results = []
        
        # 计算总进度
        total_pairs = len(group_names) * (len(group_names) - 1) // 2
        if self.target_group:
            total_pairs = len(group_names) - 1
        
        with tqdm(total=total_pairs, desc="TF-IDF计算相似度") as pbar:
            if self.target_group:
                # 目标小组模式：只计算目标小组与其他小组的相似度
                target_index = group_names.index(self.target_group)
                for j in range(len(group_names)):
                    if j == target_index:
                        continue
                    
                    results.append({
                        'group1': group_names[target_index],
                        'group2': group_names[j],
                        'similarity': sim_matrix[target_index][j]
                    })
                    pbar.update(1)
            else:
                # 全量模式：计算所有小组两两之间的相似度
                for i in range(len(group_names)):
                    for j in range(i+1, len(group_names)):
                        results.append({
                            'group1': group_names[i],
                            'group2': group_names[j],
                            'similarity': sim_matrix[i][j]
                        })
                        pbar.update(1)

        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'md5_duplicates': md5_duplicates,
            'similarity_matrix': sim_matrix,
            'group_order': group_names,
            'sim_results': results
        }