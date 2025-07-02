from utils.utils import *
from utils.config import *

class GroupMetaData:
    def __init__(self, group_dir):
        self.group_dir = group_dir  # 保存完整路径
        self.group_name = os.path.basename(group_dir)
        self.metadata_path = os.path.join(group_dir, 'metadata.csv')
        self.code_dir = os.path.join(group_dir, '代码')
        self.members = set()
        self.files = []
        self.md5s = set()
        self.md5_to_files = {}
        
        # 解析实验、年份、班级信息
        path_parts = group_dir.split(os.sep)
        pre_processed_index = -1

        config_parts = CONFIG["PRE_PROCESS_DIR"].split(os.sep)
        
        # 查找 "pre_processed" 在路径中的位置
        for i, part in enumerate(path_parts):
            # if part == "pre_processed":
            if part == config_parts[-1]:
                pre_processed_index = i
                break
        
        # 如果找到 "pre_processed"，则后面的目录结构应为: pre_processed/实验/年份/班级/小组名
        if pre_processed_index != -1 and len(path_parts) > pre_processed_index + 3:
            self.experiment = path_parts[pre_processed_index + 1]
            self.year = path_parts[pre_processed_index + 2]
            self.class_name = path_parts[pre_processed_index + 3]
        # 否则尝试从路径末尾解析
        elif len(path_parts) >= 4:
            self.experiment = path_parts[-4]
            self.year = path_parts[-3]
            self.class_name = path_parts[-2]
        else:
            self.experiment = "未知实验"
            self.year = "未知年份"
            self.class_name = "未知班级"
        
        # print(path_parts)
        # print(self.year)
        # print(self.class_name)
        # exit(0)
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
                    
                    if md5_val not in self.md5_to_files:
                        self.md5_to_files[md5_val] = []
                    self.md5_to_files[md5_val].append(os.path.basename(file_path))
                else:
                    print(f"警告: 文件不存在 {file_path}")

class DataLoader():
    def __init__(self, base_dirs=None, target_experiment=None):
        if base_dirs is None:
            base_dirs = [CONFIG['PRE_PROCESS_DIR']]
        self.base_dirs = base_dirs
        self.target_experiment = target_experiment
        self.groups = self.load_all_groups()

    def load_all_groups(self):
        groups = {}
        for base_dir in self.base_dirs:
            for root, dirs, files in os.walk(base_dir):
                for dir_name in dirs:
                    group_dir = os.path.join(root, dir_name)
                    metadata_path = os.path.join(group_dir, 'metadata.csv')
                    
                    if os.path.exists(metadata_path):
                        try:
                            group_meta = GroupMetaData(group_dir)
                            
                            # 如果指定了目标实验，只加载匹配的实验
                            if self.target_experiment:
                                # exp_key = f"{group_meta.experiment}_{group_meta.year}_{group_meta.class_name}"
                                exp_key = f"{group_meta.experiment}"
                                # exit(0)
                                if exp_key == self.target_experiment:
                                    groups[group_dir] = group_meta
                            else:
                                groups[group_dir] = group_meta
                        except Exception as e:
                            print(f"加载小组元数据失败: {group_dir} - {str(e)}")
        
        print(f"加载了{len(groups)}个小组的数据")
        return groups

class ReportGroup:
    def __init__(self, group_dir):
        self.group_dir = group_dir  # 保存完整路径
        self.group_name = os.path.basename(group_dir)
        self.report_dir = os.path.join(group_dir, "实验报告", "文字")
        self.report_path = self._find_report_text()
        self.content = ""
        self.length = 0
        self.hash = ""
        
        # 解析实验、年份、班级信息
        path_parts = group_dir.split(os.sep)
        pre_processed_index = -1

        config_parts = CONFIG["PRE_PROCESS_DIR"].split(os.sep)
        
        # 查找 "pre_processed" 在路径中的位置
        for i, part in enumerate(path_parts):
            # if part == "pre_processed":
            if part == config_parts[-1]:
                pre_processed_index = i
                break
        
        # 如果找到 "pre_processed"，则后面的目录结构应为: pre_processed/实验/年份/班级/小组名
        if pre_processed_index != -1 and len(path_parts) > pre_processed_index + 3:
            self.experiment = path_parts[pre_processed_index + 1]
            self.year = path_parts[pre_processed_index + 2]
            self.class_name = path_parts[pre_processed_index + 3]
        # 否则尝试从路径末尾解析
        elif len(path_parts) >= 4:
            self.experiment = path_parts[-4]
            self.year = path_parts[-3]
            self.class_name = path_parts[-2]
        else:
            self.experiment = "未知实验"
            self.year = "未知年份"
            self.class_name = "未知班级"
        
        # 添加实验标识符
        # if self.year == "未知年份":
        #     print(path_parts)
        #     print(self.year)
        #     print(self.class_name)
        #     exit(0)
        self.exp_key = f"{self.experiment}_{self.year}_{self.class_name}"
        
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
    def __init__(self, base_dirs=None, target_experiment=None):
        if base_dirs is None:
            base_dirs = [CONFIG["PRE_PROCESS_DIR"]]
        self.root_dirs = base_dirs
        self.target_experiment = target_experiment  # 添加目标实验参数
        self.groups = self._load_groups()

    def _load_groups(self):
        groups = {}
        valid_groups = 0

        for base_dir in self.root_dirs:
            # 递归遍历所有子目录
            for root, dirs, files in os.walk(base_dir):
                for dir_name in dirs:
                    group_dir = os.path.join(root, dir_name)
                    report_dir = os.path.join(group_dir, "实验报告", "文字")
                    
                    # 检查是否存在报告目录
                    if os.path.exists(report_dir):
                        try:
                            group_report = ReportGroup(group_dir)
                            
                            # 如果指定了目标实验，只加载匹配的实验
                            if self.target_experiment:
                                if group_report.experiment == self.target_experiment and group_report.content:
                                    groups[group_dir] = group_report
                                    valid_groups += 1
                            else:
                                if group_report.content:
                                    groups[group_dir] = group_report
                                    valid_groups += 1
                        except Exception as e:
                            print(f"加载报告数据失败: {group_dir} - {str(e)}")
        
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