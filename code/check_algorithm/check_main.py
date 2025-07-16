from utils.utils import *
from utils.config import *
from datetime import datetime

from data_loader.data_loader import *
from check_algorithm.tf_idf import *
from check_algorithm.smithWaterman import *
from check_algorithm.Winnowing import *

from results.result import *

def sim_check_main(base_dirs=None):
    # 加载所有数据
    print("\n正在加载所有作业数据...")
    dataloader = DataLoader(base_dirs)
    reportloader = ReportDataLoader(base_dirs)
    
    # 保存所有小组的元数据信息
    all_groups_info = {}
    
    # 收集代码小组信息
    for group_dir, group_meta in dataloader.groups.items():
        all_groups_info[group_meta.group_name] = {
            "experiment": group_meta.experiment,
            "year": group_meta.year,
            "class": group_meta.class_name,
            "path": group_meta.group_dir  # 添加完整路径信息
        }
    
    # 收集报告小组信息（补充可能缺失的小组）
    for group_dir, group_report in reportloader.groups.items():
        if group_report.group_name not in all_groups_info:
            all_groups_info[group_report.group_name] = {
                "experiment": group_report.experiment,
                "year": group_report.year,
                "class": group_report.class_name,
                "path": group_report.group_dir  # 添加完整路径信息
            }
    
    # 执行分析 - 对相同实验的所有小组进行对比（支持跨班）
    print("\n开始代码相似度分析（支持跨班检测）...")
    
    # 1. 按实验分组（忽略班级）
    experiments = {}
    for group_name, info in all_groups_info.items():
        exp_key = f"{info['experiment']}"
        if exp_key not in experiments:
            experiments[exp_key] = []
        experiments[exp_key].append(group_name)
    
    # 2. 对每个实验进行相似度分析（包含所有班级）
    code_winnowing_results = {}
    code_tfidf_results = {}
    report_winnowing_results = {}
    
    # 用于存储全局结果
    global_code_winnowing = []
    global_code_tfidf = []
    global_report_winnowing = []
    global_md5_duplicates = {}

    for exp_key, group_names in experiments.items():
        print(f"\n分析实验: {exp_key} (包含所有班级)")
        
        # 创建当前实验的数据加载器（忽略班级）
        exp_dataloader = DataLoader(base_dirs, target_experiment=exp_key)
        exp_reportloader = ReportDataLoader(base_dirs, target_experiment=exp_key)
        
        # TF-IDF代码分析
        code_tfidf = CODE_TFIDF(exp_dataloader)
        exp_tfidf = code_tfidf.run()
        code_tfidf_results[exp_key] = exp_tfidf
        
        # 添加到全局结果
        for match in exp_tfidf.get('sim_results', []):
            match['experiment'] = exp_key
            global_code_tfidf.append(match)
        
        # Winnowing代码分析
        code_winnowing = CODE_Winnowing(exp_dataloader, k=5, w=10)
        exp_winnowing = code_winnowing.run()
        code_winnowing_results[exp_key] = exp_winnowing
        
        # 添加到全局结果
        for match in exp_winnowing.get('sim_results', []):
            match['experiment'] = exp_key
            global_code_winnowing.append(match)
        
        # Winnowing报告分析
        report_analyzer = REPORT_Winnowing(exp_reportloader, k=10, w=15, robust=True)
        exp_report = report_analyzer.analyze()
        report_winnowing_results[exp_key] = exp_report
        
        # 添加到全局结果
        for match in exp_report.results:
            match['experiment'] = exp_key
            global_report_winnowing.append(match)
    
    # 提取所有年份用于筛选
    all_years = sorted(set(info['year'] for info in all_groups_info.values()))
    
    # 生成报告
    results = {
        "global_code_winnowing": sorted(global_code_winnowing, key=lambda x: x['similarity'], reverse=True),
        "global_code_tfidf": sorted(global_code_tfidf, key=lambda x: x['similarity'], reverse=True),
        "global_report_winnowing": sorted(global_report_winnowing, key=lambda x: x['similarity'], reverse=True),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "all_groups_info": all_groups_info,
        "experiments": experiments,
        "classes": sorted(set(info['class'] for info in all_groups_info.values())),
        "years": all_years  # 添加年份信息
    }
    
    generator = ReportGenerator()
    report_path = generator.generate_report(results)
    
    print(f"报告已生成: {report_path}")
    print(f"可通过以下方式访问:")
    print(f"1. 使用Python HTTP服务器: python -m http.server 8080 --directory results")
    print(f"2. 浏览器访问: http://10.12.44.151:8080/similarity_report.html")

    return report_path
