# from jinja2 import Environment, FileSystemLoader

# from utils.utils import *

# class ReportGenerator:
#     def __init__(self, output_dir="results"):
#         self.output_dir = output_dir
#         os.makedirs(output_dir, exist_ok=True)
        
#         # 复制样式表到输出目录
#         try:
#             # 获取当前脚本所在目录
#             current_dir = os.path.dirname(os.path.abspath(__file__))
#             # 构建模板目录路径
#             templates_dir = os.path.join(current_dir, "..", "templates")
#             css_src = os.path.join(templates_dir, "style.css")
#             css_dest = os.path.join(output_dir, "style.css")
            
#             if os.path.exists(css_src):
#                 shutil.copy(css_src, css_dest)
#                 print(f"已复制样式表到: {css_dest}")
#             else:
#                 print(f"警告: 样式表文件不存在: {css_src}")
#         except Exception as e:
#             print(f"复制样式表失败: {str(e)}")
        
#         # 设置模板环境
#         self.env = Environment(loader=FileSystemLoader(templates_dir))
#         self.env.globals.update(enumerate=enumerate)
        
#     def generate_report(self, results, template_name="template.html", top_n=100):
#         # 处理实验结果
#         processed_results = {
#             "global_results": {
#                 "code_winnowing": self._process_global_results(results["global_code_winnowing"], top_n),
#                 "code_tfidf": self._process_global_results(results["global_code_tfidf"], top_n),
#                 "report_winnowing": self._process_report_results(results["global_report_winnowing"], top_n),
#             },
#             "timestamp": results.get("timestamp", ""),
#             "all_groups_info": results["all_groups_info"],
#             "experiments": sorted(results["experiments"].keys()),
#             "classes": results["classes"]
#         }
        
#         template = self.env.get_template(template_name)
#         html = template.render(report_data=processed_results)
        
#         report_path = os.path.join(self.output_dir, "similarity_report.html")
#         with open(report_path, 'w', encoding='utf-8') as f:
#             f.write(html)
            
#         return report_path
    
#     def _process_global_results(self, results, top_n):
#         processed = []
#         for match in results[:top_n]:
#             group1 = match.get('group1', '')
#             group2 = match.get('group2', '')
            
#             processed.append({
#                 "experiment": match.get('experiment', ''),
#                 "group1": group1,
#                 "group2": group2,
#                 "similarity": match.get('similarity', 0),
#                 "similarity_percent": round(match.get('similarity', 0) * 100, 1)
#             })
#         return processed
    
#     def _process_report_results(self, results, top_n):
#         processed = []
#         for match in results[:top_n]:
#             group1 = match.get('group1', '')
#             group2 = match.get('group2', '')
            
#             processed.append({
#                 "experiment": match.get('experiment', ''),
#                 "group1": group1,
#                 "group2": group2,
#                 "similarity": match.get('similarity', 0),
#                 "similarity_percent": round(match.get('similarity', 0) * 100, 1)
#             })
#         return processed

from jinja2 import Environment, FileSystemLoader

from utils.utils import *

class ReportGenerator:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 复制样式表到输出目录
        try:
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 构建模板目录路径
            templates_dir = os.path.join(current_dir, "..", "templates")
            css_src = os.path.join(templates_dir, "style.css")
            css_dest = os.path.join(output_dir, "style.css")
            
            if os.path.exists(css_src):
                shutil.copy(css_src, css_dest)
                print(f"已复制样式表到: {css_dest}")
            else:
                print(f"警告: 样式表文件不存在: {css_src}")
        except Exception as e:
            print(f"复制样式表失败: {str(e)}")
        
        # 设置模板环境
        self.env = Environment(loader=FileSystemLoader(templates_dir))
        self.env.globals.update(enumerate=enumerate)
        
    def generate_report(self, results, template_name="template.html"):
        # 处理实验结果
        processed_results = {
            "global_results": {
                "code_winnowing": self._process_global_results(results["global_code_winnowing"], results["all_groups_info"]),
                "code_tfidf": self._process_global_results(results["global_code_tfidf"], results["all_groups_info"]),
                "report_winnowing": self._process_report_results(results["global_report_winnowing"], results["all_groups_info"]),
            },
            "timestamp": results.get("timestamp", ""),
            "all_groups_info": results["all_groups_info"],
            "experiments": sorted(results["experiments"].keys()),
            "classes": results["classes"],
            "years": results["years"]  # 添加年份信息
        }
        
        

        template = self.env.get_template(template_name)
        html = template.render(report_data=processed_results)
        
        report_path = os.path.join(self.output_dir, "similarity_report.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
            
        return report_path
    
    def _process_global_results(self, results, all_groups_info):
        processed = []
        for match in results:
            group1 = match.get('group1', '')
            group2 = match.get('group2', '')
            # group1_info = all_groups_info.get(group1, {})
            # group2_info = all_groups_info.get(group2, {})

            group1_path = group1.split(os.sep)
            group1_name = group1_path[-1]
            group1_info = all_groups_info.get(group1_name, {})
            group2_path = group2.split(os.sep)
            group2_name = group2_path[-1]
            group2_info = all_groups_info.get(group2_name, {})
            # print(match)
            # print(group1)
            # print(group2)
            # print(group1_info)
            # print(test_group1_info)
            # print(group2_info)
            # input()
            
            result = {
                "experiment": match.get('experiment', ''),
                "group1": group1_name,
                "group2": group2_name,
                "similarity": match.get('similarity', 0),
                "similarity_percent": round(match.get('similarity', 0) * 100, 1),
                "year1": group1_info.get('year', '未知年份'),
                "year2": group2_info.get('year', '未知年份'),
                "class1": group1_info.get('class', '未知班级'),
                "class2": group2_info.get('class', '未知班级'),
                "path1": group1_info.get('path', ''),
                "path2": group2_info.get('path', '')
            }

            # print(result)
            # input()

            processed.append(result)
        return processed
    
    def _process_report_results(self, results, all_groups_info):
        processed = []
        for match in results:
            group1 = match.get('group1', '')
            group2 = match.get('group2', '')
            # group1_info = all_groups_info.get(group1, {})
            # group2_info = all_groups_info.get(group2, {})
            
            group1_path = group1.split(os.sep)
            group1_name = group1_path[-1]
            group1_info = all_groups_info.get(group1_name, {})
            group2_path = group2.split(os.sep)
            group2_name = group2_path[-1]
            group2_info = all_groups_info.get(group2_name, {})

            result = {
                "experiment": match.get('experiment', ''),
                "group1": group1_name,
                "group2": group2_name,
                "similarity": match.get('similarity', 0),
                "similarity_percent": round(match.get('similarity', 0) * 100, 1),
                "year1": group1_info.get('year', '未知年份'),
                "year2": group2_info.get('year', '未知年份'),
                "class1": group1_info.get('class', '未知班级'),
                "class2": group2_info.get('class', '未知班级'),
                "path1": group1_info.get('path', ''),
                "path2": group2_info.get('path', '')
            }
            processed.append(result)
        return processed