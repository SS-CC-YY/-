from utils.extract_code import *
from utils.extract_report import *
from utils.utils import *

# 处理单个提交文件
def process_single_submission(entry_path, output_base):

    # 首先解压全部文件
    unzip_all_files(entry_path)

    # 从文件名中获取组员名字和学号

    students_info, _ = extract_info(entry_path)
    if not students_info:
        print(f"\n警告，缺少有效姓名或者学号，请在unziped_submission文件夹中手动标注姓名和学号：{os.path.basename(entry_path)}")
        return False
    
    # 创建小组文件夹，存储提取的代码和报告
    member_list = '_'.join([f"{name}_{sid}" for sid, name in students_info])
    group_output_dir = os.path.join(output_base, member_list)
    # 如果同小组提交多次则保留最新检测到的
    if os.path.exists(group_output_dir):
        print(f"\n警告：检测到重复小组 '{member_list}'")
        # 安全删除旧文件夹
        try:
            shutil.rmtree(group_output_dir)
        except Exception as e:
            print(f"删除重复小组：{member_list}时失败: {str(e)}")
            return False
        # return False
    
    
    
    os.makedirs(group_output_dir, exist_ok=True)

    # 考虑到为keil项目，先寻找是否整个提交keil项目
    # 如果是则从keil项目中提取所有代码，如果不是则遍历提取所有'.c','.h'结尾代码文件
    keil_project = find_keil_project(entry_path)    
    code_root = os.path.dirname(keil_project) if keil_project else entry_path

    code_files = collect_Ccode(code_root)

    if not code_files:
        generated_c_files = process_pdf_files(entry_path)
        for c_file in generated_c_files:
            if c_file not in code_files:
                code_files.append(c_file)


    if code_files:
        output_code_dir = os.path.join(group_output_dir, '代码')
        os.makedirs(output_code_dir, exist_ok=True)

        processed_files = []
        for src_file in code_files:
            try:
                # 创建目标路径
                base_name = os.path.basename(src_file)
                dest_name = f"{base_name}"
                dest_path = os.path.join(output_code_dir, dest_name)
                
                # 处理重名文件
                counter = 1
                while os.path.exists(dest_path):
                    # 修改文件名格式，添加序号
                    name_part, ext_part = os.path.splitext(base_name)
                    dest_name = f"{name_part}_{counter}{ext_part}"
                    dest_path = os.path.join(output_code_dir, dest_name)
                    counter += 1
                
                # 复制文件并保留元数据
                shutil.copy2(src_file, dest_path)
                
                # 获取处理后的文件信息
                file_size = os.path.getsize(dest_path)  # 使用目标文件路径
                file_md5 = hashlib.md5(open(dest_path, 'rb').read()).hexdigest()
                
                processed_files.append({
                    "source": src_file,
                    "destination": dest_path,
                    "size": file_size,
                    "md5": file_md5
                })
                
            except Exception as e:
                print(f"文件整合失败: {src_file} -> {str(e)}")
                # continue

        # 记录元数据
        metadata_path = os.path.join(group_output_dir, "metadata.csv")
        
        with open(metadata_path, 'a', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["成员", "源文件路径", "目标文件路径", "文件大小", "MD5"])
            writer.writeheader()
            
            for record in processed_files:
                writer.writerow({
                    "成员": member_list,
                    "源文件路径": record["source"],
                    "目标文件路径": record["destination"],
                    "文件大小": record["size"],
                    "MD5": record["md5"]
                })

    else: 
        print(f"\n警告，该组未检测到有效代码文件或使用text文档等其他文件格式:{member_list}")
        # return False
    

    # print(f"开始提取实验报告：{os.path.basename(entry_path)}")
    try:
        report_success = extract_file(entry_path, group_output_dir)
    except Exception as e:
        print(f"报告提取失败：{str(e)}")
        return False
    return report_success




def extract_code_report(source_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # for entry in os.listdir(source_dir):
    #     entry_dir = os.path.join(source_dir, entry)

    #     if not os.path.isdir(entry_dir):
    #         print(f"跳过非目录项: {entry}")
    #         continue
    #     process_single_submission(entry_dir, output_dir)
    problem_dir = os.path.join(source_dir, "存在问题的小组")
    os.makedirs(problem_dir, exist_ok=True)

    for entry in os.listdir(source_dir):
        entry_dir = os.path.join(source_dir, entry)
        
        # 跳过非目录项和问题文件夹本身
        if not os.path.isdir(entry_dir) or entry == "存在问题的小组":
            continue
            
        success = process_single_submission(entry_dir, output_dir)
        
        if success:
            # 成功处理后删除原始文件夹
            try:
                shutil.rmtree(entry_dir)
                # print(f"已删除成功处理的小组文件夹: {entry}")
            except Exception as e:
                print(f"删除小组文件夹失败: {entry} - {str(e)}")
        else:
            # 处理失败，移动到问题文件夹
            try:
                dest_path = os.path.join(problem_dir, entry)
                # 如果目标已存在，添加后缀
                counter = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(problem_dir, f"{entry}_{counter}")
                    counter += 1
                
                shutil.move(entry_dir, dest_path)
                print(f"已移动问题小组到: {os.path.join('存在问题的小组', os.path.basename(dest_path))}")
            except Exception as e:
                print(f"移动问题小组失败: {entry} - {str(e)}")