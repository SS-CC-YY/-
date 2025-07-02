from utils.unzip import *
from utils.extract_single_submission import *
from utils.utils import *
from utils.config import *
from check_algorithm.check_main import *


if __name__ == "__main__":

    all_preprocess_dirs = []  # 存储所有预处理目录

    while True:
        print("1. 选定实验年份与班级")
        print("2. 生成报告")
        print("3. 退出")
        choice = input().strip()
        
        if choice == "2":
            break
        elif choice == "3":
            sys.exit(0)

        # 获取用户输入的实验信息
        print("\n请输入实验信息（格式：年份 实验几 周几班），例如：2024 实验一 周一班")
        print("（直接回车结束输入）")
        experiment_info = input().strip()
        
        if not experiment_info:
            break
            
        # 解析实验信息
        parts = experiment_info.split()
        if len(parts) < 3:
            print("格式错误，请重新输入")
            continue
            
        year = parts[0]
        exp_id = parts[1]
        class_name = parts[2]
        
        # 创建对应的目录结构
        class_unzip_dir = os.path.join(CONFIG["UNZIP_FILE_DIR"], exp_id, year, class_name)
        class_preprocess_dir = os.path.join(CONFIG["PRE_PROCESS_DIR"], exp_id, year, class_name)
        os.makedirs(class_unzip_dir, exist_ok=True)
        os.makedirs(class_preprocess_dir, exist_ok=True)
        
        # 获取要处理的作业文件夹
        print("1. 提交班级作业")
        print("2. 处理有问题的小组")
        print("3. 退出")
        choice = input().strip()

        if choice =="3":
            break
        elif choice == "2":
            source_dirs = os.path.join(class_unzip_dir, "存在问题的小组").split()
        else:
            print("\n输入要处理的作业文件夹路径，多个文件夹请用空格隔开")
            print("(若为空则处理该班级解压目录中已有的文件，即预处理步骤)\n")
            print("注意：请确保文件夹内直接包含各组作业")
            source_dirs = input().split()
        
        # print(source_dirs)
        # exit(0)

        if source_dirs == []:
            print(f"\n开始预处理学生提交作业 - {year} {exp_id} {class_name}")
            extract_code_report(class_unzip_dir, class_preprocess_dir)
            print(f"预处理完成，文件保存在：{class_preprocess_dir}")
        elif source_dirs:
            for source_zip_dir in source_dirs:
                print(f"\n解压处理作业文件夹：{source_zip_dir} -> {class_unzip_dir}")
                if not os.path.isdir(source_zip_dir):
                    print(f"错误：无法找到给定作业文件夹路径：{source_zip_dir}")
                    continue
                unzip_files(source_zip_dir, class_unzip_dir)
            
            # 解压完原始提交压缩包后，对学生提交的压缩包文件做处理
            print(f"\n压缩文件解压完成，解压后的文件保存在：{class_unzip_dir}")
            print(f"开始预处理学生提交作业 - {year} {exp_id} {class_name}")
            extract_code_report(class_unzip_dir, class_preprocess_dir)
            print(f"\n预处理完成，文件保存在：{class_preprocess_dir}")
        
        # 添加到总目录列表
        all_preprocess_dirs.append(class_preprocess_dir)
        
        # 询问用户下一步操作
        print("\n请选择下一步操作：")
        print("1. 处理下一个班级")
        print("2. 开始查重检测")
        print("3. 退出")
        choice = input().strip()
        
        if choice == "2":
            break
        elif choice == "3":
            sys.exit(0)
    
    sim_check_main()  # 无参数调用