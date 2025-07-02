from utils.utils import *



# 组信息
def extract_info(folder_name):

    student_ids = re.findall(r'\d{10}', folder_name)
    if not student_ids:
        return None, None 
    
    names = re.findall(r'[\u4e00-\u9fa5]{2,4}', folder_name)
    
    students = []
    

    if len(student_ids) == len(names):

        students = list(zip(student_ids, names))
    else:

        for sid in student_ids:

            sid_pos = folder_name.find(sid)
            
            closest_name = None
            min_distance = float('inf')
            
            for name in names:
                name_pos = folder_name.find(name)

                distance = abs(name_pos - sid_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_name = name
            
            if closest_name:
                students.append((sid, closest_name))
                names.remove(closest_name)  
    
    cleaned_info = re.sub(
        r'((?!\d{10}|[\u4e00-\u9fa5]{2,4}|和|与|及).)+', 
        ' ', 
        folder_name
    )
    
    return students, cleaned_info.strip()



def find_keil_project(root_dir):
    for root, dirs, files in os.walk(root_dir):
        # dirs[:] = [d for d in dirs if not d.startswith('EXTRACTED_')]
        for file in files:
            if file.lower().endswith(KEIL_PROJECT_EXTS):
                return os.path.join(root, file)
    return None


def safe_unzip(file_path, extract_dir):

    try: 
        os.makedirs(extract_dir, exist_ok=True)

        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
        elif file_path.endswith('.rar'):
            with rarfile.RarFile(file_path, 'r') as rar_ref:
                rar_ref.extractall(extract_dir)

            
        elif any(file_path.endswith(ext) for ext in ['.tar', '.tar.gz', '.tar.bz2']):
            with tarfile.open(file_path, 'r:*', encoding='gbk') as tar_ref:
                tar_ref.extractall(extract_dir)

        
        # print(f"解压文件: {os.path.basename(file_path)}")
        os.remove(file_path)
        return True

    except Exception as e:
        print(f"解压失败 {os.path.basename(file_path)}: {str(e)}")
        return False



def unzip_all_files(entry_path):

    # 确认所有压缩文件都已经被解压
    flag = False

    for root, dirs, files in os.walk(entry_path, topdown=True):
        dirs[:] = [d for d in dirs if not d.startswith('EXTRACTED_')]

        for file in files:
            if any(file.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                file_path = os.path.join(root, file)
                extract_dir = os.path.join(
                    root,
                    f"EXTRACTED_{os.path.splitext(file)[0]}"
                )
                
                if safe_unzip(file_path, extract_dir):
                    flag = True

    if flag:
        unzip_all_files(entry_path)

# 新增功能：提取PDF中的代码
def extract_code_from_pdf(pdf_path):
    """使用pdfplumber提取PDF中的C代码"""
    try:
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        
        # 清理代码：移除页码等无关内容
        cleaned_text = re.sub(r'第\s*\d+\s*[页頁]', '', full_text)
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
        
        return cleaned_text
    except Exception as e:
        print(f"提取PDF代码失败: {pdf_path}, error: {str(e)}")
        return ""

# 新增功能：检测并处理包含"代码"的PDF文件
def process_pdf_files(root_dir):
    """查找并处理包含'代码'的PDF文件"""
    pdf_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if (file.lower().endswith('.pdf') and "代码" in file) or (file.lower().endswith('.pdf') and "源码" in file):
                pdf_files.append(os.path.join(root, file))
    
    generated_c_files = []
    for pdf_file in pdf_files:
        # print(f"发现代码PDF文件: {os.path.basename(pdf_file)}")
        code_text = extract_code_from_pdf(pdf_file)
        
        if code_text.strip():
            # 生成对应的.c文件名
            c_filename = os.path.splitext(os.path.basename(pdf_file))[0] + ".c"
            c_filepath = os.path.join(os.path.dirname(pdf_file), c_filename)
            
            # 保存提取的代码
            with open(c_filepath, 'w', encoding='utf-8') as f:
                f.write(code_text)
            # print(f"  已提取代码保存至: {c_filename}")
            generated_c_files.append(c_filepath)
    
    return generated_c_files

def collect_Ccode(code_root):
    code_files = []
    for root, _, files in os.walk(code_root):
        for file in files:
            if file.lower().endswith(CODE_EXTENSIONS):
                code_files.append(os.path.join(root, file))
    return code_files