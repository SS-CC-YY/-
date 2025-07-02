from utils.utils import *

# 确认文件后缀
def get_base_extension(filename):
    for ext in SUPPORTED_EXTENSIONS:
        if filename.endswith(ext):
            base = filename[:-len(ext)]
            return base, ext
    return None, None

# 
def extract_zip(file_path, output_dir):
    # 考虑到中文编码，因此尝试不同的编码来解压压缩文件
    encodings = ['gbk', 'utf-8', 'cp437', None]  # None为系统默认
    for encoding in encodings:
        try:
            kwargs = {'metadata_encoding': encoding} if encoding is not None else {}
            with zipfile.ZipFile(file_path, 'r', **kwargs) as zip_ref:
                zip_ref.extractall(output_dir)
            return True
        except UnicodeDecodeError:
            continue  # 使用下个编码方式
        except zipfile.BadZipFile as e:
            print(f"Bad ZIP file: {os.path.basename(file_path)} - {str(e)}")
            return False
    print(f"无法解压zip文件元数据: {os.path.basename(file_path)}")
    return False


def unzip_files(source_dir, target_dir):

    os.makedirs(target_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        
        if os.path.isfile(file_path):
            base, ext = get_base_extension(filename)
            if not ext:
                print(f"跳过不支持的文件: {filename}")
                continue

            output_dir = os.path.join(target_dir, base)
            os.makedirs(output_dir, exist_ok=True)

            try:
                if ext == '.zip':
                    extract_zip(file_path, output_dir)
                elif ext == '.rar':
                    with rarfile.RarFile(file_path, 'r') as rar_ref:
                        rar_ref.extractall(output_dir)
                else:  
                    try:
                        with tarfile.open(file_path, 'r:*', encoding='gbk') as tar_ref:
                            tar_ref.extractall(output_dir)
                    except UnicodeDecodeError:
                        with tarfile.open(file_path, 'r:*') as tar_ref: 
                            tar_ref.extractall(output_dir)
            
            except (zipfile.BadZipFile, tarfile.TarError, rarfile.Error) as e:
                print(f"失败: {filename} - {str(e)}")
            except Exception as e:
                print(f"发生未知错误: {str(e)} 处理文件: {filename}")
        
        elif os.path.isdir(file_path):
            # 在目标目录创建同名文件夹
            output_dir = os.path.join(target_dir, filename)
            os.makedirs(output_dir, exist_ok=True)
            
            # 检查文件夹中是否有压缩文件
            has_compressed_files = False
            for item in os.listdir(file_path):
                item_path = os.path.join(file_path, item)
                if os.path.isfile(item_path):
                    _, ext = get_base_extension(item)
                    if ext:
                        has_compressed_files = True
                        break
            
            # 如果有压缩文件，先处理它们
            if has_compressed_files:
                for item in os.listdir(file_path):
                    item_path = os.path.join(file_path, item)
                    if os.path.isfile(item_path):
                        base, ext = get_base_extension(item)
                        if ext:
                            item_output_dir = os.path.join(output_dir, base)
                            os.makedirs(item_output_dir, exist_ok=True)
                            
                            try:
                                if ext == '.zip':
                                    extract_zip(item_path, item_output_dir)
                                elif ext == '.rar':
                                    with rarfile.RarFile(item_path, 'r') as rar_ref:
                                        rar_ref.extractall(item_output_dir)
                                else:  
                                    try:
                                        with tarfile.open(item_path, 'r:*', encoding='gbk') as tar_ref:
                                            tar_ref.extractall(item_output_dir)
                                    except UnicodeDecodeError:
                                        with tarfile.open(item_path, 'r:*') as tar_ref: 
                                            tar_ref.extractall(item_output_dir)
                            
                            except (zipfile.BadZipFile, tarfile.TarError, rarfile.Error) as e:
                                print(f"失败: {filename}/{item} - {str(e)}")
                            except Exception as e:
                                print(f"发生未知错误: {str(e)} 处理文件: {filename}/{item}")
            
            # 整体复制文件夹内容到目标目录
            for item in os.listdir(file_path):
                item_path = os.path.join(file_path, item)
                dest_path = os.path.join(output_dir, item)
                
                # 如果是文件且不是压缩文件，或者已经处理过压缩文件，则直接复制
                if os.path.isfile(item_path):
                    _, ext = get_base_extension(item)
                    # 如果是压缩文件且已经处理过，则跳过复制
                    if ext and has_compressed_files:
                        continue
                    shutil.copy2(item_path, dest_path)
                # 如果是子文件夹，则递归复制
                elif os.path.isdir(item_path):
                    shutil.copytree(item_path, dest_path, dirs_exist_ok=True)