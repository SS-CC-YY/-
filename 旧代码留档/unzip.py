# import os
# import zipfile
# import tarfile
# os.environ['UNRAR_LIB_PATH'] = os.path.expanduser('~/.local/lib/libunrar.so')

# from unrar import rarfile
# import sys


# SUPPORTED_EXTENSIONS = {
#     '.tar.gz', '.tgz',
#     '.tar.bz2', '.tbz2',
#     '.tar',
#     '.zip',
#     '.rar'
# }

# def get_base_extension(filename):
#     for ext in SUPPORTED_EXTENSIONS:
#         if filename.endswith(ext):
#             base = filename[:-len(ext)]
#             return base, ext
#     return None, None

# def detect_file_encoding(file_path):
#         """检测文件编码"""
#         try:
#             # 先尝试读取前1024字节来检测编码
#             with open(file_path, 'rb') as f:
#                 raw_data = f.read(1024)
#                 result = chardet.detect(raw_data)
#                 encoding = result['encoding'] or 'gbk'  # 默认使用gbk
#                 confidence = result['confidence']
                
#                 # 如果置信度低于90%，尝试常见中文编码
#                 if confidence < 0.9:
#                     try:
#                         raw_data.decode('gbk')
#                         return 'gbk'
#                     except:
#                         pass
                    
#                     try:
#                         raw_data.decode('gb2312')
#                         return 'gb2312'
#                     except:
#                         pass
                    
#                     try:
#                         raw_data.decode('gb18030')
#                         return 'gb18030'
#                     except:
#                         pass
                
#                 return encoding
#         except Exception as e:
#             print(f"编码检测失败 {file_path}: {str(e)}")
#             return 'gbk' 


# def unzip_files(source_dir):

#     target_dir = ("../2024submission")
#     os.makedirs(target_dir, exist_ok=True)

#     for filename in os.listdir(source_dir):
#         file_path = os.path.join(source_dir, filename)

#         if not os.path.isfile(file_path):
#             continue
        
#         base, ext = get_base_extension(filename)
#         if not ext:
#             print(f"pass the unsupported file: {filename}")
#             continue

#         output_dir = os.path.join(target_dir, base)
#         os.makedirs(output_dir, exist_ok=True)

#         encode = detect_file_encoding(file_path)

#         try:
#             if ext == '.zip':
#                 with zipfile.ZipFile(file_path, 'r',metadata_encoding=encode) as zip_ref:
#                     zip_ref.extractall(output_dir)
            

#             elif ext == '.rar':
#                 with rarfile.RarFile(file_path, 'r') as rar_ref:
#                     rar_ref.extractall(output_dir)
            
#             else:
#                 with tarfile.open(file_path, 'r:*', encoding=encode) as tar_ref:
#                     tar_ref.extractall(output_dir)
            
#             print(f"Success: {filename} → {output_dir}")
        
#         except (zipfile.BadZipFile, tarfile.TarError, rarfile.RarCannotExec, rarfile.BadRarFile) as e:
#             print(f"Fail: {filename} → error: {str(e)}")
#         except Exception as e:
#             print(f"Unknown error: {str(e)} handling file: {filename}")





# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python unzip.py <path to the directory>")
#         sys.exit(1)
    
#     source_dir = sys.argv[1]
#     if not os.path.isdir(source_dir):
#         print(f"Error: Path {target_dir} not found")
#         sys.exit(1)


#     unzip_files(source_dir)


import os
import zipfile
import tarfile
os.environ['UNRAR_LIB_PATH'] = os.path.expanduser('~/.local/lib/libunrar.so')

from unrar import rarfile
import sys

SUPPORTED_EXTENSIONS = {
    '.tar.gz', '.tgz',
    '.tar.bz2', '.tbz2',
    '.tar',
    '.zip',
    '.rar'
}

def get_base_extension(filename):
    for ext in SUPPORTED_EXTENSIONS:
        if filename.endswith(ext):
            base = filename[:-len(ext)]
            return base, ext
    return None, None

def extract_zip(file_path, output_dir):
    """Handle ZIP extraction with multiple encoding fallbacks"""
    encodings = ['gbk', 'utf-8', 'cp437', None]  # None = system default
    for encoding in encodings:
        try:
            kwargs = {'metadata_encoding': encoding} if encoding is not None else {}
            with zipfile.ZipFile(file_path, 'r', **kwargs) as zip_ref:
                zip_ref.extractall(output_dir)
            return True
        except UnicodeDecodeError:
            continue  # Try next encoding
        except zipfile.BadZipFile as e:
            print(f"Bad ZIP file: {os.path.basename(file_path)} - {str(e)}")
            return False
    print(f"Failed decoding ZIP metadata: {os.path.basename(file_path)}")
    return False

def unzip_files(source_dir):
    target_dir = "../2024submission"
    os.makedirs(target_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        if not os.path.isfile(file_path):
            continue
        
        base, ext = get_base_extension(filename)
        if not ext:
            print(f"Skipped unsupported file: {filename}")
            continue

        output_dir = os.path.join(target_dir, base)
        os.makedirs(output_dir, exist_ok=True)

        try:
            if ext == '.zip':
                if extract_zip(file_path, output_dir):
                    print(f"Success: {filename} → {output_dir}")
            
            elif ext == '.rar':
                # Use base rarfile.Error instead of specific exceptions
                with rarfile.RarFile(file_path, 'r') as rar_ref:
                    rar_ref.extractall(output_dir)
                print(f"Success: {filename} → {output_dir}")
            
            else:  # Tar files
                # Added fallback for tar encoding issues
                try:
                    with tarfile.open(file_path, 'r:*', encoding='gbk') as tar_ref:
                        tar_ref.extractall(output_dir)
                except UnicodeDecodeError:
                    with tarfile.open(file_path, 'r:*') as tar_ref:  # Default encoding
                        tar_ref.extractall(output_dir)
                print(f"Success: {filename} → {output_dir}")
        
        except (zipfile.BadZipFile, tarfile.TarError, rarfile.Error) as e:  # Fixed exception handling
            print(f"Failed: {filename} - {str(e)}")
        except Exception as e:
            print(f"Unknown error: {str(e)} handling file: {filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python unzip.py <path to the directory>")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    if not os.path.isdir(source_dir):
        print(f"Error: Path {source_dir} not found")  # Fixed variable name
        sys.exit(1)

    unzip_files(source_dir)