from utils.utils import *

class CodePre_process:
    @staticmethod
    def pre_process(code):
        code = re.sub(r'//.*', '', code)  # 移除单行注释
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # 多行注释
        code = re.sub(r'\s+', ' ', code).strip()
        return code.lower()

    @classmethod
    def detect_file_encoding(cls, file_path):
        """检测文件编码"""
        try:
            # 先尝试读取前1024字节来检测编码
            with open(file_path, 'rb') as f:
                raw_data = f.read(1024)
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'gbk'  # 默认使用gbk
                confidence = result['confidence']
                
                # 如果置信度低于90%，尝试常见中文编码
                if confidence < 0.9:
                    try:
                        raw_data.decode('gbk')
                        return 'gbk'
                    except:
                        pass
                    
                    try:
                        raw_data.decode('gb2312')
                        return 'gb2312'
                    except:
                        pass
                    
                    try:
                        raw_data.decode('gb18030')
                        return 'gb18030'
                    except:
                        pass
                
                return encoding
        except Exception as e:
            print(f"编码检测失败 {file_path}: {str(e)}")
            return 'gbk' 


    @classmethod
    def read_group_code(cls, group_meta):
        """读取小组所有代码"""
        combined = []
        for file_path in group_meta.files:
            try:
                encoding = cls.detect_file_encoding(file_path)
                
                # 使用检测到的编码打开文件
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    code = f.read()
                    processed = cls.pre_process(code)
                    # combined.append(processed[:2000])
                    combined.append(processed)
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='gbk', errors='replace') as f:
                        code = f.read()
                        processed = cls.pre_process(code)
                        combined.append(processed[:2000])
                except:
                    print(f"最终读取失败 {file_path}")
            except Exception as e:
                print(f"读取文件错误 {file_path}: {str(e)}")
        return ' '.join(combined)