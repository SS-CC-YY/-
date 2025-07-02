from utils.utils import *

# 提取报告函数
def extract_file(dir_path, group_output_dir):
    report_output_dir = os.path.join(group_output_dir, "实验报告")
    os.makedirs(report_output_dir, exist_ok=True)


    group_files = {}
    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if not d.startswith(('.', '_'))]
        group_name = os.path.basename(root)
        for file in files:
            # 寻找报告文件，考虑到报告文件的唯一性，因此只需确保不是代码文件即可
            if (file.lower().endswith(('.pdf', '.doc', '.docx'))) and \
               ("代码" not in file and "源码" not in file):
                file_path = os.path.join(root, file)
                if group_name not in group_files:
                    group_files[group_name] = []
                group_files[group_name].append(file_path)
        
    # 如果存在多个文件，则按照pdf，docx，doc的顺序排序，并选取第一个文件
    for group_name, report_files in group_files.items():
        if len(report_files) == 1:
            report = report_files[0]
        else:
            sorted_files = sorted(report_files, key=lambda f: (
                0 if f.lower().endswith('.pdf') else
                1 if f.lower().endswith('.docx') else
                2
            ))
            report = sorted_files[0]
        flag = extract_format(group_name, report, report_output_dir)
    return flag

# 确认文件后缀
def extract_format(group_name, file_path, output_dir):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_pdf_content(group_name, file_path, output_dir)
    elif ext == '.docx':
        return extract_docx_content(group_name, file_path, output_dir)
    elif ext == '.doc':
        print(f"\n注意：小组{group_name}使用了doc格式报告")
        print("doc格式文件效果有限，建议转化为pdf或者docx格式")
        return extract_doc_content(group_name, file_path, output_dir)
    else:
        print(f"\n无法提取该组实验报告：{group_name}\n")
        return False


# 提取pdf格式报告文件
def extract_pdf_content(group_name, pdf_path, output_dir):

    # 创建文字，图片表格文件夹
    text_dir = os.path.join(output_dir, "文字")
    img_dir = os.path.join(output_dir, "图片")
    table_dir = os.path.join(output_dir, "表格")
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    all_text = []
    image_count = 0

    # 提取每页的文本内容
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        all_text.append(text)

    # 保存到全部文本之后
    full_text = "\n".join(all_text)
    with open(os.path.join(text_dir, "完整文本.txt"), "w", encoding="utf-8") as f:
        f.write(full_text)

    # 提取表格信息
    extract_table_content(pdf_path, table_dir)


    for page_num in range (len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)

        for image_inedx, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            img_ext = base_image["ext"]
            img_path = os.path.join(img_dir, f"page_{page_num+1}_img_{image_inedx+1}.{img_ext}")
            with open(img_path, "wb") as f:
                f.write(image_bytes)

            image_count += 1
    return True

def extract_table_content(pdf_path, table_dir):
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            table = page.extract_tables()

            if table:
                # print(1)
                for table_num, table in enumerate(table):
                    cleaned_table = []
                    for row in table:
                        cleaned_row = [cell.replace("\n", " ") if cell else "" for cell in row]
                        cleaned_table.append(cleaned_row)
                    
                    csv_path = os.path.join(table_dir, f"page_{page_num+1}_table_{table_num+1}")

                    with open(csv_path, "w", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(cleaned_table)

def extract_docx_images(temp_dir, img_dir):
    """从解压的DOCX文件中提取图片"""
    media_path = os.path.join(temp_dir, 'word', 'media')
    if os.path.exists(media_path):
        img_count = 1
        for img_file in os.listdir(media_path):
            src_path = os.path.join(media_path, img_file)
            if os.path.isfile(src_path):
                # 获取文件扩展名
                _, ext = os.path.splitext(img_file)
                dest_path = os.path.join(img_dir, f"image_{img_count}{ext}")
                shutil.copy(src_path, dest_path)
                img_count += 1

def extract_docx_tables(docx_path, table_dir):
    try:
        doc = Document(docx_path)
        table_count = 1
        
        for table in doc.tables:
            table_data = []
            for row in table.rows:
                row_data = []
                for cell in row.cells:
                    cell_text = " ".join(p.text for p in cell.paragraphs)
                    row_data.append(cell_text.strip().replace('\n', ' '))
                table_data.append(row_data)
            

            if table_data and any(any(cell for cell in row) for row in table_data):
                csv_path = os.path.join(table_dir, f"table_{table_count}.csv")
                with open(csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(table_data)
                table_count += 1
    except ImportError:
        print("警告：未安装python-docx库，跳过表格提取")
    except Exception as e:
        print(f"\n提取表格时出错: {e}")


def extract_xml_text(xml_path, namespaces):
    """从XML文件中提取文本内容"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 提取所有文本内容
        text_parts = []
        for t in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
            if t.text:
                text_parts.append(t.text.strip())
        
        return " ".join(text_parts)
    except Exception as e:
        print(f"解析XML文件 {xml_path} 时出错: {e}")
        return ""


def extract_docx_text(temp_dir):
    ns = {
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
    }
    text_parts = []

    document_path = os.path.join(temp_dir, 'word', 'document.xml')
    if os.path.exists(document_path):
        text_parts.append(extract_xml_text(document_path, ns))

    header_dir = os.path.join(temp_dir, 'word', 'header')
    if os.path.exists(header_dir):
        for header_file in os.listdir(header_dir):
            if header_file.endswith('.xml'):
                header_path = os.path.join(header_dir, header_file)
                text_parts.append(extract_xml_text(header_path, ns))

    footer_dir = os.path.join(temp_dir, 'word', 'footer')
    if os.path.exists(footer_dir):
        for footer_file in os.listdir(footer_dir):
            if footer_file.endswith('.xml'):
                footer_path = os.path.join(footer_dir, footer_file)
                text_parts.append(extract_xml_text(footer_path, ns))
    
    return "\n\n".join(text_parts)


def extract_docx_content(group_name, docx_path, output_dir):


    text_dir = os.path.join(output_dir, "文字")
    img_dir = os.path.join(output_dir, "图片")
    table_dir = os.path.join(output_dir, "表格")
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    tmp_dir = tempfile.mkdtemp()

    try:
        with zipfile.ZipFile(docx_path, 'r') as zip_file:
            zip_file.extractall(tmp_dir)

        text_content = extract_docx_text(tmp_dir)
        with open(os.path.join(text_dir, "完整文本.txt"), "w", encoding="utf-8") as f:
            f.write(text_content)

        extract_docx_images(tmp_dir, img_dir)

        extract_docx_tables(docx_path, table_dir)

        return True

    except Exception as e:
        print(f"\n处理该小组docx文件时出错： {group_name}：{e}")
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def extract_doc_content(group_name, doc_path, output_dir):
    """提取DOC文件内容（基本文本提取）"""
    # os.makedirs(output_dir, exist_ok=True)
    text_dir = os.path.join(output_dir, "文字")
    img_dir = os.path.join(output_dir, "图片")
    table_dir = os.path.join(output_dir, "表格")
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    
    # 尝试使用antiword提取文本（Linux）
    if sys.platform == 'linux' or sys.platform == 'linux2':
        try:
            from subprocess import check_output
            text = check_output(['antiword', doc_path]).decode('utf-8', errors='ignore')
            with open(os.path.join(text_dir, "完整文本.txt"), "w", encoding="utf-8") as f:
                f.write(text)
            return True
        except Exception:
            return False
            # pass
    
    # 基本替代方案
    with open(os.path.join(text_dir, "完整文本.txt"), "w", encoding="utf-8") as f:
        f.write("DOC格式提取需要安装antiword（Linux）或textract（Windows/Mac）")
    
    with open(os.path.join(img_dir, "说明.txt"), "w", encoding="utf-8") as f:
        f.write("DOC格式不支持图片提取")
    
    with open(os.path.join(table_dir, "说明.txt"), "w", encoding="utf-8") as f:
        f.write("DOC格式不支持表格提取")
    return True

def extract_table_content(pdf_path, table_dir):
    """提取PDF中的表格"""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            
            if tables:
                for table_num, table in enumerate(tables):
                    cleaned_table = []
                    for row in table:
                        cleaned_row = [cell.replace("\n", " ") if cell else "" for cell in row]
                        cleaned_table.append(cleaned_row)
                    
                    csv_path = os.path.join(table_dir, f"page_{page_num+1}_table_{table_num+1}.csv")
                    
                    with open(csv_path, "w", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(cleaned_table)