import os
import re
import sys
import shutil
import csv
import hashlib
import pdfplumber
import fitz
import io
import pandas as pd
import tempfile
import logging
import xml.etree.ElementTree as ET
import zipfile
import tarfile
import chardet
import jieba
import json
import numpy as np
from collections import defaultdict
from jinja2 import Environment, FileSystemLoader


logging.getLogger('pdfminer').setLevel(logging.ERROR)
os.environ['UNRAR_LIB_PATH'] = os.path.expanduser('~/.local/lib/libunrar.so')
# 设置UTF-8环境变量
os.environ['LANG'] = 'zh_CN.UTF-8'
os.environ["PYMUPDF_SILENT"] = "yes" 

from datetime import datetime
from tqdm import tqdm
from PIL import Image
from docx import Document
from unrar import rarfile


# 支持的压缩文件后缀
SUPPORTED_EXTENSIONS = {
    '.tar.gz', '.tgz',
    '.tar.bz2', '.tbz2',
    '.tar',
    '.zip',
    '.rar'
}

MIN_SID_LENGTH = 10
KEIL_PROJECT_EXTS = ('.uvproj', '.uvprojx')
CODE_EXTENSIONS = ('.c', '.h')

STOPWORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', 
    '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '样', '看', '过', '用', '她', '他', '我们', '这', '那', 
    '中', '下', '为', '来', '还', '又', '对', '能', '可以', '吧', '啊', '吗', '呢', '啦', '给', '等', '与', '跟', '让', 
    '被', '把', '而', '且', '或', '如果', '因为', '所以', '但是', '然后', '虽然', '即使', '并且', '或者', '因此', '例如', 
    '比如', '首先', '其次', '最后', '结果', '实验', '数据', '分析', '步骤', '方法', '目的', '要求', '内容', '原理', '仪器', 
    '设备', '操作', '记录', '结果', '讨论', '结论', '报告', '小组', '成员'
])