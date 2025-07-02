# 简易代码报告查重系统

## 功能：
1. 批量处理提交的学生文件
2. 自动提取项目中的代码以及文字报告
3. 针对报告和代码进行查重
4. 通过网页的方式展示结果

## 使用方式
(可选) 建议先创建虚拟环境，防止原生环境被污染以及遇到一些意料之外的问题
使用venv:\\
``python -m venv ENV_DIR``\\
``source ENV_DIR/bin/activate``(linux) ``.\ENV_DIR\Scripts\activate``(WINDOWS)\\
``deactivate``\\
使用conda：\\
``conda create -n ENV_NAME python-x.x``\\
``conda activate ENV_NAME``\\
``conda deactivate``\\

1. 安装必要的库： ``pip install requirements.txt``
2. 由于考虑到部分学生提交的作业压缩包是rar格式，因此需要在环境中引入rar包。仓库中有自带的rar包，需要将其中的``libunrar.so``放入到``~/.local/lib``中，或者直接修改utils.utils中的环境变量。
3. 修改 CONFIG中数据缓存以及作业数据文件夹的路径到想要的路径
4. 运行 ``python code/main.oy``

后续按照终端上的指引进行操作即可。

注意：
遇到存在问题的小组会存储到 ``code/数据缓存/存在问题的小组`` 这里会需求手动增添组员名称以及学号
同时遇到没有提交代码，或者使用.c/,pdf格式以外的格式提交的小组会在终端上打出并进行警告
最后结果需要先本地运行一个临时的服务器，然后就可以通过网页来访问最终的结果


## 后续可能的改进方案：
1. 本项目使用了原始的winnowing算法来进行检测，后续如果对winnowing算法改进的算法可以进行更替
2. 由于小组作业时针对keil 5项目的代码，因此暂时没有找到成功将所有小组代码转化为AST的方法。如果后续能将所有小组的代码转化为AST，即可通过语义树以及神经网络的结合来增强检测和查重的能力
3. 考虑到目前提交的小组数量较少，因此只存储了经过提取后的代码以及文字报告，后续如果小组数量进一步增加可以考虑存储所有小组的指纹，减少重复运算的时间或者可以使用其他更有效率的存储或处理方法。
4. 开发的时候是基于linux服务器进行的操作，因此后续可以考虑迁移到windows系统上。一是方便GUI的开发，二是keil5的代码编译器本身就是基于windows的，也有助于后续其他方法的开发。
5. 目前经过能成功提取小组中的文字，图片以及表格并进行了分别的处理。但是由于没有想到比较好的处理方法，因此只是针对提取出来的完整的文字部分进行检测。后续可以针对报告中的图片以及表格数据部分进行改进升级
6. 经过一些人工的翻阅，个人认为许多的报告内容存在通过AI修改的方式”改进“了报告，因此可以尝试进行ai检测。但是考虑到所有的作业都是针对相同的任务，并且现有的一些大模型确实对作业效率有提升，因此建议后续有更先进的算法的话再进行修改，或者是降低该方法的比重。


## 引用
Saul Schleimer, Daniel S. Wilkerson, and Alex Aiken. 2003. Winnowing: local algorithms for document fingerprinting. In Proceedings of the 2003 ACM SIGMOD international conference on Management of data (SIGMOD '03). Association for Computing Machinery, New York, NY, USA, 76–85. https://doi.org/10.1145/872757.872770\\
电子版文档可从stanford官网上找到：https://theory.stanford.edu/~aiken/publications/papers/sigmod03.pdf
