# -------------------------------------------------------------------------------------------------
# 导入标准库模块
# -------------------------------------------------------------------------------------------------
import sqlite3 # 导入 SQLite 数据库模块，用于存储和管理文档的元数据，如ID、文本描述和图像路径。
import os      # 导入操作系统模块，提供与操作系统交互的功能，如文件路径操作、检查文件或目录是否存在、创建目录等。
import numpy as np # 导入 NumPy 库，用于高效的数值计算，特别是在处理向量（如CLIP模型生成的特征向量）时非常有用。
from typing import List, Dict, Union, Optional, Tuple, Any # 导入类型提示模块，用于在代码中添加类型注解，以增强代码的可读性、可维护性，并帮助静态类型检查工具发现潜在错误。 (Added Any for broader compatibility in some dicts)
import json    # 导入 JSON 库，用于处理 JSON (JavaScript Object Notation) 格式的数据，常用于配置文件读写、API数据交换等。
import time    # 导入时间库，提供时间相关的函数，如获取当前时间、程序暂停（sleep）等。
import random  # 导入随机库，用于生成伪随机数，例如在示例查询中随机选择文档。
import logging # 导入日志模块，用于记录程序运行过程中的信息、警告和错误，方便调试和监控。
import sys     # 导入系统模块，提供访问由 Python 解释器使用或维护的变量和函数的接口，如此处用于配置日志输出到标准输出。
import datetime # 导入日期时间模块，用于处理日期和时间，如此处用于生成带有时间戳的目录名，确保每次运行输出的唯一性。
import re      # 导入正则表达式模块，用于进行强大的文本模式匹配和字符串操作，如此处用于清理文件名中的非法字符。

# -------------------------------------------------------------------------------------------------
# 导入第三方库模块 (需要预先安装)
# -------------------------------------------------------------------------------------------------
import faiss   # 导入 Faiss 库，一个由 Facebook AI Research 开发的高效向量相似性搜索和聚类库。
               # 安装提示: pip install faiss-cpu (CPU版本) 或 faiss-gpu (GPU版本，需CUDA环境)。
from transformers import CLIPProcessor, CLIPModel # 从 Hugging Face Transformers 库导入 CLIP 模型的处理器和模型本身。
                                                 # CLIP (Contrastive Language–Image Pre-training) 是一种强大的多模态模型，能将文本和图像编码到同一向量空间。
                                                 # 安装提示: pip install transformers torch pillow。
from PIL import Image, UnidentifiedImageError # 导入 Pillow 库 (PIL fork)，用于图像文件的加载、处理和保存。 (Added UnidentifiedImageError for specific exception handling)
import torch   # 导入 PyTorch 库，一个广泛使用的开源机器学习框架，Transformers 库基于它构建。
import zhipuai # 导入 ZhipuAI 客户端库，用于与智谱 AI 开发的大语言模型 API 进行交互。
               # 安装提示: pip install zhipuai。

# -------------------------------------------------------------------------------------------------
# 全局日志记录器设置 (在 `if __name__ == "__main__":` 中进一步配置)
# -------------------------------------------------------------------------------------------------
logger = logging.getLogger(__name__) # 初始化一个模块级别的日志记录器实例。`__name__` 会被设置成当前模块的名称。

# -------------------------------------------------------------------------------------------------
# 工具函数定义
# -------------------------------------------------------------------------------------------------
def setup_logging(log_file_path: str):
    """
    配置全局日志记录器 (logger)。
    该函数设置日志记录级别、格式，并指定日志信息同时输出到控制台和指定的日志文件。

    Args:
        log_file_path (str): 日志文件的完整路径。程序运行的所有日志信息将被写入此文件。
    """
    global logger # 声明我们要修改的是全局变量 `logger`
    logger.setLevel(logging.INFO) # 设置日志记录的最低级别为 INFO。只有 INFO 及以上级别（如 WARNING, ERROR, CRITICAL）的日志才会被处理。

    # 在添加新的处理器之前，清除可能已存在的旧处理器，以避免重复记录日志。
    # 这在脚本被多次调用或在交互式环境中使用时尤其重要。
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建一个文件处理器 (FileHandler)，用于将日志信息写入到指定的日志文件。
    # `encoding='utf-8'` 确保日志文件能正确处理中文字符。
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='w') # 'w' mode to overwrite log for each run
    file_handler.setLevel(logging.INFO) # 文件处理器也只处理 INFO 及以上级别的日志。

    # 创建一个控制台处理器 (StreamHandler)，用于将日志信息输出到标准输出（通常是终端控制台）。
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # 控制台处理器同样只处理 INFO 及以上级别的日志。

    # 定义日志格式器 (Formatter)，它决定了每条日志记录的显示格式。
    # 格式字符串包含:
    #   %(asctime)s: 日志记录的创建时间。
    #   %(levelname)s: 日志级别 (例如 INFO, WARNING)。
    #   [%(module)s.%(funcName)s:%(lineno)d]: 日志发出的模块名、函数名和行号，便于定位问题。
    #   %(message)s: 实际的日志消息内容。
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s.%(funcName)s:%(lineno)d] - %(message)s') # Changed module to filename for better clarity
    
    # 将定义好的格式器应用到文件处理器和控制台处理器。
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将配置好的文件处理器和控制台处理器添加到全局日志记录器中。
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("全局日志记录器配置完成。日志将输出到控制台，并写入文件: %s", log_file_path) # 记录一条日志，表明配置成功。

def sanitize_filename(filename: str, max_length: int = 100, is_dir_component: bool = False) -> str:
    """
    清理字符串，使其成为一个有效的文件名或目录名组件。
    该函数会替换或移除文件名中不被大多数文件系统允许的特殊字符，并将文件名截断到指定的最大长度。

    Args:
        filename (str): 需要被清理的原始字符串。
        max_length (int): 清理后文件名的最大允许长度。默认为 100 个字符。
        is_dir_component (bool): (此参数在此实现中未产生不同行为) 指示该字符串是否用作目录路径的一部分。
                                 理论上，目录组件可能对某些字符（如路径分隔符）有不同处理，但为简单和安全起见，
                                 此函数对文件名和目录名组件采用相同的严格清理规则。

    Returns:
        str: 清理和截断后的、可以用作文件系统名称的字符串。
    """
    # 如果输入的文件名为空或 None，返回一个默认的占位符名称。
    if not filename:
        return "unnamed_component" # 未命名组件
    
    # 使用正则表达式替换掉文件名中常见的非法字符。
    # 这些字符包括: \ / * ? : " < > | (反斜杠, 正斜杠, 星号, 问号, 冒号, 双引号, 小于号, 大于号, 竖线)
    # 将这些非法字符统一替换为下划线 "_"。
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    
    # 将字符串两端的空白字符（空格、制表符、换行符等）去除。
    # 然后，将字符串内部的一个或多个连续空白字符替换为单个下划线 "_"。
    sanitized = re.sub(r'\s+', '_', sanitized.strip())
    
    # 移除文件名开头可能存在的点和下划线，避免生成隐藏文件或不规范的名称
    sanitized = re.sub(r'^[\._]+', '', sanitized)

    # 将清理后的字符串截断到 `max_length` 指定的最大长度。
    # 注意：简单的切片可能在多字节字符（如某些中文）的中间截断，导致乱码。
    # 对于主要处理英文或简单场景，此方法可行。若需完美处理多字节字符，需要更复杂的截断逻辑。
    sanitized = sanitized[:max_length]

    # 再次检查，如果清理和截断后字符串变为空，或者只包含点号 "." (可能导致隐藏文件或路径问题)，
    # 则返回一个特定的占位符名称。
    if not sanitized or all(c == '.' for c in sanitized):
        return "sanitized_empty_name" # 清理后为空的名称
    
    # 避免使用 Windows 系统中的保留设备名作为文件名（不区分大小写）。
    # 例如: CON, PRN, AUX, NUL, COM1-COM9, LPT1-LPT9。
    # 如果清理后的名称（转换为大写后）匹配这些保留名，则在其前后添加下划线以作区分。
    # 这是一个简化的检查，完整的跨平台文件名验证会更复杂。
    reserved_names_check = sanitized.upper()
    if reserved_names_check in ["CON", "PRN", "AUX", "NUL"] or \
       re.match(r"COM[1-9]$", reserved_names_check) or \
       re.match(r"LPT[1-9]$", reserved_names_check):
        sanitized = f"_{sanitized}_" # 在保留名称前后加下划线

    return sanitized # 返回最终清理后的文件名字符串。

# -------------------------------------------------------------------------------------------------
# 数据加载与预处理模块
# -------------------------------------------------------------------------------------------------
def load_data_from_json_and_associate_images(json_path: str, image_dir: str) -> List[Dict[str, Any]]:
    """
    从指定的 JSON 文件加载文档的元数据 (如 ID 和描述文本)，
    并根据文档 ID (JSON中的 'name' 字段) 在指定的图像目录中查找并关联对应的图像文件。
    函数假设图像文件名是文档 ID 加上常见的图片扩展名 (如 .png, .jpg)。

    Args:
        json_path (str): 包含文档元数据的 JSON 文件路径。
                         JSON 文件应为一个列表，其中每个对象至少包含 'name' 和 'description' 字段。
        image_dir (str): 存放与 JSON 数据对应的图片文件的目录路径。

    Returns:
        List[Dict[str, Any]]: 一个包含处理后文档信息的字典列表。
                    每个字典包含以下键：
                    - 'id': 文档的唯一标识符 (来自 JSON 'name' 字段，确保为字符串)。
                    - 'text': 文档的文本描述 (来自 JSON 'description' 字段，确保为字符串或 None)。
                    - 'image_path': 找到的对应图像文件的完整路径 (str)。如果未找到图像或 image_dir 无效，则为 None。
                    如果 JSON 文件不存在、无法解析或读取失败，则返回空列表 ([]).
    """
    # 获取当前模块的日志记录器实例，用于记录此函数的执行信息。
    func_logger = logging.getLogger(__name__) # 使用模块级 logger
    func_logger.info(f"开始从 JSON 文件 '{json_path}' 加载数据，并在目录 '{image_dir}' 中关联图像...")

    # 步骤 1: 检查 JSON 文件是否存在。
    if not os.path.exists(json_path):
        func_logger.error(f"错误：JSON 文件 '{json_path}' 未找到。请检查文件路径是否正确。")
        return [] # 文件不存在，无法继续，返回空列表。

    # 初始化用于存储最终处理后文档信息的列表。
    documents: List[Dict[str, Any]] = [] 
    # 定义一个包含常见图像文件扩展名的列表，用于查找匹配的图像文件。
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'] # 增加了 .webp 格式

    # 步骤 2: 尝试打开并解析 JSON 文件。
    try:
        # 使用 'with' 语句确保文件在使用后自动关闭，即使发生错误。
        # 'r' 表示以只读模式打开文件。
        # 'encoding='utf-8'' 指定使用 UTF-8 编码读取文件，以正确处理中文字符。
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f) # 解析 JSON 数据，将其转换为 Python 的列表或字典。
            if not isinstance(json_data, list):
                func_logger.error(f"错误: JSON 文件 '{json_path}' 的顶层结构不是一个列表。请确保JSON文件格式正确。")
                return []
    except json.JSONDecodeError as e:
        # 如果 JSON 文件内容格式不正确，json.load() 会抛出 JSONDecodeError。
        func_logger.error(f"错误：JSON 文件 '{json_path}' 解析失败。错误详情: {e}")
        func_logger.error(f"        请确保文件内容是有效的 JSON 格式 (一个包含对象的列表)。")
        return [] # JSON 格式错误，返回空列表。
    except Exception as e:
        # 捕获其他可能的读取文件错误，例如权限问题。
        func_logger.error(f"错误：读取 JSON 文件 '{json_path}' 时发生未知错误。错误详情: {e}")
        return [] # 其他读取错误，返回空列表。

    func_logger.info(f"已成功从 '{json_path}' 加载 {len(json_data)} 条原始记录。")
    
    # 初始化计数器，用于统计数据处理过程中的情况。
    found_images_count = 0    # 成功关联到图像的文档数量。
    missing_key_count = 0     # 因缺少必要字段 ('name' 或 'description') 而被跳过的记录数量。

    # 步骤 3: 遍历从 JSON 文件加载的每一条原始记录。
    for item_index, item in enumerate(json_data): # 使用 enumerate 获取索引，方便日志记录
        if not isinstance(item, dict):
            func_logger.warning(f"警告：跳过第 {item_index + 1} 条记录，因其不是一个有效的字典对象。记录内容: {item}")
            missing_key_count += 1
            continue

        doc_id = item.get('name')         # 尝试获取 'name' 字段作为文档 ID。
        text_content = item.get('description') # 尝试获取 'description' 字段作为文本内容。

        # 检查关键字段 'name' 和 'description' 是否存在且有值。
        # 如果任一字段缺失，则跳过该条记录。
        if not doc_id or not text_content: # Both name and description must exist
            missing_key_count += 1
            func_logger.warning(f"警告：跳过第 {item_index + 1} 条记录（JSON索引 {item_index}），因缺少 'name' 或 'description' 字段。记录内容: {item}")
            continue # 继续处理下一条记录。

        # 初始化图像路径为 None。如果在指定目录中找不到匹配的图像，它将保持为 None。
        image_path: Optional[str] = None 
        # 检查图像目录路径是否有效（已提供且存在于文件系统中）。
        if image_dir and os.path.isdir(image_dir): # 确保 image_dir 是一个存在的目录
            # 遍历预定义的图像扩展名列表，尝试构建并查找图像文件。
            for ext in image_extensions:
                # 构建潜在的图像文件名：文档ID（来自 'name' 字段）+ 当前扩展名。
                # 使用 str(doc_id) 确保即使 doc_id 是数字也能正确拼接。
                potential_image_filename = str(doc_id) + ext
                # 使用 os.path.join 安全地构建跨平台的完整图像文件路径。
                potential_image_path = os.path.join(image_dir, potential_image_filename)
                
                # 检查构建的图像文件路径是否存在于文件系统中。
                if os.path.exists(potential_image_path) and os.path.isfile(potential_image_path): # 确保是文件
                    image_path = potential_image_path # 找到图像，记录其完整路径。
                    found_images_count += 1           # 增加找到图像的计数。
                    break # 找到一个匹配的图像后，无需再检查其他扩展名，跳出内层循环。
        elif image_dir and not os.path.isdir(image_dir):
            # 如果提供了 image_dir 但它不是一个有效的目录，记录一次警告。
            # 为避免日志泛滥，此警告只在第一次检测到时发出（通过在循环外设置一个标志）
            # (为简化，此处每次都记录，但可优化)
            func_logger.warning(f"提供的图像目录 '{image_dir}' 不是一个有效的目录，将无法关联图像。")
        elif not image_dir:
            func_logger.debug(f"未提供图像目录 (image_dir is None or empty)，将不尝试关联图像。")


        # 将处理后的文档信息（包括 ID、文本和可能的图像路径）添加到 `documents` 列表中。
        documents.append({
            'id': str(doc_id), # 确保文档 ID 是字符串类型。
            'text': str(text_content) if text_content is not None else None, # 确保文本是字符串；如果原始为 None，则保持 None。
            'image_path': image_path # 存储找到的图像路径，如果未找到则为 None。
        })

    # 步骤 4: 打印数据加载和关联过程的总结信息。
    func_logger.info(f"成功准备了 {len(documents)} 个文档用于后续处理。")
    if missing_key_count > 0:
        func_logger.warning(f"在原始 JSON 数据中，共有 {missing_key_count} 条记录因格式无效或缺少 'name'/'description' 字段而被跳过。")
    func_logger.info(f"在有效文档中，共有 {found_images_count} 个文档成功关联了图像文件。")
    
    # 如果指定了图像目录，但没有找到任何图像文件（并且至少有一个文档被处理了），则给出提示。
    if len(documents) > 0 and found_images_count == 0 and image_dir and os.path.isdir(image_dir):
         func_logger.info(f"提示: 未在目录 '{image_dir}' 中找到任何与文档 ID 匹配的图像文件。")
         func_logger.info(f"        请检查图像文件名是否严格遵循 '文档ID.扩展名' 的格式 (例如，如果文档 'name' 是 'item01'，则图像应为 'item01.png')。")
    
    func_logger.info(f"--- 数据加载与图像关联流程结束 ---")
    return documents # 返回包含所有已处理文档信息的列表。

# -------------------------------------------------------------------------------------------------
# 多模态编码器类 (MultimodalEncoder)
# -------------------------------------------------------------------------------------------------
class MultimodalEncoder:
    """
    使用 Hugging Face Transformers 库中的 CLIP (Contrastive Language–Image Pre-training) 模型
    来对文本和/或图像进行编码，将它们转换为高维向量表示 (特征向量)。
    CLIP 模型能够将文本和图像映射到同一个语义向量空间，使得它们的向量表示具有可比性，
    这是多模态检索和理解的基础。

    核心功能:
    - 在初始化时加载预训练的 CLIP 模型和对应的处理器 (processor)。
    - 提供 `encode` 方法，该方法可以接受文本字符串、图像文件路径，或两者都接受。
    - `encode` 方法对输入进行预处理、通过 CLIP 模型进行编码，然后对输出的向量进行 L2 归一化。
    - 返回一个字典，包含文本向量、图像向量以及（如果两者都提供了）两者的平均向量。
    - 自动检测并优先使用 GPU (如果 CUDA 可用) 进行计算加速，否则回退到 CPU。
    - L2 归一化对于后续使用 Faiss 进行基于内积 (Inner Product) 的相似度搜索至关重要，
      因为归一化向量的内积等价于它们之间的余弦相似度。
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        初始化 MultimodalEncoder 类。

        Args:
            model_name (str): 指定要加载的 Hugging Face Hub 上的 CLIP 模型名称。
                              例如 "openai/clip-vit-base-patch32"。
                              不同的 CLIP 模型变体具有不同的性能、速度和输出向量维度。
                              选择合适的模型取决于具体的应用需求和可用资源。
                              老板请注意: "openai/clip-vit-base-patch32" 是一个性能和资源消耗均衡的基准模型。
                              若资源极度受限，可研究更轻量模型，但可能影响编码质量。

        Raises:
            Exception: 如果在加载 CLIP 模型或处理器时发生任何错误（例如，网络问题导致无法下载模型文件、
                       指定的模型名称无效、或者相关的依赖库未正确安装），则会抛出异常。
                       由于模型是编码器的核心，加载失败意味着编码器无法工作。
        """
        # 获取一个特定于此类实例的日志记录器，方便追踪和调试。
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.info(f"开始初始化 MultimodalEncoder，尝试加载 CLIP 模型: {model_name}")
        
        try:
            # 步骤 1: 加载与指定 CLIP 模型相关联的处理器 (CLIPProcessor)。
            # 处理器负责将原始的文本和图像数据转换为 CLIP 模型期望的输入格式。
            # 对于文本，这通常包括分词 (tokenization)、添加特殊标记、转换为 token ID。
            # 对于图像，这通常包括调整大小 (resizing)、归一化 (normalization) 像素值。
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.logger.info(f"CLIP Processor for '{model_name}' 加载成功。")

            # 步骤 2: 加载预训练的 CLIP 模型本身 (CLIPModel)。
            self.model = CLIPModel.from_pretrained(model_name)
            self.logger.info(f"CLIP Model '{model_name}' 加载成功。")

            # 步骤 3: 获取模型的输出向量维度。
            # 对于 CLIP 模型，文本编码器和图像编码器的输出向量维度通常是相同的。
            # 这个维度信息对于后续创建 Faiss 索引等操作非常重要。
            # text_model.config.hidden_size 通常存储了这个维度值。
            self.vector_dimension = self.model.text_model.config.hidden_size
            self.logger.info(f"CLIP 模型的特征向量维度为: {self.vector_dimension}")

            # 步骤 4: 将模型设置为评估模式 (evaluation mode)。
            # 调用 .eval() 会关闭模型中的 Dropout 层和 Batch Normalization 层的更新行为。
            # 这对于推理（编码）阶段非常重要，以确保结果的一致性和确定性。
            self.model.eval()

            # 步骤 5: 检测可用的计算设备 (GPU 或 CPU)，并将模型迁移到该设备。
            if torch.cuda.is_available(): # 检查系统中是否有可用的 CUDA GPU。
                self.device = torch.device("cuda") # 如果有，则选择使用 GPU。
                self.logger.info("检测到 CUDA 支持，模型将运行在 GPU 上以获得更快的编码速度。")
            else:
                self.device = torch.device("cpu")  # 如果没有 GPU，则使用 CPU。
                self.logger.info("未检测到 CUDA 支持，模型将运行在 CPU 上 (编码速度可能较慢)。")
            
            self.model.to(self.device) # 将模型的所有参数和缓冲区移动到选定的设备。
            self.logger.info(f"模型已成功移动到设备: {self.device}")
            self.logger.info("MultimodalEncoder 初始化成功完成。")

        except Exception as e:
             # 如果在上述任何步骤中发生错误，记录详细的错误信息并重新抛出异常。
             self.logger.error(f"初始化 MultimodalEncoder 失败：加载 CLIP 模型 '{model_name}' 时发生严重错误。")
             self.logger.error(f"错误详情: {e}", exc_info=True) # exc_info=True 会记录完整的堆栈跟踪。
             self.logger.error("请检查以下几点：")
             self.logger.error(f"  1. 确保指定的模型名称 '{model_name}' 正确且在 Hugging Face Hub 上可用。")
             self.logger.error("  2. 确保已正确安装必要的 Python 库: 'transformers', 'torch', 'pillow'。")
             self.logger.error("     (例如，通过命令: pip install transformers torch pillow)")
             self.logger.error("  3. 确保网络连接正常，以便能够从 Hugging Face Hub 下载模型文件 (首次加载时需要)。")
             raise RuntimeError(f"MultimodalEncoder 初始化失败: {e}") from e # Re-raise as RuntimeError

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        对输入的 NumPy 向量进行 L2 范数归一化 (L2 Normalization)。
        L2 归一化将向量缩放，使其 L2 范数（欧几里得长度）为 1。
        这对于计算余弦相似度非常重要：两个 L2 归一化向量的点积（内积）等于它们之间的余弦相似度。

        Args:
            vector (np.ndarray): 需要进行 L2 归一化的 NumPy 浮点数向量。

        Returns:
            np.ndarray: 经过 L2 归一化后的向量。如果输入向量的范数非常接近于零（即零向量），
                        则直接返回一个相同形状的零向量，以避免除以零的错误。
        """
        # 计算向量的 L2 范数 (向量的欧几里得长度)。
        norm = np.linalg.norm(vector)
        
        # 检查范数是否大于一个很小的阈值 (epsilon)，以避免除以零或因浮点数精度问题导致的数值不稳定。
        # 1e-9 是一个常用的小正数。
        if norm > 1e-9: 
            # 如果范数足够大，则将向量的每个元素除以该范数，得到归一化向量。
            return vector / norm
        else:
            # 如果范数非常小（向量接近零向量），直接返回一个与输入向量形状相同但所有元素为零的向量。
            self.logger.debug("尝试归一化一个范数接近零的向量。返回零向量。")
            return np.zeros_like(vector)

    def encode(self, text: Optional[str] = None, image_path: Optional[str] = None) -> Dict[str, Optional[np.ndarray]]:
        """
        对输入的文本字符串和/或图像文件路径进行编码，生成它们对应的归一化特征向量。

        Args:
            text (Optional[str]): 需要编码的文本字符串。如果为 None 或空字符串，则不进行文本编码。
            image_path (Optional[str]): 需要编码的图像文件的完整路径。如果为 None 或路径无效，则不进行图像编码。

        Returns:
            Dict[str, Optional[np.ndarray]]: 一个字典，包含以下可能的键值对：
                - 'text_vector': 如果提供了有效的文本且编码成功，则为该文本的 L2 归一化 NumPy 向量 (float32)。否则为 None。
                - 'image_vector': 如果提供了有效的图像路径、图像文件可读且编码成功，则为该图像的 L2 归一化 NumPy 向量 (float32)。否则为 None。
                - 'mean_vector': 如果文本和图像都提供了，并且两者都成功编码，则为两者特征向量的 L2 归一化平均向量 (float32)。
                                 这个平均向量可以作为文本和图像结合的多模态表示。如果任一编码失败或未提供，则为 None。
            如果 `text` 和 `image_path` 都为 None，将记录错误并返回所有值为 None 的字典。
        """
        # 输入有效性检查：必须至少提供文本或图像路径之一。
        if text is None and image_path is None:
            self.logger.error("编码错误：必须至少提供文本或图像路径才能进行编码。")
            return {'text_vector': None, 'image_vector': None, 'mean_vector': None}

        # 初始化各个向量为 None，它们将在编码成功后被赋值。
        text_vector: Optional[np.ndarray] = None
        image_vector: Optional[np.ndarray] = None
        mean_vector: Optional[np.ndarray] = None
        
        # 使用 torch.no_grad() 上下文管理器进行推理。
        # 这会禁用 PyTorch 的梯度计算，从而减少内存消耗并加速计算，因为在编码（推理）阶段不需要进行反向传播。
        with torch.no_grad():
            # --- 步骤 A: 编码文本 (如果提供了文本) ---
            if text is not None and text.strip(): # 确保文本非None且非空（去除两端空白后）
                self.logger.debug(f"开始编码文本: '{text[:50]}{'...' if len(text)>50 else ''}'")
                try:
                    # 1. 预处理文本: 使用 CLIP Processor 将文本字符串转换为模型所需的输入格式。
                    #    `return_tensors="pt"`: 返回 PyTorch 张量 (tensors)。
                    #    `padding=True`: 将批次内的文本填充到相同长度 (批次中最长文本的长度)。
                    #    `truncation=True`: 如果文本超过模型的最大输入长度，则进行截断。
                    #    `.to(self.device)`: 将生成的输入张量移动到之前确定的计算设备 (CPU 或 GPU)。
                    text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)

                    # 2. 获取文本特征: 调用 CLIP 模型的 `get_text_features` 方法，传入预处理后的输入。
                    #    使用 `**text_inputs` 将字典解包为关键字参数。
                    text_features_tensor = self.model.get_text_features(**text_inputs)

                    # 3. 后处理特征张量:
                    #    `.squeeze()`: 如果批次大小为1，移除批次维度，得到一个一维张量 (向量)。
                    #    `.cpu()`: 将结果张量从 GPU (如果在使用) 移回 CPU，因为 NumPy 操作通常在 CPU 上进行。
                    #    `.numpy()`: 将 PyTorch 张量转换为 NumPy 数组。
                    #    `.astype('float32')`: 确保数据类型为 float32，这是 Faiss 常用的数值类型，也节省内存。
                    text_vector_raw = text_features_tensor.squeeze().cpu().numpy().astype('float32')

                    # 4. L2 归一化: 对原始的文本特征向量进行 L2 范数归一化。
                    text_vector = self._normalize_vector(text_vector_raw)
                    self.logger.debug("文本编码成功并已归一化。")

                except Exception as e:
                    # 如果文本编码过程中发生任何错误，记录错误信息。
                    self.logger.error(f"编码文本时发生错误。文本: '{text[:50]}...'. 错误详情: {e}", exc_info=False) # exc_info=False 避免在每次文本编码失败时都打印完整堆栈
                    text_vector = None # 确保在失败时 text_vector 为 None。

            # --- 步骤 B: 编码图像 (如果提供了图像路径) ---
            if image_path is not None and image_path.strip(): # 确保图像路径非None且非空
                self.logger.debug(f"开始编码图像: '{image_path}'")
                try:
                    # 1. 加载图像: 使用 Pillow (PIL) 库的 Image.open() 方法打开图像文件。
                    #    `.convert("RGB")`: 确保图像转换为 RGB 格式。CLIP 模型通常期望 RGB 图像作为输入。
                    #                       即使原始图像是 RGBA 或灰度图，也会被转换为 RGB。
                    image_pil = Image.open(image_path).convert("RGB")

                    # 2. 预处理图像: 使用 CLIP Processor 将 PIL.Image 对象转换为模型所需的输入格式。
                    #    `return_tensors="pt"`: 返回 PyTorch 张量。
                    #    `.to(self.device)`: 将输入张量移动到计算设备。
                    image_inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)

                    # 3. 获取图像特征: 调用 CLIP 模型的 `get_image_features` 方法。
                    image_features_tensor = self.model.get_image_features(**image_inputs)

                    # 4. 后处理特征张量 (与文本编码类似): 转换为归一化的 NumPy float32 数组。
                    image_vector_raw = image_features_tensor.squeeze().cpu().numpy().astype('float32')
                    image_vector = self._normalize_vector(image_vector_raw)
                    self.logger.debug(f"图像 '{os.path.basename(image_path)}' 编码成功并已归一化。")

                except FileNotFoundError:
                    # 如果指定的图像文件路径不存在。
                    self.logger.warning(f"图像编码警告: 图像文件未找到于路径 '{image_path}'。将跳过此图像的编码。")
                    image_vector = None
                except UnidentifiedImageError: # Pillow 无法识别图像格式
                    self.logger.error(f"图像编码错误: 无法识别或打开图像文件 '{image_path}'。文件可能已损坏或格式不受支持。")
                    image_vector = None
                except Exception as e:
                    # 如果在加载或处理图像时发生其他错误 (例如，图像文件损坏、权限问题)。
                    self.logger.error(f"编码图像 '{image_path}' 时发生错误。错误详情: {e}", exc_info=False)
                    image_vector = None

        # --- 步骤 C: 计算平均向量 (仅当文本和图像都成功编码时) ---
        # 检查 text_vector 和 image_vector 是否都成功生成 (即它们都不是 None)。
        if text_vector is not None and image_vector is not None:
            self.logger.debug("文本和图像均成功编码，开始计算它们的平均向量...")
            try:
                # 1. 计算平均: 使用 NumPy 的 `mean` 函数计算两个向量的逐元素平均值。
                #    `axis=0` 表示沿着第一个轴（即向量本身）计算平均值。
                #    确保结果的数据类型为 float32。
                mean_vector_raw = np.mean(np.array([text_vector, image_vector]), axis=0).astype('float32')

                # 2. L2 归一化: 对计算出的原始平均向量再次进行 L2 归一化。
                #    这很重要，因为两个单位向量的平均向量长度通常不为 1。
                mean_vector = self._normalize_vector(mean_vector_raw)
                self.logger.debug("平均向量计算并归一化成功。")
            except Exception as e:
                # 如果计算平均向量时出错。
                self.logger.error(f"计算文本和图像的平均向量时发生错误。错误详情: {e}", exc_info=False)
                mean_vector = None
        elif (text_vector is not None or image_vector is not None):
             self.logger.debug("仅文本或图像之一被成功编码，因此不计算平均向量。")


        # 总结编码结果，用于日志记录。
        results_summary = []
        if text_vector is not None: results_summary.append("文本向量")
        if image_vector is not None: results_summary.append("图像向量")
        if mean_vector is not None: results_summary.append("平均向量")
        
        input_summary_parts = []
        if text and text.strip(): input_summary_parts.append(f"文本='{text[:30]}...'")
        if image_path and image_path.strip(): input_summary_parts.append(f"图像='{os.path.basename(image_path)}'")
        input_desc = ", ".join(input_summary_parts) if input_summary_parts else "无有效输入"


        if not results_summary and ( (text and text.strip()) or (image_path and image_path.strip()) ):
             self.logger.warning(f"编码完成，但对于输入 ({input_desc})，未能生成任何有效向量。")
        elif results_summary:
             self.logger.info(f"编码完成对于 ({input_desc})。成功生成的向量: {', '.join(results_summary)}。")

        # 返回包含所有结果向量的字典。
        return {
            'text_vector': text_vector,
            'image_vector': image_vector,
            'mean_vector': mean_vector
        }

# -------------------------------------------------------------------------------------------------
# 索引器类 (Indexer)
# -------------------------------------------------------------------------------------------------
class Indexer:
    """
    Indexer 类是多模态 RAG (Retrieval Augmented Generation) 系统的数据管理核心。它负责：
    1.  **接收文档数据**: 从外部（例如，`load_data_from_json_and_associate_images` 函数）获取包含文本和图像路径的文档列表。
    2.  **调用编码器**: 使用内部的 `MultimodalEncoder` 实例对每个文档的文本内容和/或关联图像进行向量化，生成特征向量。
    3.  **存储元数据**: 将文档的原始信息（如原始ID、文本内容、图像文件路径）存储在 SQLite 数据库中。
        数据库中会为每个文档生成一个自增的整数主键 `internal_id`，这个ID将用作 Faiss 索引中对应向量的唯一标识符。
    4.  **构建和管理向量索引**:
        -   创建并维护 **三个独立** 的 Faiss 索引：一个用于存储纯文本向量，一个用于存储纯图像向量，一个用于存储文本和图像结合的平均向量。
        -   每个 Faiss 索引都使用 `IndexIDMap2` 类型，这允许我们将向量与我们自定义的 `internal_id` (来自SQLite数据库) 关联起来，方便后续检索和数据回溯。
        -   索引使用内积 (`IndexFlatIP`) 作为相似度度量方法。由于所有向量都经过了L2归一化，内积等价于余弦相似度，值越大表示越相似。
    5.  **持久化**: 能够从指定文件路径加载先前已保存的索引文件和数据库，或者在首次运行时创建它们。在关闭时，会将当前的索引状态保存到文件，以便下次使用。

    这种分离索引的设计（文本、图像、平均）允许在检索阶段根据用户查询的类型（纯文本、纯图像、或图文多模态）灵活地选择最合适的索引进行搜索，从而提高检索的准确性和效率。
    """
    def __init__(self,
                 db_path: str, 
                 faiss_text_index_path: str, 
                 faiss_image_index_path: str, 
                 faiss_mean_index_path: str, 
                 clip_model_name: str = "openai/clip-vit-base-patch32"):
        """
        初始化 Indexer 实例。

        Args:
            db_path (str): 指定 SQLite 数据库文件的保存路径。
            faiss_text_index_path (str): 指定文本向量 Faiss 索引文件的保存路径。
            faiss_image_index_path (str): 指定图像向量 Faiss 索引文件的保存路径。
            faiss_mean_index_path (str): 指定平均向量 Faiss 索引文件的保存路径。
            clip_model_name (str): 传递给内部 `MultimodalEncoder` 的 CLIP 模型名称。
                                   此模型名称必须与后续用于查询编码的模型保持一致，以确保向量空间的一致性。
        """
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.info("开始初始化 Indexer...")
        
        # 保存传入的路径和模型名称配置。
        self.db_path = db_path
        self.faiss_text_index_path = faiss_text_index_path
        self.faiss_image_index_path = faiss_image_index_path
        self.faiss_mean_index_path = faiss_mean_index_path
        self.logger.info(f"  数据库路径: {self.db_path}")
        self.logger.info(f"  文本索引路径: {self.faiss_text_index_path}")
        self.logger.info(f"  图像索引路径: {self.faiss_image_index_path}")
        self.logger.info(f"  平均向量索引路径: {self.faiss_mean_index_path}")

        # 步骤 1: 初始化多模态编码器 (MultimodalEncoder)。
        # Indexer 内部拥有一个 Encoder 实例，专门用于对其接收的文档进行编码。
        self.logger.info(f"  - 正在初始化内部 MultimodalEncoder，使用 CLIP 模型: {clip_model_name}...")
        try:
            self.encoder = MultimodalEncoder(clip_model_name) # 创建编码器实例。
            self.vector_dimension = self.encoder.vector_dimension # 从编码器获取产生的向量维度
            self.logger.info(f"  - MultimodalEncoder 初始化完成。特征向量维度为: {self.vector_dimension}。")
        except Exception as e_encoder:
            self.logger.critical(f"Indexer 初始化严重失败：内部 MultimodalEncoder 创建失败。错误: {e_encoder}", exc_info=True)
            raise RuntimeError(f"Indexer 无法初始化 Encoder: {e_encoder}") from e_encoder

        # 步骤 2: 初始化 SQLite 数据库 (用于存储文档元数据)。
        # 调用私有方法 `_init_db` 来确保数据库文件存在，并创建所需的表结构（如果尚不存在）。
        self.logger.info(f"  - 正在初始化 SQLite 数据库，路径: '{self.db_path}'...")
        try:
            self._init_db() # 此方法会处理数据库目录的创建。
            self.logger.info(f"  - SQLite 数据库初始化完成。")
        except Exception as e_db_init:
            self.logger.critical(f"Indexer 初始化严重失败：SQLite 数据库初始化失败。错误: {e_db_init}", exc_info=True)
            raise RuntimeError(f"Indexer 无法初始化数据库: {e_db_init}") from e_db_init


        # 步骤 3: 加载或创建三个独立的 Faiss 向量索引。
        # 分别为文本向量、图像向量和平均向量（文本+图像组合）加载或创建 Faiss 索引。
        # `_load_or_create_faiss_index` 方法会处理文件存在性检查、维度匹配和新索引创建的逻辑。
        self.logger.info(f"  - 正在加载或创建 Faiss 向量索引...")
        try:
            self.text_index = self._load_or_create_faiss_index(self.faiss_text_index_path, "文本(Text)")
            self.image_index = self._load_or_create_faiss_index(self.faiss_image_index_path, "图像(Image)")
            self.mean_index = self._load_or_create_faiss_index(self.faiss_mean_index_path, "平均(Mean)")
            self.logger.info(f"  - 所有 Faiss 索引均已准备就绪。")
        except Exception as e_faiss_init:
            self.logger.critical(f"Indexer 初始化严重失败：一个或多个 Faiss 索引加载/创建失败。错误: {e_faiss_init}", exc_info=True)
            raise RuntimeError(f"Indexer 无法初始化 Faiss 索引: {e_faiss_init}") from e_faiss_init


        self.logger.info("Indexer 初始化成功完成。")


    def _init_db(self):
        """
        初始化 SQLite 数据库连接并创建所需的 'documents' 表（如果它还不存在）。
        这个表用于存储文档的元数据，并将原始文档 ID (doc_id) 映射到数据库生成的
        自增主键 `internal_id`。这个 `internal_id` 将作为 Faiss 索引中对应向量的 ID。
        此方法还会确保数据库文件所在的目录存在。
        """
        self.logger.info(f"正在连接并初始化数据库表结构于路径: '{self.db_path}'...")
        
        # 确保数据库文件所在的目录存在，如果不存在则创建它。
        db_directory = os.path.dirname(self.db_path)
        if db_directory and not os.path.exists(db_directory): 
            try:
                os.makedirs(db_directory, exist_ok=True) # exist_ok=True 表示如果目录已存在则不抛出错误。
                self.logger.debug(f"已确保数据库目录 '{db_directory}' 存在 (或已创建)。")
            except OSError as e:
                self.logger.error(f"创建数据库目录 '{db_directory}' 失败: {e}", exc_info=True)
                raise # Re-raise the exception as this is critical

        try:
            # 使用 'with' 语句确保数据库连接在使用后自动关闭，并能自动处理事务（默认提交，出错回滚）。
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor() # 获取数据库游标，用于执行 SQL 命令。
                
                # SQL 语句，用于创建 'documents' 表。
                # `IF NOT EXISTS` 确保如果表已经存在，则不会尝试重新创建它，从而避免错误。
                # 表结构定义:
                #   - internal_id: 整数类型，主键，自动增长。这是数据库内部ID，也将用作Faiss索引的ID。
                #   - doc_id: 文本类型，唯一约束，不能为空。这是原始文档的唯一标识符 (例如来自JSON的'name'字段)。
                #   - text: 文本类型，存储文档的文本内容，允许为空 (NULL)。
                #   - image_path: 文本类型，存储关联图像文件的路径，允许为空。
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS documents (
                        internal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        doc_id TEXT UNIQUE NOT NULL,
                        text TEXT,
                        image_path TEXT
                    )
                ''')
                
                # 可选：在 'doc_id' 列上创建一个索引。
                # 这可以加快通过原始 `doc_id` 查找记录的速度，例如在 `index_documents` 方法中检查重复文档时。
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON documents (doc_id)")
                
                conn.commit() # 提交事务，使表结构更改和索引创建生效。
                self.logger.info(f"数据库表 'documents' (及索引 'idx_doc_id') 初始化成功，或已存在。")
        except sqlite3.Error as e: # Catch specific SQLite errors
             self.logger.error(f"严重错误：初始化 SQLite 数据库 '{self.db_path}' 失败。错误详情: {e}", exc_info=True)
             raise RuntimeError(f"SQLite数据库操作失败: {e}") from e # Re-raise as a more generic runtime error
        except Exception as e_general:
            self.logger.error(f"初始化 SQLite 数据库 '{self.db_path}' 时发生未知错误。错误详情: {e_general}", exc_info=True)
            raise RuntimeError(f"SQLite数据库初始化未知错误: {e_general}") from e_general


    def _load_or_create_faiss_index(self, index_path: str, index_type_description: str) -> faiss.Index:
        """
        尝试从指定路径加载一个 Faiss 索引文件。
        - 如果文件存在且其内部存储的向量维度与当前编码器 (`self.encoder`) 的输出维度匹配，则加载该索引。
        - 如果文件不存在，或者文件存在但维度不匹配（表明该索引可能是用不同模型创建的），则创建一个新的、空的 Faiss 索引。
        - 使用 `faiss.IndexIDMap2` 类型的索引，它允许我们将自定义的 64 位整数 ID 与每个向量关联起来。
        此方法还会确保索引文件所在的目录存在。

        Args:
            index_path (str): Faiss 索引文件的期望路径。
            index_type_description (str): 索引类型的描述性名称 (例如 "文本", "图像", "平均")，主要用于日志记录。

        Returns:
            faiss.Index: 加载的或新创建的 Faiss 索引对象 (具体类型为 `faiss.IndexIDMap2`)。
        """
        self.logger.info(f"正在为 '{index_type_description}' 索引加载或创建 Faiss 文件于路径: '{index_path}'...")

        # 确保 Faiss 索引文件所在的目录存在，如果不存在则创建它。
        index_directory = os.path.dirname(index_path)
        if index_directory and not os.path.exists(index_directory): 
            try:
                os.makedirs(index_directory, exist_ok=True)
                self.logger.debug(f"已确保 '{index_type_description}' 索引的目录 '{index_directory}' 存在 (或已创建)。")
            except OSError as e:
                self.logger.error(f"创建Faiss索引目录 '{index_directory}' 失败: {e}", exc_info=True)
                raise # Re-raise as this is critical


        try:
            # 检查指定的索引文件是否已经存在于文件系统中。
            if os.path.exists(index_path) and os.path.isfile(index_path): # 确保是文件
                self.logger.info(f"发现已存在的 '{index_type_description}' Faiss 索引文件，尝试加载: {index_path}")
                # 使用 faiss.read_index 函数读取磁盘上的索引文件。
                index = faiss.read_index(index_path)
                self.logger.info(f"文件 '{index_path}' 读取成功，包含 {index.ntotal} 个向量，维度为 {index.d}。")

                # **重要**: 检查加载的索引的维度 (`index.d`) 是否与当前编码器模型产生的向量维度 (`self.vector_dimension`) 一致。
                if index.d != self.vector_dimension:
                    # 如果维度不匹配，这意味着已加载的索引是用不同的（或不同配置的）CLIP 模型创建的，因此不能直接使用。
                    self.logger.warning(f"维度不匹配警告! 加载的 '{index_type_description}' 索引维度 ({index.d}) 与当前编码器配置的维度 ({self.vector_dimension}) 不一致。")
                    self.logger.warning(f"这通常意味着之前的索引是用不同的模型创建的。将忽略已加载的旧索引，并创建一个新的空 '{index_type_description}' 索引。")
                    # 创建一个新的、空的 Faiss 索引来替换掉加载的不兼容的旧索引。
                    index = self._create_new_faiss_index(index_type_description)
                else:
                    # 维度匹配，加载成功。
                    self.logger.info(f"成功加载 '{index_type_description}' Faiss 索引，维度 ({index.d}) 与当前模型匹配。索引中包含 {index.ntotal} 个向量。")
            else:
                # 如果索引文件不存在。
                self.logger.info(f"未找到 '{index_type_description}' Faiss 索引文件: '{index_path}'。将创建一个新的空索引。")
                # 调用内部方法创建新的空索引。
                index = self._create_new_faiss_index(index_type_description)
        except Exception as e:
            # 处理在加载或读取索引文件过程中可能发生的任何其他错误。
            self.logger.error(f"错误：加载或处理 '{index_type_description}' Faiss 索引 '{index_path}' 失败。错误详情: {e}", exc_info=True)
            self.logger.info(f"作为安全回退机制，将创建一个新的空 '{index_type_description}' 索引。")
            # 即使加载失败，也创建一个新的空索引，以保证程序能够继续运行（尽管可能没有历史数据）。
            index = self._create_new_faiss_index(index_type_description)
        return index

    def _create_new_faiss_index(self, index_type_description: str) -> faiss.Index:
         """
         创建一个新的、空的 Faiss 索引。
         该索引被配置为使用内积 (`IndexFlatIP`) 进行相似度搜索，并使用 `IndexIDMap2` 包装器
         来支持为每个向量存储自定义的 64 位整数 ID。
         `IndexFlatIP` 适用于存储原始（未压缩）向量并进行精确的、暴力的内积搜索。
         对于已经 L2 归一化的向量，内积得分等价于余弦相似度。

         Args:
             index_type_description (str): 索引类型的描述 (例如 "文本", "图像")，用于日志记录。

         Returns:
             faiss.Index: 新创建的、空的 `faiss.IndexIDMap2` 索引对象。
         """
         self.logger.info(f"开始为 '{index_type_description}' 创建一个新的空 Faiss 索引...")
         # 步骤 1: 创建基础索引 (也称为 quantizer，在更复杂的索引类型中作用更明显)。
         # 这里使用 `faiss.IndexFlatIP`:
         #   - `IndexFlat`: 表示 Faiss 将存储完整的、未经压缩或量化的原始向量。这提供了最精确的搜索结果，但需要更多内存。
         #   - `IP` (Inner Product): 表示该索引将使用内积作为向量间的距离/相似度度量。
         #   当存储的向量都经过 L2 归一化时，它们之间的内积值等于它们之间的余弦相似度。
         #   `self.vector_dimension` 是从 CLIP 模型获取的特征向量的维度。
         quantizer = faiss.IndexFlatIP(self.vector_dimension)
         self.logger.debug(f"  为 '{index_type_description}' 创建了 IndexFlatIP 基础索引，维度: {self.vector_dimension}。")

         # 步骤 2: 创建 ID 映射包装器 `faiss.IndexIDMap2`。
         #   - `IndexIDMap2` 包装了一个基础索引 (此处是 `quantizer`)。
         #   - 它允许我们在向索引添加向量时，为每个向量指定一个我们自己定义的 64 位整数 ID。
         #   - 在搜索时，它会返回这些我们指定的 ID，而不是 Faiss 内部的连续行号。
         #   - '2' 在名称中通常表示它使用了更现代或更灵活的内部ID重映射机制。
         #   - 我们将使用从 SQLite 数据库生成的 `internal_id` 作为这个自定义 ID。
         index = faiss.IndexIDMap2(quantizer)
         self.logger.debug(f"  将 IndexFlatIP 包装在 IndexIDMap2 中，以支持自定义向量 ID。")

         self.logger.info(f"已成功为 '{index_type_description}' 创建一个新的、空的 Faiss 索引 (类型: IndexIDMap2 包裹 IndexFlatIP)。")
         self.logger.info(f"    索引维度: {self.vector_dimension}。")
         self.logger.info(f"    相似度度量: 内积 (Inner Product) - 对于归一化向量，这等同于余弦相似度。")
         return index

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        核心的文档索引流程。
        该方法接收一个文档列表，对每个文档进行多模态编码（文本和/或图像），
        然后将文档的元数据存储到 SQLite 数据库中，并将生成的特征向量（文本向量、图像向量、平均向量）
        及其对应的数据库内部ID (`internal_id`) 添加到各自的 Faiss 索引中。
        此方法会处理基于 `doc_id` 的重复文档（即，如果一个具有相同 `doc_id` 的文档已存在于数据库中，则跳过它）。
        为了提高效率，向量会分批收集，然后一次性批量添加到 Faiss 索引中。

        Args:
            documents (List[Dict[str, Any]]): 一个字典列表，其中每个字典代表一个待索引的文档。
                                    每个字典应至少包含 'id' (原始文档ID), 'text' (文本内容), 
                                    和 'image_path' (关联图像的路径，可能为None) 这几个键。
                                    这通常是 `load_data_from_json_and_associate_images` 函数的输出格式。
        """
        # 检查输入的文档列表是否为空。
        if not documents:
            self.logger.info("未提供任何文档进行索引操作。流程结束。")
            return

        self.logger.info(f"开始执行文档索引流程，准备处理 {len(documents)} 个文档...")
        
        # 初始化列表，用于批量收集需要添加到 Faiss 索引的向量和它们对应的 ID。
        # 分别为文本、图像和平均向量准备独立的批处理列表。
        text_vectors_batch: List[np.ndarray] = []   # 存储文本特征向量 (NumPy 数组)。
        text_ids_batch: List[int] = []              # 存储与文本向量对应的 `internal_id` (整数)。
        image_vectors_batch: List[np.ndarray] = []  # 存储图像特征向量。
        image_ids_batch: List[int] = []             # 存储与图像向量对应的 `internal_id`。
        mean_vectors_batch: List[np.ndarray] = []   # 存储平均（文本+图像）特征向量。
        mean_ids_batch: List[int] = []              # 存储与平均向量对应的 `internal_id`。

        # 初始化计数器，用于跟踪索引过程的各种统计数据。
        processed_count = 0          # 成功处理并至少尝试了编码的文档数量。
        skipped_duplicate_count = 0  # 因 `doc_id` 已存在于数据库中而被跳过的文档数量。
        encoding_failure_count = 0   # 因编码阶段（文本或图像）出错而未能为其生成向量的文档数量。
        db_insert_error_count = 0    # 因数据库插入操作出错而被跳过的文档数量。

        conn: Optional[sqlite3.Connection] = None # 初始化数据库连接变量，确保在 try...finally 块中可见。
        try:
            # 步骤 1: 建立与 SQLite 数据库的连接。
            self.logger.debug(f"正在连接到数据库: {self.db_path}")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor() # 获取数据库游标。
            # sqlite3 默认在执行 DML 语句 (如 INSERT) 时会自动开始一个事务。
            # 我们将在所有文档处理完毕后，在循环外部统一提交 (commit) 或回滚 (rollback) 事务，
            # 以确保数据库操作的原子性（相对于整个批次而言）。

            # 步骤 2: 遍历每个待索引的文档。
            self.logger.info(f"开始遍历 {len(documents)} 个文档进行处理和编码...")
            for i, doc_data in enumerate(documents): # 使用 enumerate 获取索引和文档数据。
                doc_id = doc_data.get('id')          # 获取原始文档 ID。
                text_content = doc_data.get('text')  # 获取文本内容。
                image_file_path = doc_data.get('image_path') # 获取图像路径。

                self.logger.debug(f"处理文档 {i+1}/{len(documents)}: ID='{doc_id}'")

                # 基本有效性验证：`doc_id` 必须存在。
                if not doc_id:
                    self.logger.warning(f"跳过列表中的第 {i+1} 条记录（原始索引 {i}），因其缺少 'id' 字段。记录: {doc_data}")
                    continue # 跳到下一个文档。
                # 至少需要文本或图像路径之一才能进行有意义的编码。
                if not text_content and not image_file_path:
                     self.logger.warning(f"跳过文档 ID '{doc_id}'，因为它既没有文本内容，也没有关联的图像路径。无法为其生成任何向量。")
                     continue # 跳到下一个文档。

                # --- 2a. 检查文档是否已在数据库中存在 (基于 `doc_id`) ---
                try:
                    cursor.execute("SELECT internal_id FROM documents WHERE doc_id = ?", (str(doc_id),)) # Ensure doc_id is string
                    existing_record = cursor.fetchone() # 获取查询结果（如果存在的话）。
                    if existing_record:
                         self.logger.debug(f"文档 ID '{doc_id}' 已存在于数据库中 (其 internal_id 为: {existing_record[0]})。将跳过此重复文档的索引。")
                         skipped_duplicate_count += 1
                         continue # 跳到下一个文档。
                except sqlite3.Error as e_check:
                    self.logger.error(f"检查文档 ID '{doc_id}' 是否存在时发生数据库错误: {e_check}。将跳过此文档。")
                    db_insert_error_count +=1 # 计入数据库错误
                    continue


                # --- 2b. 将文档元数据插入到数据库，并获取生成的 `internal_id` ---
                internal_id: Optional[int] = None # 初始化 internal_id。
                try:
                    # 执行 INSERT 语句将新文档的元数据插入到 'documents' 表。
                    # 使用参数化查询 (问号占位符) 来防止 SQL 注入攻击。
                    cursor.execute(
                        "INSERT INTO documents (doc_id, text, image_path) VALUES (?, ?, ?)",
                        (str(doc_id), text_content, image_file_path) # Ensure doc_id is string
                    )
                    # 获取刚刚插入行的自增主键 (`internal_id`)。
                    # `cursor.lastrowid` 返回最后插入行的 ROWID。
                    internal_id = cursor.lastrowid
                    if internal_id is None: 
                        self.logger.error(f"严重数据库错误：为文档 '{doc_id}' 插入元数据后，未能获取有效的 internal_id (lastrowid is None)。")
                        raise sqlite3.Error(f"数据库错误：未能为文档 '{doc_id}' 获取 internal_id。")
                    self.logger.debug(f"文档 '{doc_id}' 的元数据已成功插入数据库，获得的 internal_id: {internal_id}")

                except sqlite3.IntegrityError:
                    # 当尝试插入的 `doc_id` 违反了表的 UNIQUE 约束时（理论上不应发生，因为前面已检查过）。
                    self.logger.error(f"数据库完整性错误：尝试插入已存在的文档 ID '{doc_id}'（可能是并发问题或检查逻辑遗漏）。将跳过此文档。")
                    skipped_duplicate_count += 1 
                    continue 
                except sqlite3.Error as db_e:
                    self.logger.error(f"数据库错误：在为文档 '{doc_id}' 插入元数据时发生错误: {db_e}。将跳过此文档的索引。")
                    db_insert_error_count += 1
                    continue 

                # --- 2c. 使用内部 Encoder 对文档进行多模态向量化 ---
                encoded_data: Optional[Dict[str, Optional[np.ndarray]]] = None # 初始化编码结果。
                if internal_id is not None: # 只有成功获取 internal_id 后才进行编码
                    try:
                        self.logger.debug(f"开始为文档 '{doc_id}' (internal_id: {internal_id}) 进行多模态编码...")
                        encoded_data = self.encoder.encode(text=text_content, image_path=image_file_path)
                    except Exception as encode_e:
                        self.logger.error(f"严重错误：在编码文档 '{doc_id}' (internal_id: {internal_id}) 时发生意外错误: {encode_e}", exc_info=True)
                        self.logger.warning(f"注意：此文档的元数据可能已存入数据库，但其向量将不会被添加到 Faiss 索引中，因为它编码失败。")
                        encoding_failure_count += 1
                        continue 

                # --- 2d. 将成功编码的向量添加到对应的批处理列表中 ---
                at_least_one_vector_added_for_doc = False
                if encoded_data and internal_id is not None: 
                    if encoded_data.get('text_vector') is not None:
                        text_vectors_batch.append(encoded_data['text_vector']) # type: ignore
                        text_ids_batch.append(internal_id)
                        at_least_one_vector_added_for_doc = True
                        self.logger.debug(f"  文本向量已为文档 '{doc_id}' (internal_id: {internal_id}) 准备好加入批处理。")
                    
                    if encoded_data.get('image_vector') is not None:
                        image_vectors_batch.append(encoded_data['image_vector']) # type: ignore
                        image_ids_batch.append(internal_id)
                        at_least_one_vector_added_for_doc = True
                        self.logger.debug(f"  图像向量已为文档 '{doc_id}' (internal_id: {internal_id}) 准备好加入批处理。")

                    if encoded_data.get('mean_vector') is not None:
                        mean_vectors_batch.append(encoded_data['mean_vector']) # type: ignore
                        mean_ids_batch.append(internal_id)
                        at_least_one_vector_added_for_doc = True
                        self.logger.debug(f"  平均向量已为文档 '{doc_id}' (internal_id: {internal_id}) 准备好加入批处理。")
                    
                    if not at_least_one_vector_added_for_doc and (text_content or image_file_path):
                        self.logger.warning(f"文档 '{doc_id}' (internal_id: {internal_id}) 有内容，但编码后未生成任何有效向量。")
                        encoding_failure_count += 1
                    elif at_least_one_vector_added_for_doc:
                        processed_count += 1 
                
                elif internal_id is not None and (text_content or image_file_path): 
                    self.logger.warning(f"文档 '{doc_id}' (internal_id: {internal_id}) 编码返回 None，尽管有内容。计为编码失败。")
                    encoding_failure_count += 1


            # --- 文档遍历和初步处理完成 ---
            self.logger.info(f"所有 {len(documents)} 个输入文档已遍历处理完毕。")
            self.logger.info(f"准备将收集到的向量批量添加到 Faiss 索引中...")
            self.logger.info(f"  - 待添加文本向量数量: {len(text_ids_batch)}")
            self.logger.info(f"  - 待添加图像向量数量: {len(image_ids_batch)}")
            self.logger.info(f"  - 待添加平均向量数量: {len(mean_ids_batch)}")

            # --- 步骤 3: 批量将向量和 ID 添加到对应的 Faiss 索引 ---
            if text_vectors_batch: 
                ids_np_text = np.array(text_ids_batch, dtype='int64')
                vectors_np_text = np.array(text_vectors_batch, dtype='float32')
                self.text_index.add_with_ids(vectors_np_text, ids_np_text) 
                self.logger.info(f"已成功向文本(Text) Faiss 索引批量添加 {len(text_vectors_batch)} 个向量。当前索引总数: {self.text_index.ntotal}")

            if image_vectors_batch:
                ids_np_image = np.array(image_ids_batch, dtype='int64')
                vectors_np_image = np.array(image_vectors_batch, dtype='float32')
                self.image_index.add_with_ids(vectors_np_image, ids_np_image)
                self.logger.info(f"已成功向图像(Image) Faiss 索引批量添加 {len(image_vectors_batch)} 个向量。当前索引总数: {self.image_index.ntotal}")

            if mean_vectors_batch:
                ids_np_mean = np.array(mean_ids_batch, dtype='int64')
                vectors_np_mean = np.array(mean_vectors_batch, dtype='float32')
                self.mean_index.add_with_ids(vectors_np_mean, ids_np_mean)
                self.logger.info(f"已成功向平均(Mean) Faiss 索引批量添加 {len(mean_vectors_batch)} 个向量。当前索引总数: {self.mean_index.ntotal}")

            # --- 步骤 4: 提交数据库事务 ---
            if conn: 
                conn.commit()
                self.logger.info("数据库事务已成功提交。所有元数据更改已持久化。")

        except Exception as e:
            self.logger.critical(f"严重错误：在文档索引过程中发生意外的顶级异常: {e}", exc_info=True)
            if conn:
                self.logger.info("检测到严重错误，正在尝试回滚数据库事务以撤销未提交的更改...")
                try:
                    conn.rollback()
                    self.logger.info("数据库事务已成功回滚。")
                except Exception as rb_e:
                    self.logger.error(f"错误：尝试回滚数据库事务时失败: {rb_e}", exc_info=True)
        finally:
            if conn:
                conn.close()
                self.logger.debug("数据库连接已关闭。")

        # --- 打印索引过程的最终总结信息 ---
        self.logger.info(f"\n--- 文档索引过程总结 ---")
        self.logger.info(f"- 输入文档总数: {len(documents)}")
        self.logger.info(f"- 成功处理并为其生成了至少一个向量的文档数: {processed_count}")
        self.logger.info(f"- 因 'doc_id' 在数据库中已存在而跳过的文档数: {skipped_duplicate_count}")
        self.logger.info(f"- 因编码阶段（文本/图像）错误而未能生成向量的文档数: {encoding_failure_count}")
        self.logger.info(f"- 因数据库操作（如插入元数据）错误而跳过的文档数: {db_insert_error_count}")
        self.logger.info(f"- 当前文本 Faiss 索引中的向量总数: {getattr(self.text_index, 'ntotal', 'N/A')}")
        self.logger.info(f"- 当前图像 Faiss 索引中的向量总数: {getattr(self.image_index, 'ntotal', 'N/A')}")
        self.logger.info(f"- 当前平均 Faiss 索引中的向量总数: {getattr(self.mean_index, 'ntotal', 'N/A')}")
        
        db_final_count = self.get_document_count() 
        self.logger.info(f"- 当前 SQLite 数据库中存储的文档元数据记录总数: {db_final_count}")
        
        max_faiss_vectors = 0
        if hasattr(self.text_index, 'ntotal'): max_faiss_vectors = max(max_faiss_vectors, self.text_index.ntotal)
        if hasattr(self.image_index, 'ntotal'): max_faiss_vectors = max(max_faiss_vectors, self.image_index.ntotal)
        if hasattr(self.mean_index, 'ntotal'): max_faiss_vectors = max(max_faiss_vectors, self.mean_index.ntotal)

        if db_final_count < max_faiss_vectors:
             self.logger.warning(f"数据一致性警告：数据库记录数 ({db_final_count}) 少于某个 Faiss 索引中的最大向量数 ({max_faiss_vectors})。数据可能存在不一致！请检查日志。")
        self.logger.info(f"--- 文档索引过程结束 ---")


    def get_document_by_internal_id(self, internal_id: int) -> Optional[Dict[str, Any]]:
        """
        根据 Faiss 搜索返回的 `internal_id` (即数据库中的主键)，从 SQLite 数据库中检索对应的原始文档元数据。

        Args:
            internal_id (int): 要查询的文档在数据库中的 `internal_id` (通常由 Faiss 搜索返回)。

        Returns:
            Optional[Dict[str, Any]]: 如果找到文档，则返回一个包含文档信息的字典。
                            该字典通常包含 'id' (原始 doc_id), 'text', 'image_path', 和 'internal_id'。
                            如果数据库中找不到具有该 `internal_id` 的记录，则返回 None。
        """
        self.logger.debug(f"尝试从数据库根据 internal_id '{internal_id}' 获取文档元数据...")
        try:
            # 连接到 SQLite 数据库。
            with sqlite3.connect(self.db_path) as conn:
                # 设置 conn.row_factory = sqlite3.Row 使得查询结果可以像字典一样通过列名访问，更方便。
                conn.row_factory = sqlite3.Row 
                cursor = conn.cursor()
                # 执行 SELECT 查询，根据 internal_id 查找记录。
                cursor.execute(
                    "SELECT internal_id, doc_id, text, image_path FROM documents WHERE internal_id = ?",
                    (internal_id,) # 注意参数是一个元组。
                )
                row = cursor.fetchone() # 获取一行结果。
                
                if row:
                    # 如果找到了记录 (row 不为 None)。
                    doc_data = dict(row) # 将 sqlite3.Row 对象转换为标准的 Python 字典。
                    # 为了与外部接口或数据结构保持一致（例如，原始输入时的 'id'），
                    # 将从数据库中取出的 'doc_id' 键重命名为 'id'。
                    doc_data['id'] = doc_data.pop('doc_id')
                    self.logger.debug(f"成功为 internal_id '{internal_id}' 找到文档元数据: ID='{doc_data['id']}'")
                    return doc_data # 返回包含文档信息的字典。
                else:
                    # 如果没有找到具有该 internal_id 的记录。
                    self.logger.warning(f"未能在数据库中找到 internal_id 为 '{internal_id}' 的文档元数据。")
                    return None # 返回 None。
        except sqlite3.Error as e_sql:
             self.logger.error(f"数据库错误：从数据库根据 internal_id '{internal_id}' 获取文档时发生错误: {e_sql}", exc_info=True)
             return None 
        except Exception as e_general:
             self.logger.error(f"未知错误：从数据库根据 internal_id '{internal_id}' 获取文档时发生: {e_general}", exc_info=True)
             return None 

    def get_documents_by_internal_ids(self, internal_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        根据一个 `internal_id` 的列表，从 SQLite 数据库中批量检索对应的多个文档的元数据。
        使用批量查询 (SELECT ... WHERE internal_id IN (...)) 通常比多次单独查询更高效，
        尤其是在处理 Faiss 返回的 Top-K 结果列表时。

        Args:
            internal_ids (List[int]): 一个包含多个数据库 `internal_id` 的整数列表。

        Returns:
            Dict[int, Dict[str, Any]]: 一个字典，其中键是 `internal_id`，值是对应的文档数据字典
                             (通常包含 'id', 'text', 'image_path', 'internal_id')。
                             如果列表中的某个 ID 在数据库中找不到，则结果字典中不会包含该 ID 的条目。
                             如果输入的 `internal_ids` 列表为空，则返回一个空字典。
        """
        # 如果输入的 ID 列表为空，直接返回空字典，无需查询数据库。
        if not internal_ids:
            self.logger.debug("请求批量获取文档，但提供的 internal_id 列表为空。返回空结果。")
            return {}

        self.logger.debug(f"尝试从数据库根据 internal_id 列表 (共 {len(internal_ids)} 个) 批量获取文档元数据...")
        results: Dict[int, Dict[str, Any]] = {} # 初始化结果字典。
        try:
            # 连接到 SQLite 数据库。
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row # 设置行工厂，方便处理查询结果。
                cursor = conn.cursor()

                # 构建 SQL 查询语句，使用 IN 操作符和参数占位符进行批量查询。
                # 1. 创建占位符字符串: "(?, ?, ..., ?)" - 每个 ID 对应一个 '?'。
                placeholders = ','.join('?' for _ in internal_ids)
                # 2. 构建完整的 SQL 查询语句。
                query = f"SELECT internal_id, doc_id, text, image_path FROM documents WHERE internal_id IN ({placeholders})"
                self.logger.debug(f"执行批量查询SQL: {query} (参数数量: {len(internal_ids)})")

                # 执行查询，将 ID 列表作为参数传递给 execute 方法。
                cursor.execute(query, internal_ids)
                rows = cursor.fetchall() # 获取所有匹配的行。
                self.logger.debug(f"数据库批量查询返回了 {len(rows)} 行记录。")

                # 遍历查询结果。
                for row in rows:
                    doc_data = dict(row) # 将 sqlite3.Row 转换为字典。
                    doc_data['id'] = doc_data.pop('doc_id') # 重命名 'doc_id' 为 'id'。
                    # 使用 internal_id 作为键，将文档数据存入结果字典。
                    results[doc_data['internal_id']] = doc_data
                
                # 检查是否有ID未找到 (如果 len(results) < len(internal_ids))
                if len(results) < len(internal_ids):
                    found_ids_set = set(results.keys()) # More efficient for checking
                    missing_ids = [id_val for id_val in internal_ids if id_val not in found_ids_set]
                    if missing_ids:
                         self.logger.warning(f"在批量获取文档时，以下 internal_id 未在数据库中找到: {missing_ids}")

        except sqlite3.Error as e_sql:
             self.logger.error(f"数据库错误：从数据库根据 internal_id 列表批量获取文档时发生: {e_sql}", exc_info=True)
        except Exception as e_general:
            self.logger.error(f"未知错误：从数据库根据 internal_id 列表批量获取文档时发生: {e_general}", exc_info=True)
        
        self.logger.debug(f"批量获取文档元数据完成，共返回 {len(results)} 个文档的信息。")
        return results

    def get_document_count(self) -> int:
         """
         获取当前 SQLite 数据库 'documents' 表中存储的文档总数量。

         Returns:
             int: 数据库中 'documents' 表的总行数。如果发生错误，则返回 0。
         """
         self.logger.debug(f"开始从数据库 '{self.db_path}' 获取文档总数...")
         try:
             # 连接到 SQLite 数据库。
             with sqlite3.connect(self.db_path) as conn:
                 cursor = conn.cursor()
                 # 执行 COUNT(*) 查询获取总行数。
                 cursor.execute("SELECT COUNT(*) FROM documents")
                 # fetchone() 返回一个包含单个值的元组，例如 (50,)。
                 count_result = cursor.fetchone()
                 # 提取元组中的计数值。如果查询无结果（理论上 COUNT(*) 总有结果，但做个健壮性检查），则默认为 0。
                 count = count_result[0] if count_result and count_result[0] is not None else 0
                 self.logger.debug(f"数据库中文档总数为: {count}")
                 return count
         except sqlite3.Error as e_sql:
              self.logger.error(f"数据库错误：从数据库获取文档总数时发生错误: {e_sql}", exc_info=True)
              return 0 
         except Exception as e_general:
              self.logger.error(f"未知错误：从数据库获取文档总数时发生: {e_general}", exc_info=True)
              return 0 

    def save_indices(self):
        """
        将内存中的所有三个 Faiss 索引（文本、图像、平均）分别保存到它们对应的文件路径中。
        这个方法用于持久化索引的状态，以便在下次程序启动时可以加载这些索引，从而避免重新处理和编码所有文档。
        只有当索引非空（即包含至少一个向量）时，才会执行保存操作。
        """
        self.logger.info("开始尝试将所有 Faiss 索引保存到磁盘文件...")
        # 调用内部辅助方法 `_save_single_index` 分别保存每个索引。
        # 传递索引对象、目标文件路径和索引类型描述（用于日志）。
        if hasattr(self, 'text_index'):
             self._save_single_index(self.text_index, self.faiss_text_index_path, "文本(Text)")
        else:
            self.logger.warning("文本(Text)索引对象不存在，无法保存。")

        if hasattr(self, 'image_index'):
            self._save_single_index(self.image_index, self.faiss_image_index_path, "图像(Image)")
        else:
            self.logger.warning("图像(Image)索引对象不存在，无法保存。")
        
        if hasattr(self, 'mean_index'):
            self._save_single_index(self.mean_index, self.faiss_mean_index_path, "平均(Mean)")
        else:
            self.logger.warning("平均(Mean)索引对象不存在，无法保存。")

        self.logger.info("所有 Faiss 索引的保存操作已完成（或已跳过空索引/不存在的索引）。")

    def _save_single_index(self, index: Optional[faiss.Index], index_path: str, index_type_description: str):
        """
        辅助方法：保存单个 Faiss 索引到指定的文件路径。
        仅当索引对象有效且包含至少一个向量时才执行保存。
        此方法还会确保索引文件要保存到的目录存在。

        Args:
            index (Optional[faiss.Index]): 需要保存的 Faiss 索引对象。可能是 None。
            index_path (str): 保存索引的目标文件完整路径。
            index_type_description (str): 索引类型的描述性名称 (例如 "文本", "图像")，用于日志记录。
        """
        self.logger.debug(f"准备保存 '{index_type_description}' Faiss 索引到路径: '{index_path}'...")
        
        if index is None:
            self.logger.warning(f"  警告：'{index_type_description}' Faiss 索引对象为 None，无法执行保存操作。")
            return

        # 检查索引对象是否有效（存在 `ntotal` 属性，表示向量数量）以及向量数量是否大于 0。
        if hasattr(index, 'ntotal') and index.ntotal > 0:
            try:
                # 确保索引文件要保存到的目录存在，如果不存在则创建它。
                index_directory = os.path.dirname(index_path)
                if index_directory and not os.path.exists(index_directory): 
                    os.makedirs(index_directory, exist_ok=True)
                    self.logger.debug(f"  已确保 '{index_type_description}' 索引的保存目录 '{index_directory}' 存在 (或已创建)。")

                # 使用 faiss.write_index 函数将内存中的索引对象写入到指定的磁盘文件。
                faiss.write_index(index, index_path)
                self.logger.info(f"  成功：'{index_type_description}' Faiss 索引 (包含 {index.ntotal} 个向量) 已保存到: {index_path}")
            except Exception as e:
                # 处理在保存索引过程中可能发生的错误 (例如，磁盘空间不足、文件写入权限问题)。
                self.logger.error(f"  错误：保存 '{index_type_description}' Faiss 索引到 '{index_path}' 失败。错误详情: {e}", exc_info=True)
        elif hasattr(index, 'ntotal'): # 索引存在但为空 (ntotal == 0)
             self.logger.info(f"  跳过：'{index_type_description}' Faiss 索引为空 (ntotal={index.ntotal})，因此不保存到 '{index_path}'。")
        else: # 索引对象无效或未正确初始化 
             self.logger.warning(f"  警告：'{index_type_description}' Faiss 索引似乎未正确初始化 (缺少 ntotal 属性)，无法执行保存操作。")

    def close(self):
        """
        关闭 Indexer 实例时调用的清理方法。
        主要职责是确保所有内存中的 Faiss 索引都已尝试保存到磁盘。
        SQLite 数据库连接是通过 `with sqlite3.connect(...)` 语句管理的，在每个相关方法结束时会自动关闭，
        因此这里不需要显式关闭数据库连接。
        Faiss 索引对象本身在 Python 中是内存对象，它们不需要像文件句柄那样显式关闭；保存它们的状态即是“关闭”操作。
        """
        self.logger.info("开始关闭 Indexer 实例...")
        # 调用 save_indices 方法，确保存储所有 Faiss 索引的最新状态。
        self.save_indices()
        self.logger.info("Indexer 实例关闭完成。所有 Faiss 索引已尝试保存。")

# -------------------------------------------------------------------------------------------------
# 检索器类 (Retriever)
# -------------------------------------------------------------------------------------------------
class Retriever:
    """
    Retriever 类负责处理用户的查询（可以是文本、图像路径或两者结合的多模态查询），
    并从已建立的索引中检索最相关的文档。其工作流程如下：

    1.  **接收查询**: 用户通过 `retrieve` 方法提交查询。
    2.  **查询编码**: 利用与 `Indexer` 中相同的 `MultimodalEncoder` 实例对用户查询进行向量化，
        将其转换为与索引文档相同向量空间中的特征向量（可能包括文本向量、图像向量和/或平均向量）。
    3.  **选择策略**: 根据查询的类型（纯文本、纯图像、多模态）和可用性，
        选择最合适的 Faiss 索引（文本索引、图像索引或平均向量索引）以及对应的查询向量进行搜索。
        例如，纯文本查询将使用文本向量在文本索引中搜索。
    4.  **相似度搜索**: 在选定的 Faiss 索引中执行 Top-K 相似度搜索，找出与查询向量最相似的 K 个向量。
        搜索结果是这些向量的 `internal_id` (与数据库主键对应) 和它们与查询向量的相似度得分。
    5.  **获取元数据**: 使用检索到的 `internal_id` 列表，通过 `Indexer` 的接口从 SQLite 数据库中批量获取
        这些最相关文档的完整元数据（如原始ID、文本内容、图像路径等）。
    6.  **结果组合与返回**: 将获取到的文档元数据与它们各自的相似度得分结合起来，
        并按照相似度得分从高到低（表示最相关）排序，最终返回一个包含这些信息的文档列表。

    Retriever 依赖于一个已经初始化并填充了数据和索引的 `Indexer` 实例。
    它复用 `Indexer` 的编码器以保证查询和文档编码的一致性，并访问 `Indexer` 中的 Faiss 索引和数据库。
    """
    def __init__(self, indexer: Indexer):
        """
        初始化 Retriever 实例。

        Args:
            indexer (Indexer): 一个已经初始化并包含了数据和索引的 `Indexer` 类的实例。
                               Retriever 的所有操作都依赖于这个 `Indexer` 实例提供的资源。

        Raises:
            ValueError: 如果传入的 `indexer` 不是 `Indexer` 类的有效实例，或者该实例似乎缺少
                        必要的 Faiss 索引属性 (text_index, image_index, mean_index)，则抛出此异常。
                        一个没有有效索引源的 Retriever 是无法工作的。
        """
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.info("开始初始化 Retriever...")
        
        # 验证传入的 indexer 参数的有效性。
        if not isinstance(indexer, Indexer):
             msg = "Retriever 初始化错误: 需要一个有效的 Indexer 实例，但收到的不是。"
             self.logger.error(msg)
             raise ValueError(msg)
        
        # 进一步验证 Indexer 实例是否已成功创建了所需的 Faiss 索引对象。
        required_indices_attributes = ['text_index', 'image_index', 'mean_index', 'encoder', 'vector_dimension']
        missing_attrs = [attr for attr in required_indices_attributes if not hasattr(indexer, attr) or getattr(indexer, attr) is None]
        if missing_attrs:
            msg = f"Retriever 初始化错误: 提供的 Indexer 实例缺少以下必需的属性: {', '.join(missing_attrs)}。请确保 Indexer 已成功初始化。"
            self.logger.error(msg)
            raise ValueError(msg)

        # 保存对传入的 Indexer 实例的引用。
        self.indexer: Indexer = indexer
        # 复用 Indexer 内部的 Encoder 实例。
        self.encoder: MultimodalEncoder = self.indexer.encoder
        # 从 Indexer 获取向量维度。
        self.vector_dimension: int = self.indexer.vector_dimension
        self.logger.info(f"  Retriever 将使用 Indexer 的编码器 (向量维度: {self.vector_dimension})。")

        # 获取对 Indexer 中三个 Faiss 索引的直接引用。
        self.text_index: faiss.Index = self.indexer.text_index
        self.image_index: faiss.Index = self.indexer.image_index
        self.mean_index: faiss.Index = self.indexer.mean_index

        # 检查所有关联的 Faiss 索引是否都为空。
        text_index_ntotal = getattr(self.text_index, 'ntotal', 0) 
        image_index_ntotal = getattr(self.image_index, 'ntotal', 0)
        mean_index_ntotal = getattr(self.mean_index, 'ntotal', 0)
        
        if text_index_ntotal == 0 and image_index_ntotal == 0 and mean_index_ntotal == 0:
             self.logger.warning("Retriever 初始化警告: Indexer 中的所有 Faiss 索引当前都为空。")
             self.logger.warning("  这意味着任何检索操作都将无法找到任何匹配的文档。")
             self.logger.warning("  请确保 Indexer 已成功索引了数据，或者检查索引建立过程的日志。")
        else:
             self.logger.info(f"Retriever 初始化成功。关联的 Indexer 状态如下:")
             self.logger.info(f"    - 文本(Text)索引中向量数: {text_index_ntotal}")
             self.logger.info(f"    - 图像(Image)索引中向量数: {image_index_ntotal}")
             self.logger.info(f"    - 平均(Mean)索引中向量数: {mean_index_ntotal}")
        self.logger.info("Retriever 初始化完成。")


    def retrieve(self, query: Union[str, Dict[str, str]], k: int = 5) -> List[Dict[str, Any]]:
        """
        执行完整的检索流程：接收用户查询 -> 对查询进行编码 -> 根据查询类型选择合适的索引和查询向量 ->
        在选定的 Faiss 索引中搜索相似向量 -> 获取这些向量对应的原始文档元数据 -> 组合信息并返回结果。

        Args:
            query (Union[str, Dict[str, str]]): 用户提交的查询。可以是：
                - str: 一个纯文本查询字符串。
                - Dict: 一个字典，用于表示更复杂的查询类型：
                    - {'text': '文本内容', 'image_path': '图像路径'} : 多模态查询，结合文本和图像信息。
                    - {'image_path': '图像路径'} : 纯图像查询，仅使用图像内容进行检索。
                    - {'text': '文本内容'} : 纯文本查询 (与直接传入字符串的效果相同，但通过字典形式提供)。
                    字典中必须至少包含 'text' 或 'image_path' 键及其对应值，且值不能为空字符串。
            k (int): 指定希望检索的最相似文档的数量 (Top-K)。默认为 5。

        Returns:
            List[Dict[str, Any]]: 一个按相似度得分降序排列的文档列表。
                        列表中的每个字典代表一个检索到的文档，包含以下键（但不限于）：
                        - 'id': 原始文档 ID (str)。
                        - 'text': 文档的文本内容 (str 或 None)。
                        - 'image_path': 关联图像的路径 (str 或 None)。
                        - 'internal_id': 数据库和 Faiss 使用的内部 ID (int)。
                        - 'score': 该文档与查询的相似度得分 (float)。对于内积搜索，得分越高表示越相似。
                        如果查询无效、编码失败、所选索引为空或搜索无结果，则返回空列表 `[]`。
        """
        self.logger.info(f"开始执行检索流程，目标是获取 Top-{k} 最相关的文档...")
        self.logger.debug(f"  接收到的原始查询: {str(query)[:200]}{'...' if len(str(query))>200 else ''}, k={k}")

        query_text: Optional[str] = None        
        query_image_path: Optional[str] = None  
        query_type: str = "unknown"             

        # --- 步骤 1: 解析查询输入，确定查询类型和具体内容 ---
        self.logger.debug("  - Retriever步骤 1: 解析用户查询输入...")
        if isinstance(query, str): 
            query_text_stripped = query.strip()
            if query_text_stripped:
                query_text = query_text_stripped
                query_type = "纯文本"
                self.logger.info(f"    查询类型确定为: {query_type} (字符串输入)")
                self.logger.info(f"    查询文本内容: '{query_text[:100]}{'...' if len(query_text)>100 else ''}'")
            else:
                self.logger.error("查询错误: 纯文本查询字符串为空或只包含空白。")
                return []
        elif isinstance(query, dict): 
            query_text_from_dict = query.get('text')
            query_image_path_from_dict = query.get('image_path')

            query_text = query_text_from_dict.strip() if isinstance(query_text_from_dict, str) and query_text_from_dict.strip() else None
            query_image_path = query_image_path_from_dict.strip() if isinstance(query_image_path_from_dict, str) and query_image_path_from_dict.strip() else None
            
            if query_text and query_image_path: 
                query_type = "多模态"
                self.logger.info(f"    查询类型确定为: {query_type}")
                self.logger.info(f"    查询文本部分: '{query_text[:50]}{'...' if len(query_text)>50 else ''}'")
                self.logger.info(f"    查询图像部分: '{os.path.basename(query_image_path)}'")
            elif query_image_path: 
                query_type = "纯图像"
                self.logger.info(f"    查询类型确定为: {query_type}")
                self.logger.info(f"    查询图像路径: '{os.path.basename(query_image_path)}'")
            elif query_text: 
                query_type = "纯文本"
                self.logger.info(f"    查询类型确定为: {query_type} (字典输入)")
                self.logger.info(f"    查询文本内容: '{query_text[:100]}{'...' if len(query_text)>100 else ''}'")
            else: 
                self.logger.error("查询错误: 查询字典无效，必须至少包含有效的 'text' 或 'image_path' 键及其对应值。")
                return [] 
        else: 
            self.logger.error(f"查询错误: 不支持的查询类型 ({type(query)}) 或查询内容为空。查询必须是有效的非空字符串或包含有效内容的字典。")
            return [] 

        # --- 步骤 2: 使用内部的 MultimodalEncoder 对查询进行编码 ---
        self.logger.debug(f"  - Retriever步骤 2: 使用 MultimodalEncoder 对 '{query_type}' 查询进行编码...")
        try:
            encoded_query_vectors = self.encoder.encode(text=query_text, image_path=query_image_path)
            query_text_vec = encoded_query_vectors.get('text_vector')
            query_image_vec = encoded_query_vectors.get('image_vector')
            query_mean_vec = encoded_query_vectors.get('mean_vector')
            
            if query_text_vec is None and query_image_vec is None and query_mean_vec is None:
                 self.logger.warning("查询编码警告: MultimodalEncoder 未能为当前查询生成任何有效的特征向量。无法继续检索。")
                 return [] 
            self.logger.info("    查询编码完成。")
            if query_text_vec is not None: self.logger.debug("      - 生成了文本查询向量。")
            if query_image_vec is not None: self.logger.debug("      - 生成了图像查询向量。")
            if query_mean_vec is not None: self.logger.debug("      - 生成了平均查询向量。")

        except Exception as e:
            self.logger.error(f"查询编码严重错误: 在对查询进行编码时发生意外错误: {e}", exc_info=True)
            return [] 

        # --- 步骤 3: 根据查询类型选择目标 Faiss 索引和相应的查询向量 ---
        self.logger.debug(f"  - Retriever步骤 3: 根据查询类型 '{query_type}' 选择搜索策略 (Faiss索引和查询向量)...")
        target_faiss_index: Optional[faiss.Index] = None   
        search_query_vector: Optional[np.ndarray] = None   
        selected_index_name: str = "N/A"                   

        text_index_ntotal = getattr(self.text_index, 'ntotal', 0)
        image_index_ntotal = getattr(self.image_index, 'ntotal', 0)
        mean_index_ntotal = getattr(self.mean_index, 'ntotal', 0)

        if query_type == "纯文本":
            if query_text_vec is not None and text_index_ntotal > 0:
                target_faiss_index = self.text_index
                search_query_vector = query_text_vec
                selected_index_name = "文本(Text)索引"
                self.logger.info(f"    搜索策略: 使用文本查询向量，在 {selected_index_name} (含 {text_index_ntotal} 个向量) 中搜索。")
            else:
                 reason = "文本查询向量编码失败" if query_text_vec is None else f"文本(Text) Faiss 索引为空 (仅含 {text_index_ntotal} 个向量)"
                 self.logger.warning(f"无法执行纯文本查询，因为: {reason}。")
                 return []
        elif query_type == "纯图像":
            if query_image_vec is not None and image_index_ntotal > 0:
                target_faiss_index = self.image_index
                search_query_vector = query_image_vec
                selected_index_name = "图像(Image)索引"
                self.logger.info(f"    搜索策略: 使用图像查询向量，在 {selected_index_name} (含 {image_index_ntotal} 个向量) 中搜索。")
            else:
                reason = "图像查询向量编码失败" if query_image_vec is None else f"图像(Image) Faiss 索引为空 (仅含 {image_index_ntotal} 个向量)"
                self.logger.warning(f"无法执行纯图像查询，因为: {reason}。")
                return []
        elif query_type == "多模态":
            if query_mean_vec is not None and mean_index_ntotal > 0:
                target_faiss_index = self.mean_index
                search_query_vector = query_mean_vec
                selected_index_name = "平均(Mean)索引"
                self.logger.info(f"    搜索策略: 使用平均查询向量，在 {selected_index_name} (含 {mean_index_ntotal} 个向量) 中搜索。")
            elif query_text_vec is not None and text_index_ntotal > 0:
                 self.logger.warning("多模态查询警告: 平均(Mean)索引或平均查询向量不可用/索引为空。")
                 self.logger.info(f"    应用回退策略: 改为使用文本查询向量，在文本(Text)索引 (含 {text_index_ntotal} 个向量) 中搜索。")
                 target_faiss_index = self.text_index
                 search_query_vector = query_text_vec
                 selected_index_name = "文本(Text)索引 (作为多模态查询的回退)"
            else:
                reason_parts = []
                if query_mean_vec is None and query_text_vec is None: reason_parts.append("平均查询向量和文本查询向量都编码失败")
                if mean_index_ntotal == 0 and text_index_ntotal == 0: reason_parts.append(f"平均(Mean)索引(含{mean_index_ntotal}向量)和文本(Text)索引(含{text_index_ntotal}向量)都为空")
                if not reason_parts: reason_parts.append("由于平均(Mean)索引和文本(Text)索引（用于回退）都不可用，或对应的查询向量缺失")
                final_reason = "; ".join(reason_parts)
                self.logger.warning(f"无法执行多模态查询，因为: {final_reason}。")
                return []
        else: 
             self.logger.error("内部逻辑错误: 无法为当前查询确定有效的查询类型或找不到可用的查询向量/索引组合。")
             return []
        
        if target_faiss_index is None or search_query_vector is None:
            self.logger.error("内部错误: 搜索目标 Faiss 索引或查询向量未能正确设置，尽管已尝试选择策略。无法继续搜索。")
            return []


        # --- 步骤 4: 在选定的 Faiss 索引中执行 Top-K 相似度搜索 ---
        self.logger.debug(f"  - Retriever步骤 4: 在选定的 '{selected_index_name}' 中执行 Faiss Top-{k} 搜索...")
        try:
            query_vector_for_faiss = search_query_vector.reshape(1, self.vector_dimension)

            self.logger.debug(f"    Faiss search: k={k}, query_vector_shape={query_vector_for_faiss.shape}")
            scores_matrix, internal_ids_matrix = target_faiss_index.search(query_vector_for_faiss, k)
            self.logger.debug(f"    Faiss search returned scores_matrix shape: {scores_matrix.shape}, ids_matrix shape: {internal_ids_matrix.shape}")

            retrieved_internal_ids: List[int] = []
            retrieved_scores: List[float] = []
            
            for id_val, score_val in zip(internal_ids_matrix[0], scores_matrix[0]):
                if id_val != -1: 
                    retrieved_internal_ids.append(int(id_val))   
                    retrieved_scores.append(float(score_val)) 
                else:
                    self.logger.debug(f"    Faiss search: Encountered -1 ID, indicating fewer than k={k} results or padding.")
            
            if not retrieved_internal_ids:
                self.logger.info(f"    Faiss 搜索在 '{selected_index_name}' 中完成，但未返回任何有效的结果 ID。")
                return [] 
            
            self.logger.info(f"    Faiss 搜索在 '{selected_index_name}' 中完成，初步找到 {len(retrieved_internal_ids)} 个候选文档的 internal_id。")

        except Exception as e:
            self.logger.error(f"Faiss 搜索错误: 在 '{selected_index_name}' 中执行 Faiss 搜索时发生错误: {e}", exc_info=True)
            return [] 

        # --- 步骤 5: 根据检索到的 internal_ids 从 SQLite 数据库批量获取这些文档的完整元数据 ---
        self.logger.debug(f"  - Retriever步骤 5: 使用找到的 {len(retrieved_internal_ids)} 个 internal_id，从 SQLite 数据库批量获取文档元数据...")
        documents_map_from_db = self.indexer.get_documents_by_internal_ids(retrieved_internal_ids)
        self.logger.info(f"    已成功从数据库中获取了 {len(documents_map_from_db)} 条与 internal_id 对应的文档记录。")

        # --- 步骤 6: 组合结果：将元数据与相似度得分结合，并保持 Faiss 返回的原始排序 ---
        self.logger.debug(f"  - Retriever步骤 6: 组合文档元数据与相似度得分，并按 Faiss 原始顺序排列...")
        final_retrieved_docs: List[Dict[str, Any]] = [] 
        
        for internal_id, score in zip(retrieved_internal_ids, retrieved_scores):
            doc_data_from_db = documents_map_from_db.get(internal_id)
            
            if doc_data_from_db:
                doc_data_from_db['score'] = score 
                final_retrieved_docs.append(doc_data_from_db)
            else:
                self.logger.warning(f"数据不一致警告: 在数据库中未能找到 Faiss 返回的 internal_id: {internal_id}。")
                self.logger.warning(f"                 这可能表示 Faiss 索引与数据库元数据之间存在不一致。将跳过此条检索结果。")

        self.logger.info(f"检索流程成功完成，最终返回 {len(final_retrieved_docs)} 个文档（已按相似度排序）。")
        return final_retrieved_docs


    def close(self):
        """
        关闭 Retriever 实例时调用的清理方法。
        Retriever 本身通常没有需要显式关闭的外部资源 (因为它主要依赖于 Indexer 提供的资源)。
        此方法主要用于记录 Retriever 的关闭事件。
        """
        self.logger.info("开始关闭 Retriever 实例...")
        # 通常无需执行特定的资源释放操作，因为 Encoder, Faiss索引, DB连接等由 Indexer 管理。
        self.logger.info("Retriever 实例关闭完成。")

# -------------------------------------------------------------------------------------------------
# 生成器类 (Generator)
# -------------------------------------------------------------------------------------------------
class Generator:
    """
    Generator 类负责与大语言模型 (LLM) API (此处特指 ZhipuAI 的 API) 进行交互，
    以根据用户查询和检索到的上下文信息生成最终的自然语言答案。

    其核心工作流程包括：
    1.  **构建提示 (Prompt)**: 将用户的原始查询和由 Retriever 检索到的相关文档上下文列表，
        组合成一个结构化的提示 (prompt)。这个提示会指导 LLM 如何根据提供的上下文来回答问题，
        并遵循特定的规则（例如，要求 LLM 仅基于提供的上下文作答、如何处理信息不足的情况等）。
    2.  **调用 LLM API**: 将构建好的提示发送给指定的 ZhipuAI 大语言模型 API。
    3.  **处理响应**: 对 LLM API 返回的原始文本响应进行基本的后处理（例如，去除多余的空白字符）。

    此类依赖于 `zhipuai` Python 库来与 ZhipuAI API 进行通信，并且需要一个有效的 API Key。
    API Key 的获取优先级如下：
    - 首先，从构造函数参数 `api_key` 获取。
    - 如果构造函数未提供，则尝试从环境变量 `ZHIPUAI_API_KEY` 读取。
    如果没有有效的 API Key，Generator 将无法工作。
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "glm-4-flash"):
        """
        初始化 Generator 实例。

        Args:
            api_key (Optional[str]): ZhipuAI 的 API Key。如果在此处提供，将优先使用这个 Key。
                                     如果为 None，则会尝试从环境变量 `ZHIPUAI_API_KEY` 中读取。
            model_name (str): 指定要调用的 ZhipuAI 平台的模型名称。例如 "glm-4-flash", "glm-4" 等。
                              不同的模型具有不同的能力、速度、上下文窗口大小和调用成本。
                              默认值为 "glm-4-flash"，这是一个速度较快且性价比较高的模型。
                              老板请注意: "glm-4-flash" 已经是智谱AI的轻快版模型，平衡了性能与资源。
                              若需调整，请参考智谱AI官方文档选择合适的模型。
                              请查阅 ZhipuAI 官方文档以获取最新的可用模型列表和特性。

        Raises:
            ValueError: 如果 `api_key` 参数为 None 并且在环境变量 `ZHIPUAI_API_KEY` 中也找不到有效的 Key。
                        没有 API Key，Generator 将无法与 ZhipuAI 服务通信。
            RuntimeError: 如果 ZhipuAI 客户端在初始化过程中发生其他错误 (例如，网络问题、`zhipuai`库安装问题等)。
        """
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.info(f"开始初始化 Generator，准备使用 ZhipuAI 模型: {model_name}")
        
        # 决定最终使用的 API Key：优先使用通过参数传入的，否则尝试从环境变量获取。
        final_api_key = api_key if api_key else os.getenv("ZHIPUAI_API_KEY")

        # 检查是否成功获取到 API Key。
        if not final_api_key:
            error_message = ("Generator 初始化错误: ZhipuAI API Key 未提供。\n"
                             "请通过以下方式之一提供 API Key：\n"
                             "  1. 在初始化 Generator 时，通过 'api_key' 参数传入。\n"
                             "  2. 将 API Key 设置到名为 'ZHIPUAI_API_KEY' 的环境变量中。")
            self.logger.critical(error_message) 
            raise ValueError(error_message)
        else:
            self.logger.info("成功获取到 ZhipuAI API Key (来源可能是参数或环境变量)。")

        try:
            self.client = zhipuai.ZhipuAI(api_key=final_api_key)
            self.model_name = model_name
            self.logger.info(f"ZhipuAI 客户端已使用模型 '{self.model_name}' 成功初始化。")
        except Exception as e:
             self.logger.error(f"Generator 初始化错误: 初始化 ZhipuAI 客户端失败。错误详情: {e}", exc_info=True)
             self.logger.error(f"请确认以下几点：")
             self.logger.error(f"  - 提供的 API Key 是否有效且具有调用模型 '{self.model_name}' 的权限。")
             self.logger.error(f"  - 'zhipuai' Python 库是否已正确安装 (例如，通过 pip install zhipuai)。")
             self.logger.error(f"  - 网络连接是否正常，能否访问 ZhipuAI API 服务端点。")
             raise RuntimeError(f"ZhipuAI客户端初始化失败: {e}") from e 
        
        self.logger.info("Generator 初始化成功完成。")

    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        根据用户提供的原始查询和由 Retriever 返回的文档上下文列表，调用大语言模型 (LLM) 生成回答。

        Args:
            query (str): 用户提出的原始问题或查询字符串。
            context (List[Dict[str, Any]]): 一个文档字典列表，通常由 Retriever 的 `retrieve` 方法返回。
                                 每个字典代表一个检索到的相关文档，应包含诸如 'id', 'text', 
                                 'image_path', 'score' 等信息。

        Returns:
            str: 由大语言模型生成并经过基本后处理的文本响应。
                 如果在调用 LLM API 时发生错误，会返回一条包含错误信息的提示性字符串。
        """
        self.logger.info(f"开始为查询生成最终响应...")
        self.logger.info(f"  接收到的用户查询: '{query[:100]}{'...' if len(query)>100 else ''}'")
        self.logger.info(f"  使用 {len(context)} 个检索到的文档作为生成上下文。")

        # --- 步骤 1: 构建发送给 LLM 的 Prompt (通常表现为消息列表 `messages`) ---
        self.logger.debug("  - Generator步骤 1: 构建 Prompt (包含系统指令、上下文和用户查询)...")
        messages_for_llm = self._build_messages(query, context)

        if messages_for_llm:
            if messages_for_llm[0]['role'] == 'system':
                 system_prompt_content = messages_for_llm[0]['content']
                 context_start_marker = "# 参考文档:"
                 context_start_index = system_prompt_content.find(context_start_marker)
                 if context_start_index != -1:
                     system_instructions_part = system_prompt_content[:context_start_index].strip()
                     self.logger.debug(f"    生成的系统消息 (指令部分): {system_instructions_part[:400]}{'...' if len(system_instructions_part)>400 else ''}")
                 else: 
                     self.logger.debug(f"    生成的系统消息 (部分): {system_prompt_content[:400]}{'...' if len(system_prompt_content)>400 else ''}")

            if len(messages_for_llm) > 1 and messages_for_llm[1]['role'] == 'user':
                 self.logger.debug(f"    生成的用户消息 (原始查询): {messages_for_llm[1]['content']}")
        self.logger.debug("    Prompt 构建完成。")

        # --- 步骤 2: 调用 ZhipuAI Chat Completions API ---
        self.logger.info(f"  - Generator步骤 2: 开始调用 ZhipuAI Chat API (使用模型: {self.model_name})...")
        llm_raw_response_content = "抱歉，在尝试从语言模型生成响应时遇到了一个未知问题。" 
        try:
            api_response = self.client.chat.completions.create(
                model=self.model_name,      
                messages=messages_for_llm,   
                temperature=0.7,            
                max_tokens=1500,            
            )
            
            if api_response.choices and api_response.choices[0].message and api_response.choices[0].message.content:
                llm_raw_response_content = api_response.choices[0].message.content
                self.logger.info(f"    ZhipuAI API 调用成功。已接收到模型的响应。")
            else:
                self.logger.warning("    ZhipuAI API 调用似乎成功，但响应结构不符合预期 (choices, message, or content为空)。将使用默认错误消息。")
                # Log the actual response for debugging if it's not as expected
                self.logger.debug(f"    Actual API response object: {api_response.model_dump_json(indent=2)}")


            if hasattr(api_response, 'usage') and api_response.usage:
                completion_tokens = api_response.usage.completion_tokens 
                prompt_tokens = api_response.usage.prompt_tokens         
                total_tokens = api_response.usage.total_tokens           
                self.logger.info(f"      Token 使用情况 -> 输入提示: {prompt_tokens} tokens, 生成响应: {completion_tokens} tokens, 总计: {total_tokens} tokens.")
            else:
                self.logger.info("      未能从 API 响应中获取详细的 token 使用情况。")


        except zhipuai.APIStatusError as e_status:
             self.logger.error(f"  错误：ZhipuAI API 返回了状态错误。")
             self.logger.error(f"        HTTP 状态码: {e_status.status_code}")
             # Accessing type and message safely
             error_type = getattr(e_status, 'type', 'N/A')
             error_message_detail = getattr(e_status, 'message', str(e_status))
             self.logger.error(f"        错误类型: {error_type}")
             self.logger.error(f"        错误消息: {error_message_detail}")
             llm_raw_response_content = (f"抱歉，调用语言模型时遇到 API 错误 (状态码: {e_status.status_code})。 "
                                         f"请检查您的 API Key、账户状态或请求参数，或稍后重试。错误信息: {error_message_detail}")
        except zhipuai.APIConnectionError as e_conn:
             self.logger.error(f"  错误：无法连接到 ZhipuAI API 服务器: {e_conn}")
             llm_raw_response_content = ("抱歉，无法连接到语言模型服务。 "
                                         "请检查您的网络连接，或确认 ZhipuAI API 端点是否正确且可访问。")
        except zhipuai.APIRequestFailedError as e_req_failed:
            self.logger.error(f"  错误: ZhipuAI API 请求失败: {e_req_failed}")
            error_message_detail = getattr(e_req_failed, 'message', str(e_req_failed))
            llm_raw_response_content = f"抱歉，语言模型API请求失败。可能原因包括请求参数无效或服务内部错误。详情: {error_message_detail}"
        except zhipuai.APITimeoutError as e_timeout:
            self.logger.error(f"  错误: ZhipuAI API 请求超时: {e_timeout}")
            llm_raw_response_content = "抱歉，与语言模型的通信超时。请稍后重试，或检查网络延迟。"
        except Exception as e_unknown:
             self.logger.error(f"  错误：调用 LLM 时发生未预料的异常: {e_unknown}", exc_info=True) 
             llm_raw_response_content = ("抱歉，在与语言模型交互并生成响应的过程中，发生了一个意外的内部错误。 "
                                         "请查看详细日志以获取更多信息。")

        # --- 步骤 3: 对 LLM 的原始响应进行后处理 ---
        self.logger.debug("  - Generator步骤 3: 对 LLM 的原始响应进行后处理...")
        final_processed_response = self._postprocess_response(llm_raw_response_content)
        self.logger.debug("    LLM 响应后处理完成。")
        
        self.logger.info("LLM 响应生成流程结束。")
        return final_processed_response

    def _build_messages(self, query: str, context: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        根据用户查询和检索到的上下文文档，构建符合 ZhipuAI Chat API 要求的消息列表 (messages)。
        这个列表通常包含两条主要消息：
        1.  **System Message (`role: "system"`)**: 提供系统级的指令，设定 LLM 的角色、行为规则，
            并在此处注入检索到的上下文信息（格式化为文本）。
        2.  **User Message (`role: "user"`)**: 包含用户原始的查询字符串。

        Args:
            query (str): 用户的原始查询。
            context (List[Dict[str, Any]]): Retriever 返回的上下文文档列表。每个文档字典应包含 'id', 'text', 'score' 等键。

        Returns:
            List[Dict[str, str]]: 一个包含字典的列表，每个字典有 'role' 和 'content' 键，
                                  可以直接传递给 ZhipuAI Chat API 的 `messages` 参数。
        """
        self.logger.debug("开始构建用于 LLM 的消息列表...")
        system_message_content = """
        你是一个高度专业且严谨的文档问答助手。你的任务是根据下面提供的 "参考文档" 部分中的信息来精确地回答用户提出的问题。

        # 核心指令与行为准则:
        1.  **严格依据参考信息**: 你的回答必须 **完全且仅** 基于 "参考文档" 中明确提供的信息。严禁使用任何你在训练数据中学习到的外部知识、个人观点、进行任何形式的推断、猜测或联想超出文档内容。
        2.  **处理信息不足**: 如果 "参考文档" 中的信息不足以回答用户的问题，或者问题与所有提供的文档内容均不相关，你必须明确指出信息的缺乏。标准回答是：“根据提供的参考文档，我无法找到回答该问题所需的信息。”或者类似表述，如“参考文档中没有包含足够的信息来回答关于...的问题。”。不要试图编造答案。
        3.  **关于图像内容的理解**: 你无法直接“看到”或解析图像。你对图像的理解 **必须且只能** 来源于 "参考文档" 中与该图像关联的 **文本描述内容**，以及文档中可能提及的 **图像文件名**。绝不能声称你能直接感知图像内容。
        4.  **回答涉及图像的问题**:
            - 如果用户的问题涉及到某张图片（例如，通过图片文件名或描述性提问），请首先在 "参考文档" 的文本描述中仔细查找是否有与该图片相关的说明。
            - 如果找到了相关的文本描述，请依据该文本描述来回答。
            - 如果文档中只提供了图片的文件名但没有相应的文本描述，你可以提及这个文件名（例如，“文档提到了一个名为 'circuit_diagram.png' 的关联图片”），并明确说明文档中缺少对该图片内容的具体文字描述，因此无法进一步回答。
            - 如果文档中既没有图片描述也没有文件名信息，或者问题与文档中提及的任何图片都无关，请按照上述第2条“处理信息不足”的规则进行回复。
        5.  **引用来源 (推荐)**: 在可能的情况下，如果你的答案基于某一个或某几个特定的参考文档，请在回答中指明这些来源。例如：“根据文档 ID 'BGREF_01' 的描述...” 或 “参考文档 1 (ID: XXX) 和文档 3 (ID: YYY) 提到...”。这有助于用户追溯信息源。
        6.  **回答风格与格式**: 你的回答应尽可能地简洁、清晰、直接，并且专业。避免使用冗长的前缀、不必要的客套话或模棱两可的表述。如果答案包含多个要点，可以使用列表或分点来组织，以提高可读性。

        # 参考文档:
        --- 开始参考文档部分 ---
        """.strip() 

        context_parts_for_prompt: List[str] = [] 
        if not context:
            self.logger.info("    注意: 未向LLM提供任何检索到的上下文文档 (可能是因为检索无结果)。")
            context_parts_for_prompt.append("\n（系统提示：本次未能从知识库中检索到与用户问题相关的文档。请基于此情况进行回答，并遵循“处理信息不足”的规则。）")
        else:
            self.logger.info(f"    正在将 {len(context)} 个检索到的文档格式化为 LLM 的上下文...")
            for i, doc_info in enumerate(context):
                doc_id = doc_info.get('id', '未知ID')
                score_value = doc_info.get('score', 'N/A') 
                text_content = doc_info.get('text', '无可用文本内容')
                image_file_path = doc_info.get('image_path') 

                image_info_str = f"关联图片文件名: '{os.path.basename(image_file_path)}'" if image_file_path else "无明确关联的图片信息"

                max_text_len_for_llm = 700 
                truncated_text_content = text_content[:max_text_len_for_llm] + \
                                         ('...' if len(text_content) > max_text_len_for_llm else '')

                context_parts_for_prompt.append(f"\n--- 参考文档 {i+1} ---") 
                context_parts_for_prompt.append(f"  原始文档ID: {doc_id}")
                context_parts_for_prompt.append(f"  与查询的相关度得分: {score_value:.4f}" if isinstance(score_value, float) else f"  与查询的相关度得分: {score_value}")
                context_parts_for_prompt.append(f"  文本内容摘要: {truncated_text_content}")
                context_parts_for_prompt.append(f"  {image_info_str}")

        formatted_context_section = "\n".join(context_parts_for_prompt)
        system_message_content += "\n" + formatted_context_section + "\n--- 结束参考文档部分 ---"

        final_messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_message_content}, 
            {"role": "user", "content": query}                     
        ]
        self.logger.debug(f"为 LLM 构建的消息列表完成。共 {len(final_messages)} 条消息。")
        return final_messages

    def _postprocess_response(self, llm_raw_response: str) -> str:
        """
        对从 LLM API 获取的原始响应字符串进行基本的后处理。
        目前主要执行去除首尾空白字符的操作。
        未来可以根据需要在这里添加更复杂的处理逻辑。

        Args:
            llm_raw_response (str): 从 LLM API 收到的原始文本响应。

        Returns:
            str: 经过后处理的文本响应，准备好呈现给用户或用于后续流程。
        """
        self.logger.debug(f"开始对 LLM 原始响应进行后处理。原始响应 (前100字符): '{llm_raw_response[:100]}...'")
        processed_response = llm_raw_response.strip()
        
        # 示例：移除特定前缀 (如果模型经常添加)
        # unwanted_prefix = "根据您提供的参考文档，"
        # if processed_response.startswith(unwanted_prefix):
        #     processed_response = processed_response[len(unwanted_prefix):].strip()
        #     self.logger.debug(f"  移除了不希望的前缀 '{unwanted_prefix}'。")

        self.logger.debug(f"LLM 响应后处理完成。处理后响应 (前100字符): '{processed_response[:100]}...'")
        return processed_response

    def close(self):
        """
        关闭 Generator 实例时调用的清理方法。
        ZhipuAI 客户端通常不需要显式关闭。
        """
        self.logger.info("开始关闭 Generator 实例...")
        self.logger.info("Generator 实例关闭完成。")

# -------------------------------------------------------------------------------------------------
# 主程序执行入口 (示例使用流程)
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # =============================================================================================
    # 步骤 0: 配置运行参数和输出目录 (老板，请在这里按需修改！)
    # =============================================================================================
    
    # --- 用户可配置的运行标识符 ---
    # 请为本次运行设置一个有意义的描述性名称，例如项目名、实验批次等。
    # 这将作为顶级输出目录名称的一部分，方便区分和查找不同运行的产出。
    RUN_IDENTIFIER_BASE: str = "Multimodal_RAG_System_Run" # 使用英文以保证跨平台兼容性
    
    # 清理运行标识符，确保它能作为有效目录名。
    sanitized_run_identifier: str = sanitize_filename(RUN_IDENTIFIER_BASE, max_length=50) 

    # --- 构建本次运行的顶级输出目录 ---
    # 目录名格式: [清理后的运行标识符]_[时间戳YYYYMMDD_HHMMSS]
    run_timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_BASE_DIR: str = f"{sanitized_run_identifier}_{run_timestamp}"
    
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # --- 配置日志记录 ---
    LOG_DIR: str = os.path.join(OUTPUT_BASE_DIR, "run_logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE_PATH: str = os.path.join(LOG_DIR, "system_execution_log.txt")
    setup_logging(LOG_FILE_PATH) 

    logger.info("\n" + "="*80)
    logger.info("========= Multimodal Retrieval Augmented Generation (RAG) System =========")
    logger.info("=========                     Main Program Execution Start               =========")
    logger.info("="*80 + "\n")
    logger.info(f"User Defined Run Identifier (Base): {RUN_IDENTIFIER_BASE}")
    logger.info(f"Sanitized Run Identifier: {sanitized_run_identifier}")
    logger.info(f"All output data will be saved in the directory: {os.path.abspath(OUTPUT_BASE_DIR)}")

    # --- 数据源配置文件路径 ---
    JSON_DATA_PATH: str = 'data.json'
    IMAGE_DIR_PATH: str = 'images'
    logger.info(f"Data Source Config: JSON Metadata File='{JSON_DATA_PATH}', Image Directory='{IMAGE_DIR_PATH}'")

    # --- 持久化存储文件路径定义 ---
    DB_STORAGE_DIR: str = os.path.join(OUTPUT_BASE_DIR, "data_storage")
    DB_DIR: str = os.path.join(DB_STORAGE_DIR, "database")
    DB_FILE: str = os.path.join(DB_DIR, 'multimodal_document_library.db')

    FAISS_DIR: str = os.path.join(DB_STORAGE_DIR, "vector_indices")
    FAISS_TEXT_INDEX_FILE: str = os.path.join(FAISS_DIR, 'text_vector_index.faiss')
    FAISS_IMAGE_INDEX_FILE: str = os.path.join(FAISS_DIR, 'image_vector_index.faiss')
    FAISS_MEAN_INDEX_FILE: str = os.path.join(FAISS_DIR, 'mean_vector_index.faiss')
    
    QUERY_RESULTS_DIR: str = os.path.join(OUTPUT_BASE_DIR, "query_session_results")
    os.makedirs(QUERY_RESULTS_DIR, exist_ok=True) 

    logger.info(f"Database file will be saved to: {DB_FILE}")
    logger.info(f"Faiss index files will be saved to directory: {FAISS_DIR}")
    logger.info(f"Query session results will be saved to directory: {QUERY_RESULTS_DIR}")


    # --- 模型配置 ---
    # 老板请注意：以下模型是经过审慎选择的，旨在平衡性能与资源消耗。
    # CLIP 模型: "openai/clip-vit-base-patch32" 是一个广泛应用的基准模型。
    # LLM 模型: "glm-4-flash" 是智谱AI提供的轻快版模型，适合快速响应和资源有限的场景。
    CLIP_MODEL: str = "openai/clip-vit-base-patch32"
    LLM_MODEL: str = "glm-4-flash" 
    logger.info(f"Model Configuration: CLIP Model='{CLIP_MODEL}', Large Language Model (LLM)='{LLM_MODEL}'")

    # =============================================================================================
    # 步骤 1: 加载原始数据并尝试关联图片
    # =============================================================================================
    logger.info("\n--- [Main Flow] Step 1: Loading document data from JSON and associating image files ---")
    documents_to_index: List[Dict[str, Any]] = load_data_from_json_and_associate_images(JSON_DATA_PATH, IMAGE_DIR_PATH)
    
    if not documents_to_index:
        logger.critical("CRITICAL ERROR: Failed to load any valid document data from the JSON file, or the list is empty after image association.")
        logger.critical(f"          Please check if '{JSON_DATA_PATH}' exists, is correctly formatted, and contains valid records.")
        logger.critical("          The program will now exit due to this critical data loading failure.")
        exit(1) 
    logger.info(f"--- [Main Flow] Step 1 Complete: Successfully loaded and prepared {len(documents_to_index)} documents for indexing. ---\n")
    time.sleep(0.2) 

    # =============================================================================================
    # 步骤 2: 初始化 Indexer 并对加载的文档建立索引
    # =============================================================================================
    logger.info("--- [Main Flow] Step 2: Initializing Indexer and building indices for loaded documents ---")
    indexer_instance: Optional[Indexer] = None 
    try:
        indexer_instance = Indexer(
            db_path=DB_FILE,
            faiss_text_index_path=FAISS_TEXT_INDEX_FILE,
            faiss_image_index_path=FAISS_IMAGE_INDEX_FILE,
            faiss_mean_index_path=FAISS_MEAN_INDEX_FILE,
            clip_model_name=CLIP_MODEL
        )
        indexer_instance.index_documents(documents_to_index)

        logger.info("Index building/loading complete. Current status:")
        text_count = getattr(indexer_instance.text_index, 'ntotal', 0)   
        image_count = getattr(indexer_instance.image_index, 'ntotal', 0) 
        mean_count = getattr(indexer_instance.mean_index, 'ntotal', 0)  
        db_doc_count = indexer_instance.get_document_count()             
        logger.info(f"  - SQLite Database ('{os.path.basename(DB_FILE)}') document records: {db_doc_count}")
        logger.info(f"  - Text Faiss Index ('{os.path.basename(FAISS_TEXT_INDEX_FILE)}') vectors: {text_count}")
        logger.info(f"  - Image Faiss Index ('{os.path.basename(FAISS_IMAGE_INDEX_FILE)}') vectors: {image_count}")
        logger.info(f"  - Mean Faiss Index ('{os.path.basename(FAISS_MEAN_INDEX_FILE)}') vectors: {mean_count}")

        if text_count == 0 and image_count == 0 and mean_count == 0 and db_doc_count > 0:
             logger.warning("WARNING: Database contains document records, but all Faiss indices are empty!")
             logger.warning("      This might indicate that the encoding process failed for all documents or no valid vectors were generated.")
             logger.warning("      Subsequent retrieval operations will not be able to return results based on vector similarity. Please review Indexer and Encoder logs carefully.")
        elif db_doc_count == 0: 
             logger.warning("WARNING: Both the database and all Faiss indices are currently empty.")
             logger.warning("      This could be due to empty input JSON data or all entries being skipped during loading and processing.")

    except Exception as e:
         logger.critical(f"CRITICAL ERROR: A top-level exception occurred during Indexer initialization or index building: {e}", exc_info=True)
         logger.critical("          As the Indexer failed to prepare successfully, subsequent retrieval and generation steps may not function correctly.")
         indexer_instance = None 

    logger.info("--- [Main Flow] Step 2 Complete. ---\n")
    time.sleep(0.2)

    # =============================================================================================
    # 步骤 3: 初始化 Retriever (检索器)
    # =============================================================================================
    logger.info("--- [Main Flow] Step 3: Initializing Retriever ---")
    retriever_instance: Optional[Retriever] = None 
    if indexer_instance and (getattr(indexer_instance.text_index, 'ntotal', 0) > 0 or 
                             getattr(indexer_instance.image_index, 'ntotal', 0) > 0 or 
                             getattr(indexer_instance.mean_index, 'ntotal', 0) > 0):
        try:
            retriever_instance = Retriever(indexer=indexer_instance)
        except Exception as e:
             logger.error(f"ERROR: An exception occurred during Retriever initialization: {e}", exc_info=True)
             retriever_instance = None 
    else:
         logger.warning("Skipping Retriever initialization.")
         if indexer_instance is None:
             logger.warning("  Reason: Indexer failed to initialize. Please check logs for Step 2 (Indexer initialization and index building).")
         elif indexer_instance: 
             logger.warning("  Reason: Indexer initialized successfully, but all its Faiss indices are currently empty.")
             logger.warning("        This might be due to encoding issues, data problems, or index building logic. Please check detailed logs for Step 2.")
             logger.warning("        Without searchable vectors, the Retriever cannot perform effective operations.")

    logger.info("--- [Main Flow] Step 3 Complete. ---\n")
    time.sleep(0.2)

    # =============================================================================================
    # 步骤 4: 初始化 Generator (生成器)
    # =============================================================================================
    logger.info("--- [Main Flow] Step 4: Initializing Generator (will interact with ZhipuAI API) ---")
    generator_instance: Optional[Generator] = None 
    zhipuai_api_key_from_env: Optional[str] = os.getenv("ZHIPUAI_API_KEY")
    if not zhipuai_api_key_from_env:
        logger.warning("WARNING: 'ZHIPUAI_API_KEY' not found in environment variables.")
        logger.warning("      The Generator will not be able to communicate with the ZhipuAI API, and the answer generation step will be skipped.")
        logger.warning("      To enable LLM answer generation, please do one of the following:")
        logger.warning("        1. (Recommended) Set your ZhipuAI API Key as an environment variable named 'ZHIPUAI_API_KEY'.")
        logger.warning("           Example (Linux/macOS): export ZHIPUAI_API_KEY='your_valid_api_key'")
        logger.warning("           Then, re-run this script in the same terminal session.")
        logger.warning("        2. (Alternative) Pass the API Key directly via the `api_key` parameter when initializing the Generator in the code (less secure for this example).")
    else:
        logger.info("ZHIPUAI_API_KEY detected in environment variables. Attempting to initialize Generator...")
        try:
            generator_instance = Generator(api_key=zhipuai_api_key_from_env, model_name=LLM_MODEL)
        except Exception as e:
             logger.error(f"ERROR: An exception occurred during Generator initialization: {e}", exc_info=True)
             generator_instance = None 

    logger.info("--- [Main Flow] Step 4 Complete. ---\n")
    time.sleep(0.2)

    # =============================================================================================
    # 步骤 5: 执行 RAG 查询示例 (包括检索 和 生成 两个阶段)
    # =============================================================================================
    logger.info("--- [Main Flow] Step 5: Executing RAG Query Examples (Retrieval + Generation) ---")

    if retriever_instance and generator_instance:
        logger.info("Retriever and Generator have been successfully initialized. Proceeding with example queries...")

        def log_retrieved_docs_summary_for_main_process(docs_list: List[Dict[str, Any]], query_log_prefix: str = "    "):
            if not docs_list:
                logger.info(f"{query_log_prefix}>> Retrieval Result: No relevant documents found for the query.")
                return
            logger.info(f"{query_log_prefix}>> Retrieval Result: Found Top-{len(docs_list)} relevant documents. Summary:")
            for i, doc_item_data in enumerate(docs_list): 
                score = doc_item_data.get('score', 'N/A') 
                score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
                text_preview = doc_item_data.get('text', 'No text content')[:70] 
                if len(doc_item_data.get('text', '')) > 70: text_preview += "..." 
                img_filename_info = ""
                if doc_item_data.get('image_path'):
                    img_filename_info = f", Associated Image: '{os.path.basename(doc_item_data['image_path'])}'"
                logger.info(f"{query_log_prefix}  {i+1}. DocID: {doc_item_data.get('id', 'N/A')} (Score: {score_str})")
                logger.info(f"{query_log_prefix}     Text Preview: '{text_preview}'{img_filename_info}")
            logger.info(f"{query_log_prefix}{'-'*40}") 

        # --- 准备示例查询数据 (减少数量以适应资源限制) ---
        text_queries_examples: List[str] = [
            "What is a bandgap voltage reference and its main purpose?", # 英文查询以配合英文目录和文件名
            "Explain how PTAT current is generated and its role in bandgap circuits.",
            # "Describe the basic topology of a classic bandgap reference circuit using BJTs.", # Removed for brevity
        ]

        image_docs_available_for_queries: List[Dict[str, Any]] = []
        if documents_to_index: 
            for doc_data_item_source in documents_to_index: 
                img_path_source = doc_data_item_source.get('image_path')
                if img_path_source and os.path.exists(img_path_source) and os.path.isfile(img_path_source):
                    image_docs_available_for_queries.append({
                        'id': doc_data_item_source.get('id'), 
                        'image_path': img_path_source,
                        'text': doc_data_item_source.get('text', '') 
                    })

        image_queries_examples_data: List[Dict[str, Any]] = []
        multimodal_queries_examples_data: List[Dict[str, Any]] = []

        if image_docs_available_for_queries:
            num_image_query_samples = min(1, len(image_docs_available_for_queries)) # Reduced to 1 sample
            logger.info(f"Found {len(image_docs_available_for_queries)} documents with valid images. Will randomly select {num_image_query_samples} for image/multimodal query examples.")
            selected_image_docs_for_queries = random.sample(image_docs_available_for_queries, num_image_query_samples)

            for selected_doc_info in selected_image_docs_for_queries:
                doc_id_for_query = selected_doc_info['id']
                img_path_for_query = selected_doc_info['image_path']
                img_filename_for_query = os.path.basename(img_path_for_query) 

                image_queries_examples_data.append({
                    'query_input': {'image_path': img_path_for_query},
                    'query_for_generator': f"This image (filename: {img_filename_for_query}) primarily shows what circuit structure or key concept? Please explain in detail based on the text description in the associated document.",
                    'description': f"PureImageQuery_About_{img_filename_for_query}"
                })

                multimodal_queries_examples_data.append({
                    'query_input': {
                        'text': f"Combining the document content and this image (filename: {img_filename_for_query}), please explain the working principle, key features, or design considerations of the circuit shown.", 
                        'image_path': img_path_for_query
                    },
                    'query_for_generator': f"Combining the document content and this image (filename: {img_filename_for_query}), please explain the working principle, key features, or design considerations of the circuit shown.",
                    'description': f"MultimodalQuery_Explain_Image_{img_filename_for_query}"
                })
        else:
             logger.warning("WARNING: No valid, existing image files found in the loaded data.")
             logger.warning("      Therefore, pure image query and multimodal query examples will be skipped.")


        all_example_queries_groups: List[Tuple[str, List[Any]]] = [ # Adjusted type for List[Any]
            ("Pure_Text_Queries", text_queries_examples),
            ("Pure_Image_Queries", image_queries_examples_data),
            ("Multimodal_Queries", multimodal_queries_examples_data)
        ]

        overall_query_counter = 0 
        for query_group_name, queries_in_group in all_example_queries_groups:
            logger.info(f"\n{'#'*70}\n>>> Starting Example Queries for Type: [{query_group_name}] (Total in this group: {len(queries_in_group)}) <<<\n{'#'*70}\n")

            if not queries_in_group:
                logger.info(f"    (Skipping [{query_group_name}] type queries as no query data is available.)")
                continue 

            for query_index_in_group, query_data_item in enumerate(queries_in_group):
                overall_query_counter += 1
                
                query_input_for_retriever: Union[str, Dict[str, str], None] = None 
                query_text_for_generator: Optional[str] = None          
                query_description_for_logging: Optional[str] = None     
                query_file_prefix_for_saving: Optional[str] = None     

                if query_group_name == "Pure_Text_Queries":
                    query_input_for_retriever = str(query_data_item) 
                    query_text_for_generator = str(query_data_item)
                    query_description_for_logging = str(query_data_item)
                    query_file_prefix_for_saving = sanitize_filename(f"TextQuery_{query_data_item}", max_length=60)
                else: 
                    query_input_for_retriever = query_data_item['query_input'] # type: ignore
                    query_text_for_generator = query_data_item['query_for_generator'] # type: ignore
                    # Use the pre-defined description from the dict, as it's already sanitized for filename use
                    query_description_for_logging = query_data_item['description'] # type: ignore
                    query_file_prefix_for_saving = sanitize_filename(query_data_item['description'], max_length=60) # type: ignore
                
                logger.info(f"\n--- Processing Query #{overall_query_counter} (Type: {query_group_name} - Index: {query_index_in_group+1}/{len(queries_in_group)}) ---")
                logger.info(f"Query Description: {str(query_description_for_logging)[:120]}{'...' if len(str(query_description_for_logging))>120 else ''}")
                if isinstance(query_input_for_retriever, dict):
                    if 'text' in query_input_for_retriever:
                        logger.info(f"  -> Input text for Retriever: '{str(query_input_for_retriever['text'])[:80]}...'")
                    if 'image_path' in query_input_for_retriever:
                        logger.info(f"  -> Input image for Retriever: '{os.path.basename(str(query_input_for_retriever['image_path']))}'")
                logger.info(f"  -> Question text for Generator: '{str(query_text_for_generator)[:100]}...'")
                logger.info("-" * 30) 
                
                current_query_specific_output_dir = os.path.join(QUERY_RESULTS_DIR, f"query_{overall_query_counter:03d}_{query_file_prefix_for_saving}")
                os.makedirs(current_query_specific_output_dir, exist_ok=True)
                logger.info(f"  Detailed results for this query will be saved in: {current_query_specific_output_dir}")


                retrieved_context_docs_list: List[Dict[str, Any]] = [] 
                final_generated_response_text: str = "LLM generation step was not executed or failed due to an error." 
                
                try:
                    query_input_filename = "input_to_retriever.json" if isinstance(query_input_for_retriever, dict) else "input_to_retriever.txt"
                    query_input_save_path = os.path.join(current_query_specific_output_dir, query_input_filename)
                    with open(query_input_save_path, 'w', encoding='utf-8') as f_query_in:
                        if isinstance(query_input_for_retriever, dict):
                            json.dump(query_input_for_retriever, f_query_in, ensure_ascii=False, indent=4)
                        else:
                            f_query_in.write(str(query_input_for_retriever))
                    logger.debug(f"  Query input saved to: {query_input_save_path}")

                    logger.info("  [Retrieval Phase] Calling Retriever.retrieve() method...")
                    # Using k=2 for faster example runs
                    retrieved_context_docs_list = retriever_instance.retrieve(query_input_for_retriever, k=2) 
                    log_retrieved_docs_summary_for_main_process(retrieved_context_docs_list, query_log_prefix="    ")
                    
                    retrieved_context_save_path = os.path.join(current_query_specific_output_dir, "retrieved_context_documents.json")
                    with open(retrieved_context_save_path, 'w', encoding='utf-8') as f_retrieved_ctx:
                        json.dump(retrieved_context_docs_list, f_retrieved_ctx, ensure_ascii=False, indent=4)
                    logger.debug(f"  Full retrieved context documents saved to: {retrieved_context_save_path}")

                    if retrieved_context_docs_list: # Only generate if context is found
                        logger.info("  [Generation Phase] Calling Generator.generate() method (using retrieved context)...")
                        if query_text_for_generator is not None: 
                            final_generated_response_text = generator_instance.generate(query_text_for_generator, retrieved_context_docs_list)

                            logger.info(f"\n  <<< LLM Final Response for Query #{overall_query_counter} >>>")
                            logger.info("-" * 35)
                            logger.info(final_generated_response_text) 
                            logger.info("-" * 35)
                        else:
                            logger.error("  [Generation Phase] ERROR: Query text for Generator is None. Cannot generate response.")
                            final_generated_response_text = "Error: Query text provided to the generator was empty."
                    else:
                         logger.info("  [Generation Phase] Skipped: No relevant context documents were retrieved by the Retriever, so LLM generation is not performed.")
                         final_generated_response_text = "LLM generation was not performed as no relevant context was found."

                    llm_response_save_path = os.path.join(current_query_specific_output_dir, "final_llm_generated_response.txt")
                    with open(llm_response_save_path, 'w', encoding='utf-8') as f_llm_resp:
                        f_llm_resp.write(final_generated_response_text)
                    logger.debug(f"  LLM generated response saved to: {llm_response_save_path}")

                except Exception as e_query_processing:
                     logger.error(f"CRITICAL ERROR during processing of query '{query_description_for_logging}' (Query #{overall_query_counter}): {e_query_processing}", exc_info=True)
                     try:
                        llm_response_path = os.path.join(current_query_specific_output_dir, "final_llm_generated_response_ERROR.txt")
                        with open(llm_response_path, 'w', encoding='utf-8') as f_llm_err:
                            f_llm_err.write(f"A critical error occurred while processing this query: {e_query_processing}\nPlease check the main log file for a detailed stack trace.")
                     except Exception as e_save_err:
                        logger.error(f"Additional error: Failed to save query processing error information to file: {e_save_err}")

                logger.info(f"--- Query #{overall_query_counter} processing complete ---")
                if query_index_in_group < len(queries_in_group) - 1: 
                    delay_seconds = 0.5 # Reduced delay for faster testing
                    logger.info(f"\n...Pausing for {delay_seconds} seconds before next query in this group...\n" + "-"*70 + "\n")
                    time.sleep(delay_seconds)
            
            logger.info(f"\n{'#'*70}\n>>> All example queries for type [{query_group_name}] have been processed <<<\n{'#'*70}\n")
            time.sleep(0.5) 

    else:
        logger.critical("\nCRITICAL SYSTEM ISSUE: RAG query example flow cannot be executed due to failure in initializing one or more core components.")
        if not retriever_instance:
            logger.critical("  - Reason: Retriever failed to initialize.")
            logger.critical("    Please carefully review logs for Step 2 (Indexer initialization) and Step 3 (Retriever initialization) to identify the root cause.")
        if not generator_instance:
            logger.critical("  - Reason: Generator failed to initialize.")
            logger.critical("    Please carefully review logs for Step 4 (Generator initialization), especially checks for ZHIPUAI_API_KEY and ZhipuAI client initialization status.")
        logger.critical("Please resolve the initialization issues and try again.")

    logger.info("--- [Main Flow] Step 5 (RAG Query Examples) Complete. ---\n")

    # =============================================================================================
    # 步骤 6: 清理和关闭资源
    # =============================================================================================
    logger.info("--- [Main Flow] Step 6: Cleaning up and closing system resources ---")
    if retriever_instance: 
        retriever_instance.close()
    else:
        logger.info("  Retriever was not initialized or failed, no cleanup needed.")
        
    if generator_instance: 
        generator_instance.close()
    else:
        logger.info("  Generator was not initialized or failed, no cleanup needed.")
        
    if indexer_instance:   
        indexer_instance.close()
    else:
        logger.info("  Indexer was not initialized or failed, no cleanup (and Faiss indices might not be saved).")
        
    logger.info("--- [Main Flow] System resource cleanup and shutdown procedures complete. ---\n")

    logger.info("\n" + "="*80)
    logger.info("========= Multimodal RAG System Example Program Execution Finished =========")
    logger.info(f"All outputs (logs, database, indices, query results) have been saved in the top-level directory:")
    logger.info(f"  {os.path.abspath(OUTPUT_BASE_DIR)}")
    logger.info("Key subdirectories overview:")
    logger.info(f"  - {os.path.join(OUTPUT_BASE_DIR, 'run_logs/')}")
    logger.info(f"  - {os.path.join(OUTPUT_BASE_DIR, 'data_storage', 'database/')}")
    logger.info(f"  - {os.path.join(OUTPUT_BASE_DIR, 'data_storage', 'vector_indices/')}")
    logger.info(f"  - {os.path.join(OUTPUT_BASE_DIR, 'query_session_results/')}")
    logger.info(f"    (Under query_session_results/, each 'query_XXX_...' subdirectory contains detailed I/O for a single query)")
    logger.info("="*80 + "\n")
