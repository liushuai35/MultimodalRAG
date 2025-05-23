import sqlite3
import os
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
import json
import time
import random
import logging
import datetime
import re
import faiss
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, UnidentifiedImageError
import torch
import zhipuai
# -------------------------------------------------------------------------------------------------
# 数据加载与预处理模块
# 这是系统获取原始知识的基础步骤，必须确保数据的准确性和完整性。
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
    # 获取当前模块的日志记录器实例，用于记录此函数的执行信息。详细的日志有助于追踪数据加载过程。
    func_logger = logging.getLogger(__name__) # 使用模块级 logger
    func_logger.info(f"开始从 JSON 文件 '{json_path}' 加载数据，并在目录 '{image_dir}' 中关联图像...")

    # 步骤 1: 检查 JSON 文件是否存在。如果文件不存在，这是个严重问题，必须立即报告并停止。
    if not os.path.exists(json_path):
        func_logger.error(f"错误：JSON 文件 '{json_path}' 未找到。请检查文件路径是否正确。")
        return [] # 文件不存在，无法继续，返回空列表。

    # 初始化用于存储最终处理后文档信息的列表。
    documents: List[Dict[str, Any]] = []
    # 定义一个包含常见图像文件扩展名的列表。我们必须考虑多种可能的图像格式。
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'] # 增加了 .webp 格式以支持更多现代格式

    # 步骤 2: 尝试打开并解析 JSON 文件。文件读取和解析是容易出错的地方，必须小心处理异常。
    try:
        # 使用 'with' 语句确保文件在使用后自动关闭，即使发生错误。这是良好的资源管理习惯。
        # 'r' 表示以只读模式打开文件。
        # 'encoding='utf-8'' 指定使用 UTF-8 编码读取文件，以正确处理包括中文在内的各种字符。
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f) # 解析 JSON 数据，将其转换为 Python 的列表或字典。
            # 验证 JSON 数据的顶层结构必须是列表。不符合预期的格式是数据源常见问题。
            if not isinstance(json_data, list):
                func_logger.error(f"错误: JSON 文件 '{json_path}' 的顶层结构不是一个列表。请确保JSON文件格式正确。")
                return []
    except json.JSONDecodeError as e:
        # 如果 JSON 文件内容格式不正确，json.load() 会抛出 JSONDecodeError。这是 JSON 数据格式问题的明确指示。
        func_logger.error(f"错误：JSON 文件 '{json_path}' 解析失败。错误详情: {e}")
        func_logger.error(f"        请确保文件内容是有效的 JSON 格式 (一个包含对象的列表)。")
        return [] # JSON 格式错误，返回空列表。
    except Exception as e:
        # 捕获其他可能的读取文件错误，例如文件权限问题或磁盘问题。必须考虑到所有可能性。
        func_logger.error(f"错误：读取 JSON 文件 '{json_path}' 时发生未知错误。错误详情: {e}")
        return [] # 其他读取错误，返回空列表。

    func_logger.info(f"已成功从 '{json_path}' 加载 {len(json_data)} 条原始记录。")

    # 初始化计数器，用于统计数据处理过程中的情况。清晰的统计数据有助于了解数据质量。
    found_images_count = 0    # 成功关联到图像的文档数量。
    missing_key_count = 0     # 因缺少必要字段 ('name' 或 'description') 或内容无效而被跳过的记录数量。
    image_dir_warning_issued = False # 用于控制图像目录无效警告只输出一次。

    # 步骤 3: 遍历从 JSON 文件加载的每一条原始记录。这是核心的处理循环。
    for item_index, item in enumerate(json_data): # 使用 enumerate 获取索引，方便日志记录和问题定位。
        # 确保每一项都是字典。非字典项是无效数据。
        if not isinstance(item, dict):
            func_logger.warning(f"警告：跳过第 {item_index + 1} 条记录（JSON索引 {item_index}），因其不是一个有效的字典对象。记录内容: {item}")
            missing_key_count += 1
            continue

        doc_id = item.get('name')         # 尝试获取 'name' 字段作为文档 ID。使用 .get() 是安全的，即使键不存在也不会引发错误。
        text_content = item.get('description') # 尝试获取 'description' 字段作为文本内容。

        # 检查关键字段 'name' 和 'description' 是否存在且有值。这两个字段是文档的基本信息，缺失则无法索引。
        # 如果任一字段缺失或为空字符串（去除空白后），则跳过该条记录。
        valid_doc_id = doc_id is not None and str(doc_id).strip()
        valid_text = text_content is not None and str(text_content).strip()

        if not valid_doc_id or not valid_text:
            missing_key_count += 1
            reason = []
            if not valid_doc_id: reason.append("'name'字段缺失或为空")
            if not valid_text: reason.append("'description'字段缺失或为空")
            # 记录详细警告，包括原始索引和部分内容，便于用户排查数据源。
            func_logger.warning(f"警告：跳过第 {item_index + 1} 条记录（原始JSON索引 {item_index}），原因: {', '.join(reason)}。记录内容: {item}")
            continue # 继续处理下一条记录。

        # 初始化图像路径为 None。如果在指定目录中找不到匹配的图像，它将保持为 None。
        image_path: Optional[str] = None
        # 检查图像目录路径是否有效（已提供且存在于文件系统中）。只有目录有效时才尝试查找图像。
        if image_dir and os.path.isdir(image_dir): # 确保 image_dir 是一个存在的目录
            # 遍历预定义的图像扩展名列表，尝试构建并查找图像文件。
            for ext in image_extensions:
                # 构建潜在的图像文件名：文档ID（来自 'name' 字段）+ 当前扩展名。
                # 使用 str(doc_id) 确保即使 doc_id 是数字也能正确拼接。
                potential_image_filename = str(doc_id) + ext
                # 使用 os.path.join 安全地构建跨平台的完整图像文件路径。
                potential_image_path = os.path.join(image_dir, potential_image_filename)

                # 检查构建的图像文件路径是否存在于文件系统中，并且是一个文件。
                if os.path.exists(potential_image_path) and os.path.isfile(potential_image_path): # 确保是文件
                    image_path = potential_image_path # 找到图像，记录其完整路径。
                    found_images_count += 1           # 增加找到图像的计数。
                    break # 找到一个匹配的图像后，无需再检查其他扩展名，跳出内层循环。
        elif image_dir and not os.path.isdir(image_dir) and not image_dir_warning_issued:
            # 如果提供了 image_dir 但它不是一个有效的目录，记录一次警告。使用标志避免重复警告。
            func_logger.warning(f"提供的图像目录 '{image_dir}' 不是一个有效的目录，将无法关联图像。")
            image_dir_warning_issued = True # 设置标志，表示警告已发出。
        elif not image_dir:
             # 如果未提供图像目录，在 DEBUG 级别记录，避免在正常运行时输出过多信息。
             func_logger.debug(f"未提供图像目录 (image_dir 为 None 或空)，将不尝试关联图像。")


        # 将处理后的文档信息（包括 ID、文本和可能的图像路径）添加到 `documents` 列表中。
        documents.append({
            'id': str(doc_id), # 确保文档 ID 是字符串类型，以便后续一致处理。
            'text': str(text_content) if text_content is not None else None, # 确保文本是字符串；如果原始为 None，则保持 None。
            'image_path': image_path # 存储找到的图像路径，如果未找到则为 None。
        })

    # 步骤 4: 打印数据加载和关联过程的总结信息。清晰的总结是流程结束的标志。
    func_logger.info(f"成功准备了 {len(documents)} 个文档用于后续处理。")
    if missing_key_count > 0:
        func_logger.warning(f"在原始 JSON 数据中，共有 {missing_key_count} 条记录因格式无效或缺少有效 'name'/'description' 字段而被跳过。")
    func_logger.info(f"在有效文档中，共有 {found_images_count} 个文档成功关联了图像文件。")

    # 如果指定了图像目录，但没有找到任何图像文件（并且至少有一个文档被处理了），则给出提示，帮助用户排查图片关联问题。
    if len(documents) > 0 and found_images_count == 0 and image_dir and os.path.isdir(image_dir):
         func_logger.info(f"提示: 未在目录 '{image_dir}' 中找到任何与文档 ID 匹配的图像文件。")
         func_logger.info(f"        请检查图像文件名是否严格遵循 '文档ID.扩展名' 的格式 (例如，如果文档 'name' 是 'item01'，则图像应为 'item01.png')。")

    func_logger.info(f"--- 数据加载与图像关联流程结束 ---")
    return documents # 返回包含所有已处理文档信息的列表。


class MultimodalEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.vector_dimension = self.model.text_model.config.hidden_size

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm > 1e-9:
            return vector / norm
        else:
            return np.zeros_like(vector)

    def encode(self, text: Optional[str] = None, image_path: Optional[str] = None) -> Dict[str, Optional[np.ndarray]]:
        is_text_valid = text is not None and text.strip()
        is_image_path_valid = image_path is not None and image_path.strip()

        if not is_text_valid and not is_image_path_valid:
            print("Error: At least one of text or image_path must be provided.")
            return {'text_vector': None, 'image_vector': None, 'mean_vector': None}

        text_vector = None
        image_vector = None
        mean_vector = None

        with torch.no_grad():
            if is_text_valid:
                try:
                    text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    text_features_tensor = self.model.get_text_features(**text_inputs)
                    text_vector_raw = text_features_tensor.squeeze().cpu().numpy().astype('float32')
                    text_vector = self._normalize_vector(text_vector_raw)
                except Exception as e:
                    print(f"Error encoding text: {e}")
                    text_vector = None

            if is_image_path_valid:
                try:
                    image = Image.open(image_path).convert("RGB")
                    image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    image_features_tensor = self.model.get_image_features(**image_inputs)
                    image_vector_raw = image_features_tensor.squeeze().cpu().numpy().astype('float32')
                    image_vector = self._normalize_vector(image_vector_raw)
                except FileNotFoundError:
                    print(f"Image file not found: {image_path}")
                    image_vector = None
                except UnidentifiedImageError:
                    print(f"Failed to open or identify image file: {image_path}")
                    image_vector = None
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    image_vector = None

        if text_vector is not None and image_vector is not None:
            mean_vector = self._normalize_vector((text_vector + image_vector) / 2)

        return {
            'text_vector': text_vector,
            'image_vector': image_vector,
            'mean_vector': mean_vector
        }

class Indexer:
    def __init__(self, db_path: str, faiss_text_index_path: str, faiss_image_index_path: str, faiss_mean_index_path: str, clip_model_name: str = "openai/clip-vit-base-patch32"):
        self.db_path = db_path
        self.faiss_text_index_path = faiss_text_index_path
        self.faiss_image_index_path = faiss_image_index_path
        self.faiss_mean_index_path = faiss_mean_index_path
        self.encoder = MultimodalEncoder(clip_model_name)
        self.vector_dimension = self.encoder.vector_dimension
        self.text_index = self._load_or_create_faiss_index(self.faiss_text_index_path, "text")
        self.image_index = self._load_or_create_faiss_index(self.faiss_image_index_path, "image")
        self.mean_index = self._load_or_create_faiss_index(self.faiss_mean_index_path, "mean")
        self._init_db()

    def _init_db(self):
        db_directory = os.path.dirname(self.db_path)
        if db_directory and not os.path.exists(db_directory):
            os.makedirs(db_directory, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    internal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT UNIQUE NOT NULL,
                    text TEXT,
                    image_path TEXT
                )
            ''')
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON documents (doc_id)")
            conn.commit()

    def _load_or_create_faiss_index(self, index_path: str, index_type_description: str) -> faiss.Index:
        index_directory = os.path.dirname(index_path)
        if index_directory and not os.path.exists(index_directory):
            os.makedirs(index_directory, exist_ok=True)

        if os.path.exists(index_path) and os.path.isfile(index_path):
            index = faiss.read_index(index_path)
            if index.d != self.vector_dimension:
                print(f"Dimension mismatch for {index_type_description} index. Creating a new empty index.")
                index = self._create_new_faiss_index(index_type_description)
        else:
            print(f"No existing {index_type_description} index found. Creating a new empty index.")
            index = self._create_new_faiss_index(index_type_description)

        return index

    def _create_new_faiss_index(self, index_type_description: str) -> faiss.Index:
        quantizer = faiss.IndexFlatIP(self.vector_dimension)
        index = faiss.IndexIDMap2(quantizer)
        print(f"Created a new empty {index_type_description} Faiss index with dimension {self.vector_dimension}.")
        return index

    def index_documents(self, documents: List[Dict[str, Any]]):
        if not documents:
            print("No documents provided for indexing. Process aborted.")
            return

        print(f"Starting document indexing process. Processing {len(documents)} documents...")

        text_vectors_batch = []
        text_ids_batch = []
        image_vectors_batch = []
        image_ids_batch = []
        mean_vectors_batch = []
        mean_ids_batch = []

        processed_count = 0
        skipped_duplicate_count = 0
        skipped_invalid_input_count = 0
        encoding_failure_count = 0
        db_check_error_count = 0
        db_insert_error_count = 0

        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for i, doc_data in enumerate(documents):
                doc_id = doc_data.get('id')
                text_content = doc_data.get('text')
                image_file_path = doc_data.get('image_path')

                doc_id_str = str(doc_id).strip() if doc_id is not None else None
                if not doc_id_str:
                    print(f"Skipping document {i+1}/{len(documents)} due to missing or invalid 'id' field.")
                    skipped_invalid_input_count += 1
                    continue

                has_valid_text = text_content is not None and str(text_content).strip()
                has_valid_image = image_file_path is not None and str(image_file_path).strip()
                if not has_valid_text and not has_valid_image:
                    print(f"Skipping document ID '{doc_id_str}' as it has neither valid text content nor a valid image path.")
                    skipped_invalid_input_count += 1
                    continue

                try:
                    cursor.execute("SELECT internal_id FROM documents WHERE doc_id = ?", (doc_id_str,))
                    existing_record = cursor.fetchone()
                    if existing_record:
                        print(f"Document ID '{doc_id_str}' already exists in the database. Skipping duplicate.")
                        skipped_duplicate_count += 1
                        continue
                except sqlite3.Error as e:
                    print(f"Database error checking document ID '{doc_id_str}': {e}")
                    db_check_error_count += 1
                    continue

                encoded_data = self.encoder.encode(text=text_content, image_path=image_file_path)
                if not encoded_data or (encoded_data.get('text_vector') is None and encoded_data.get('image_vector') is None and encoded_data.get('mean_vector') is None):
                    print(f"Encoding failed for document '{doc_id_str}'. Skipping.")
                    encoding_failure_count += 1
                    continue

                internal_id = None
                try:
                    cursor.execute("INSERT INTO documents (doc_id, text, image_path) VALUES (?, ?, ?)", (doc_id_str, text_content, image_file_path))
                    internal_id = cursor.lastrowid
                    if internal_id is None:
                        print(f"Failed to retrieve internal_id for document '{doc_id_str}'. Skipping.")
                        db_insert_error_count += 1
                        continue
                except sqlite3.IntegrityError:
                    print(f"Integrity error inserting document ID '{doc_id_str}'. Skipping duplicate.")
                    skipped_duplicate_count += 1
                    continue
                except sqlite3.Error as e:
                    print(f"Database error inserting document '{doc_id_str}': {e}")
                    db_insert_error_count += 1
                    continue

                if internal_id is not None and encoded_data:
                    if encoded_data.get('text_vector') is not None:
                        text_vectors_batch.append(encoded_data['text_vector'])
                        text_ids_batch.append(internal_id)
                    if encoded_data.get('image_vector') is not None:
                        image_vectors_batch.append(encoded_data['image_vector'])
                        image_ids_batch.append(internal_id)
                    if encoded_data.get('mean_vector') is not None:
                        mean_vectors_batch.append(encoded_data['mean_vector'])
                        mean_ids_batch.append(internal_id)
                    processed_count += 1

            if text_vectors_batch:
                self.text_index.add_with_ids(np.array(text_vectors_batch, dtype='float32'), np.array(text_ids_batch, dtype='int64'))
            if image_vectors_batch:
                self.image_index.add_with_ids(np.array(image_vectors_batch, dtype='float32'), np.array(image_ids_batch, dtype='int64'))
            if mean_vectors_batch:
                self.mean_index.add_with_ids(np.array(mean_vectors_batch, dtype='float32'), np.array(mean_ids_batch, dtype='int64'))

            conn.commit()

        except Exception as e:
            print(f"An unexpected error occurred during the indexing process: {e}")
        finally:
            if conn:
                conn.close()

        print(f"Indexing process summary:")
        print(f"- Total input documents: {len(documents)}")
        print(f"- Documents skipped due to invalid input: {skipped_invalid_input_count}")
        print(f"- Documents skipped due to existing ID: {skipped_duplicate_count}")
        print(f"- Documents skipped due to database check error: {db_check_error_count}")
        print(f"- Documents skipped due to encoding failure: {encoding_failure_count}")
        print(f"- Documents skipped due to database insert error: {db_insert_error_count}")
        print(f"- Successfully processed documents: {processed_count}")

    def get_document_by_internal_id(self, internal_id: int) -> Optional[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT internal_id, doc_id, text, image_path FROM documents WHERE internal_id = ?", (internal_id,))
                row = cursor.fetchone()
                if row:
                    doc_data = dict(row)
                    doc_data['id'] = doc_data.pop('doc_id')
                    return doc_data
        except sqlite3.Error as e:
            print(f"Database error retrieving document by internal_id '{internal_id}': {e}")
        except Exception as e:
            print(f"Error retrieving document by internal_id '{internal_id}': {e}")
        return None

    def get_documents_by_internal_ids(self, internal_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        results = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                placeholders = ','.join('?' for _ in internal_ids)
                query = f"SELECT internal_id, doc_id, text, image_path FROM documents WHERE internal_id IN ({placeholders})"
                cursor.execute(query, internal_ids)
                rows = cursor.fetchall()
                for row in rows:
                    doc_data = dict(row)
                    doc_data['id'] = doc_data.pop('doc_id')
                    results[doc_data['internal_id']] = doc_data
        except sqlite3.Error as e:
            print(f"Database error retrieving documents by internal_ids: {e}")
        except Exception as e:
            print(f"Error retrieving documents by internal_ids: {e}")
        return results

    def get_document_count(self) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM documents")
                count = cursor.fetchone()[0]
                return count
        except sqlite3.Error as e:
            print(f"Database error retrieving document count: {e}")
        except Exception as e:
            print(f"Error retrieving document count: {e}")
        return 0

    def save_indices(self):
        if hasattr(self, 'text_index'):
            faiss.write_index(self.text_index, self.faiss_text_index_path)
        if hasattr(self, 'image_index'):
            faiss.write_index(self.image_index, self.faiss_image_index_path)
        if hasattr(self, 'mean_index'):
            faiss.write_index(self.mean_index, self.faiss_mean_index_path)

    def close(self):
        self.save_indices()

class Retriever:
    def __init__(self, indexer: 'Indexer'):
        self.indexer = indexer
        self.encoder = self.indexer.encoder
        self.vector_dimension = self.indexer.vector_dimension
        self.text_index = self.indexer.text_index
        self.image_index = self.indexer.image_index
        self.mean_index = self.indexer.mean_index

        # Check if the provided indexer has the necessary attributes
        required_attributes = ['text_index', 'image_index', 'mean_index', 'encoder', 'vector_dimension']
        if not all(hasattr(self.indexer, attr) for attr in required_attributes):
            raise ValueError("The provided indexer is missing some necessary attributes.")

        # Check if the Faiss indexes are not empty
        indexes = [self.text_index, self.image_index, self.mean_index]
        index_names = ["Text", "Image", "Mean"]
        for index, name in zip(indexes, index_names):
            if index is not None and hasattr(index, 'ntotal') and index.ntotal == 0:
                print(f"Warning: The {name} index is empty. Retrieval operations may not find any documents.")

    def retrieve(self, query: Union[str, Dict[str, str]], k: int = 5) -> List[Dict[str, Any]]:
        query_text = None
        query_image_path = None
        query_type = "unknown"

        if isinstance(query, str):
            query_text = query.strip()
            query_type = "text"
        elif isinstance(query, dict):
            query_text = query.get('text', '').strip()
            query_image_path = query.get('image_path', '').strip()
            if query_text and query_image_path:
                query_type = "multimodal"
            elif query_image_path:
                query_type = "image"
            elif query_text:
                query_type = "text"
            else:
                raise ValueError("The query dictionary must contain at least one of 'text' or 'image_path'.")

        encoded_query_vectors = self.encoder.encode(text=query_text, image_path=query_image_path)
        query_text_vec = encoded_query_vectors.get('text_vector')
        query_image_vec = encoded_query_vectors.get('image_vector')
        query_mean_vec = encoded_query_vectors.get('mean_vector')

        if query_text_vec is None and query_image_vec is None and query_mean_vec is None:
            print("Error: The query could not be encoded into a vector.")
            return []

        target_index = None
        search_vector = None

        if query_type == "text":
            if query_text_vec is not None and self.text_index is not None:
                target_index = self.text_index
                search_vector = query_text_vec
        elif query_type == "image":
            if query_image_vec is not None and self.image_index is not None:
                target_index = self.image_index
                search_vector = query_image_vec
        elif query_type == "multimodal":
            if query_mean_vec is not None and self.mean_index is not None:
                target_index = self.mean_index
                search_vector = query_mean_vec
            elif query_text_vec is not None and self.text_index is not None:
                target_index = self.text_index
                search_vector = query_text_vec
                print("Warning: Using text index as a fallback for multimodal query.")

        if target_index is None or search_vector is None:
            print("Error: Could not determine a target index or search vector for the query.")
            return []

        query_vector_for_faiss = search_vector.astype('float32').reshape(1, -1)
        scores, internal_ids = target_index.search(query_vector_for_faiss, k)

        retrieved_internal_ids = [int(id) for id in internal_ids[0] if id != -1]
        retrieved_scores = [float(score) for score in scores[0] if score != -1]

        documents_map = self.indexer.get_documents_by_internal_ids(retrieved_internal_ids)

        final_retrieved_docs = []
        for internal_id, score in zip(retrieved_internal_ids, retrieved_scores):
            doc_data = documents_map.get(internal_id)
            if doc_data:
                doc_data['score'] = score
                final_retrieved_docs.append(doc_data)

        return final_retrieved_docs

    def close(self):
        pass

class Generator:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "glm-4-flash"):
        # 决定最终使用的 API Key：优先使用通过参数传入的，否则尝试从环境变量获取。
        final_api_key = api_key if api_key else os.getenv("ZHIPUAI_API_KEY")

        # 检查是否成功获取到 API Key。如果获取不到，必须报错并终止初始化。
        if not final_api_key:
            raise ValueError("ZhipuAI API Key 未提供。请通过以下方式之一提供 API Key：\n"
                             "  1. 在初始化 Generator 时，通过 'api_key' 参数传入。\n"
                             "  2. 将 API Key 设置到名为 'ZHIPUAI_API_KEY' 的环境变量中。")
        try:
            # 初始化 ZhipuAI 客户端。
            self.client = zhipuai.ZhipuAI(api_key=final_api_key)
            self.model_name = model_name
        except Exception as e:
            raise RuntimeError(f"ZhipuAI客户端初始化失败: {e}") from e

    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        messages_for_llm = self._build_messages(query, context)
        if not messages_for_llm:
            return "抱歉，在准备向语言模型发送请求时遇到了内部错误（Prompt构建失败）。"

        try:
            # 调用 API，传入模型名称、消息列表、温度和最大Token数。
            api_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_for_llm,
                temperature=0.7,
                max_tokens=1500
            )
            if api_response and api_response.choices and len(api_response.choices) > 0:
                choice = api_response.choices[0]
                if choice.message and choice.message.content:
                    return choice.message.content
        except Exception as e:
            return f"抱歉，在与语言模型交互并生成响应的过程中，发生了一个意外的内部错误。错误信息: {str(e)[:200]}{'...' if len(str(e))>200 else ''}"

        return "抱歉，在尝试从语言模型生成响应时遇到了一个未知问题。"

    def _build_messages(self, query: str, context: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        system_message_content_parts = [
            "你是一个高度专业且严谨的文档问答助手。你的任务是根据下面提供的 \"参考文档\" 部分中的信息来精确地回答用户提出的问题,中文回答。",
            "\n# 核心指令与行为准则:",
            "1.  **严格依据参考信息**: 你的回答必须 **完全且仅** 基于 \"参考文档\" 中明确提供的信息。严禁使用任何你在训练数据中学习到的外部知识、个人观点、进行任何形式的推断、猜测或联想超出文档内容。这是确保回答可靠性的最重要原则。",
            "2.  **处理信息不足**: 如果 \"参考文档\" 中的信息不足以回答用户的问题，或者问题与所有提供的文档内容均不相关，你必须明确指出信息的缺乏。标准回答是：“根据提供的参考文档，我无法找到回答该问题所需的信息。”或者类似表述，如“参考文档中没有包含足够的信息来回答关于...的问题。”。不要试图编造答案，诚实地报告信息不足是专业的表现。",
            "3.  **关于图像内容的理解**: 你无法直接“看到”或解析图像文件本身的内容。你对图像的理解 **必须且只能** 来源于 \"参考文档\" 中与该图像关联的 **文本描述内容**，以及文档中可能提及的 **图像文件名**。绝不能声称你能直接感知图像内容或对其进行视觉分析。",
            "4.  **回答涉及图像的问题**:",
            "    - 如果用户的问题涉及到某张图片（例如，通过图片文件名或描述性提问），请首先在 \"参考文档\" 的文本描述中仔细查找是否有与该图片相关的说明。",
            "    - 如果找到了相关的文本描述，请依据该文本描述来回答。",
            "    - 如果文档中只提供了图片的文件名但没有相应的文本描述，你可以提及这个文件名（例如，“文档提到了一个名为 'circuit_diagram.png' 的关联图片”），并明确说明文档中缺少对该图片内容的具体文字描述，因此无法进一步回答。",
            "    - 如果文档中既没有图片描述也没有文件名信息，或者问题与文档中提及的任何图片都无关，请按照上述第2条“处理信息不足”的规则进行回复。",
            "5.  **引用来源 (推荐)**: 在可能的情况下，如果你的答案基于某一个或某几个特定的参考文档，请在回答中指明这些来源。例如：“根据文档 ID 'BGREF_01' 的描述...” 或 “参考文档 1 (ID: XXX) 和文档 3 (ID: YYY) 提到...”。这有助于用户追溯信息源，提高答案的可信度。",
            "6.  **回答风格与格式**: 你的回答应尽可能地简洁、清晰、直接，并且专业。避免使用冗长的前缀、不必要的客套话或模棱两可的表述。如果答案包含多个要点，可以使用列表或分点来组织，以提高可读性。",
            "\n# 参考文档:",
            "--- 开始参考文档部分 ---"
        ]
        system_message_content = "\n".join(system_message_content_parts).strip()

        context_parts_for_prompt = []
        if not context:
            context_parts_for_prompt.append("\n（系统提示：本次未能从知识库中检索到与用户问题相关的文档。请基于此情况进行回答，并遵循“处理信息不足”的规则。）")
        else:
            for i, doc_info in enumerate(context):
                doc_id = doc_info.get('id', '未知ID')
                score_value = doc_info.get('score', 'N/A')
                text_content = doc_info.get('text', '无可用文本内容')
                image_file_path = doc_info.get('image_path')

                image_filename = os.path.basename(image_file_path) if image_file_path else None
                image_info_str = f"关联图片文件名: '{image_filename}'" if image_filename else "无明确关联的图片信息"

                max_text_len_for_llm = 700
                truncated_text_content = text_content[:max_text_len_for_llm] + ('...' if len(text_content) > max_text_len_for_llm else '')

                doc_context_parts = [
                    f"\n--- 参考文档 {i+1} ---",
                    f"  原始文档ID: {doc_id}",
                    f"  与查询的相关度得分: {score_value:.4f}" if isinstance(score_value, float) else f"  与查询的相关度得分: {score_value}",
                    f"  文本内容摘要: {truncated_text_content}",
                    f"  {image_info_str}"
                ]
                context_parts_for_prompt.extend(doc_context_parts)

        formatted_context_section = "\n".join(context_parts_for_prompt)
        system_message_content += "\n" + formatted_context_section + "\n--- 结束参考文档部分 ---"

        final_messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": query}
        ]
        return final_messages

    def close(self):
        pass

if __name__ == "__main__":
    RUN_IDENTIFIER_BASE = "multimodal_rag_system_output"
    sanitized_run_identifier = re.sub(r'[^\w\s]', '_', RUN_IDENTIFIER_BASE)
    OUTPUT_BASE_DIR = sanitized_run_identifier
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    LOG_DIR = os.path.join(OUTPUT_BASE_DIR, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE_PATH = os.path.join(LOG_DIR, "system_execution_log.txt")

    JSON_DATA_PATH = 'data.json'
    IMAGE_DIR_PATH = 'images'

    DB_STORAGE_DIR = os.path.join(OUTPUT_BASE_DIR, "data_storage")
    DB_DIR = os.path.join(DB_STORAGE_DIR, "database")
    DB_FILE = os.path.join(DB_DIR, 'multimodal_doc_store.db')

    FAISS_DIR = os.path.join(DB_STORAGE_DIR, "vector_indices")
    FAISS_TEXT_INDEX_FILE = os.path.join(FAISS_DIR, 'text_vector_index.faiss')
    FAISS_IMAGE_INDEX_FILE = os.path.join(FAISS_DIR, 'image_vector_index.faiss')
    FAISS_MEAN_INDEX_FILE = os.path.join(FAISS_DIR, 'mean_vector_index.faiss')

    QUERY_RESULTS_DIR = os.path.join(OUTPUT_BASE_DIR, "query_session_results")
    os.makedirs(QUERY_RESULTS_DIR, exist_ok=True)

    CLIP_MODEL = "openai/clip-vit-base-patch32"
    LLM_MODEL = "glm-4-flash"

    documents_to_index = load_data_from_json_and_associate_images(JSON_DATA_PATH, IMAGE_DIR_PATH)

    if not documents_to_index:
        print("Failed to load any valid document data from the JSON file.")
        exit(1)

    indexer_instance = Indexer(
        db_path=DB_FILE,
        faiss_text_index_path=FAISS_TEXT_INDEX_FILE,
        faiss_image_index_path=FAISS_IMAGE_INDEX_FILE,
        faiss_mean_index_path=FAISS_MEAN_INDEX_FILE,
        clip_model_name=CLIP_MODEL
    )
    indexer_instance.index_documents(documents_to_index)

    retriever_instance = Retriever(indexer=indexer_instance)

    zhipuai_api_key_from_env = os.getenv("ZHIPUAI_API_KEY")
    if not zhipuai_api_key_from_env:
        print("Environment variable 'ZHIPUAI_API_KEY' not found.")
        print("Skipping Generator initialization.")
    else:
        generator_instance = Generator(api_key=zhipuai_api_key_from_env, model_name=LLM_MODEL)

    if retriever_instance and generator_instance:
        text_queries_examples = [
            "What is a bandgap voltage reference and its main purpose?",
            "Explain how the PTAT current is generated and its role in a bandgap circuit."
        ]

        for query in text_queries_examples:
            retrieved_context_docs_list = retriever_instance.retrieve(query, k=2)
            if retrieved_context_docs_list:
                final_generated_response_text = generator_instance.generate(query, retrieved_context_docs_list)
                print(f"Final Response for Query: {query}")
                print(final_generated_response_text)
                print("-" * 35)
            else:
                print(f"No relevant context found for query: {query}")

    indexer_instance.close()
    retriever_instance.close()
    if generator_instance:
        generator_instance.close()