# utils/semantic_extract.py
import os
import glob
import torch
import json
import numpy as np
import sys
from typing import List, Union
import torch.nn.functional as F
import faiss
from transformers import AutoTokenizer, AutoModel
import gc
from tqdm import tqdm
from pathlib import Path
import io


def GET_PROJECT_ROOT(marker_names=("dict", "utils"), max_up=15):
    """
    Tìm thư mục gốc dự án bằng cách đi lên tối đa max_up cấp và kiểm tra marker folders.
    Trả về thư mục cha của file hiện tại nếu không tìm thấy (tránh vòng lặp vô hạn).
    """
    p = Path(__file__).resolve()
    for _ in range(max_up):
        if all((p / m).exists() for m in marker_names):
            return str(p)
        if p.parent == p:
            break
        p = p.parent
    return str(Path(__file__).resolve().parent)


PROJECT_ROOT = GET_PROJECT_ROOT()


class semantic_extract:
    def __init__(
            self,
            model: str = 'sentence-transformers/stsb-xlm-r-multilingual',
            context_path: Union[str, list] = os.path.join(PROJECT_ROOT, "dict/captions"),
            context_vector_path: str = os.path.join(PROJECT_ROOT,
                                                    "data/TransNetDatabase/CaptionFeatures/context_vector.npy"),
            input_datatype: str = 'txt',
            output_datatype: str = 'torch',
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.context_path = context_path
        self.context_vector_path = context_vector_path

        if not os.path.exists(context_vector_path):
            self.raw_data = self.generate_context_embedding(context_path, context_vector_path, output_datatype,
                                                            input_datatype)
        else:
            self.raw_data = self.generate_raw_data(context_path, input_datatype)

    def get_embedding(self, inputs: List[str]):
        encode_inputs = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        with torch.no_grad():
            model_output = self.model(
                input_ids=encode_inputs['input_ids'].to(self.device),
                attention_mask=encode_inputs['attention_mask'].to(self.device),
                token_type_ids=encode_inputs.get('token_type_ids', None).to(
                    self.device) if 'token_type_ids' in encode_inputs else None,
            )
            sentences_embed = self.mean_pooling(model_output, encode_inputs['attention_mask'].to(self.device))
            sentences_embed = F.normalize(sentences_embed, p=2, dim=1)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return sentences_embed

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # first element
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def generate_raw_data(context_path: Union[str, list], type_input: str = 'txt'):
        """
        Đọc dữ liệu thô từ TXT/JSON (UTF-8, fallback UTF-8-SIG) và trả về list[str].
        context_path: có thể là 1 string (thư mục/file) hoặc list các thư mục.
        """
        raw_data = []

        if type_input == 'txt':
            # context_path có thể là 1 đường dẫn thư mục hoặc list thư mục
            dir_list = []
            if isinstance(context_path, list):
                dir_list = context_path
            else:
                dir_list = [context_path]

            txt_files = []
            for base in dir_list:
                base = os.path.abspath(base)
                if os.path.isdir(base):
                    txt_files.extend(glob.glob(os.path.join(base, '*.txt')))
                else:
                    # nếu truyền thẳng 1 file
                    if base.endswith('.txt') and os.path.exists(base):
                        txt_files.append(base)

            # sort ổn định: nếu tên file có 3 số cuối thì sort theo số, ngược lại sort mặc định
            try:
                txt_files.sort(key=lambda s: int(Path(s).stem[-3:]))
            except Exception:
                txt_files.sort()

            for path in txt_files:
                try:
                    with io.open(path, 'r', encoding='utf-8') as f:
                        data = f.readlines()
                except UnicodeDecodeError:
                    with io.open(path, 'r', encoding='utf-8-sig') as f:
                        data = f.readlines()
                data = [word.strip() for word in data]
                raw_data.extend(data)

        elif type_input == 'json':
            # context_path là thư mục chứa các thư mục con, mỗi thư mục con có nhiều .json
            if isinstance(context_path, list):
                roots = context_path
            else:
                roots = [context_path]

            context_dirs = []
            for root in roots:
                context_dirs.extend(glob.glob(os.path.join(root, '*')))
            context_dirs.sort()

            for cxx_context_path in context_dirs:
                paths = glob.glob(os.path.join(cxx_context_path, '*.json'))
                try:
                    paths.sort(reverse=False, key=lambda x: int(Path(x).stem[-3:]))
                except Exception:
                    paths.sort()

                for path in paths:
                    # Đọc JSON UTF-8, fallback UTF-8-SIG
                    try:
                        with io.open(path, 'r', encoding='utf-8') as f:
                            payload = json.load(f)
                    except UnicodeDecodeError:
                        with io.open(path, 'r', encoding='utf-8-sig') as f:
                            payload = json.load(f)
                    # payload là list[...] -> thay rỗng bằng "nan"
                    data = ["nan" if (x == '' or x == []) else x for x in payload]
                    raw_data += data
        else:
            print(f'Not support reading {type_input}')
            sys.exit(1)

        return raw_data

    def generate_context_embedding(self, context_path, save_tensor_path, type_output: str, type_input: str = 'txt'):
        raw_data = self.generate_raw_data(context_path, type_input)
        if len(raw_data) == 0:
            print("Warning: raw_data is empty; skip embedding build.")
            return raw_data

        chunk_range = 100
        context_embedding = []
        print('running embedding: ')
        for i in tqdm(range(0, len(raw_data), chunk_range)):
            context_embedding.append(self.get_embedding(raw_data[i:i + chunk_range]))
        context_embedding = torch.cat(context_embedding)

        # tạo thư mục cha
        os.makedirs(os.path.abspath(os.path.join(save_tensor_path, '..')), exist_ok=True)

        if type_output == 'numpy':
            numpy_context_embedding = context_embedding.cpu().numpy()
            np.save(save_tensor_path, numpy_context_embedding)
        elif type_output == 'torch':
            torch_context_embedding = context_embedding.cpu()
            torch.save(torch_context_embedding, save_tensor_path)
        elif type_output == 'bin':
            index = faiss.IndexFlatL2(context_embedding.shape[-1])
            print('running save faiss: ')
            for vector in tqdm(context_embedding.cpu().numpy()):
                index.add(vector.reshape(1, -1))
            faiss.write_index(index, save_tensor_path)
        else:
            print(f'Unknown type_output: {type_output}')
        return raw_data