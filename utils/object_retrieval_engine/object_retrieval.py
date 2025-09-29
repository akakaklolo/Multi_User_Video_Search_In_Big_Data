# utils/object_retrieval_engine/object_retrieval.py
import os
import sys
import glob
import scipy
from scipy import sparse as sp
import pickle
import numpy as np
import json
import re
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import io

# Bảo đảm có thể import utils/*
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
sys.path.extend([parent_dir, grand_dir])
from utils.combine_utils import merge_searching_results_by_addition


def GET_PROJECT_ROOT(marker_names=("dict", "utils"), max_up=15):
    """
    Tìm thư mục gốc dự án bằng cách đi lên tối đa max_up cấp và kiểm tra sự tồn tại của các thư mục 'marker_names'.
    Nếu không tìm thấy, trả về thư mục cha của file hiện tại (an toàn, tránh vòng lặp vô hạn).
    """
    p = Path(__file__).resolve()
    for _ in range(max_up):
        if all((p / m).exists() for m in marker_names):
            return str(p)
        if p.parent == p:  # chạm root ổ đĩa
            break
        p = p.parent
    return str(Path(__file__).resolve().parent)


PROJECT_ROOT = GET_PROJECT_ROOT()

# ========= Helpers chống rỗng / fallback =========

_TOKEN_RE = re.compile(r"(?u)\b\w+\b")

def _has_any_token(texts):
    for t in texts:
        if isinstance(t, str) and _TOKEN_RE.search(t or ""):
            return True
    return False

def _fit_tfidf_safe(texts, ngram_range=(1, 1)):
    """
    Trả về (X, vec).
    - Nếu tất cả tài liệu rỗng → trả về CSR (n,0), vec=None.
    - Thử word-level trước; nếu empty vocab → fallback char_wb(3,5).
    """
    texts = [t if isinstance(t, str) else "" for t in texts]
    n = len(texts)

    if not _has_any_token(texts):
        warnings.warn("[ObjectRetrieval] All docs empty → returning (n,0) matrix.")
        return sp.csr_matrix((n, 0), dtype=np.float32), None

    vec = TfidfVectorizer(
        input='content',
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",     # giữ token 1 ký tự, hợp đa ngôn ngữ
        ngram_range=ngram_range,
        strip_accents="unicode",
        lowercase=True,
        stop_words=None,                  # KHÔNG dùng 'english' cho dữ liệu TV
        min_df=1, max_df=1.0
    )
    try:
        X = vec.fit_transform(texts).tocsr()
        if not vec.vocabulary_:
            raise ValueError("empty vocabulary")
        return X, vec
    except ValueError as e:
        if "empty vocabulary" in str(e).lower():
            warnings.warn("[ObjectRetrieval] Empty vocab → fallback char_wb(3,5).")
            vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1, max_df=1.0,
                strip_accents=None, lowercase=False
            )
            X = vec.fit_transform(texts).tocsr()
            return X, vec
        raise

def _transform_safe(vec, text):
    """Trả về CSR (1, d). Nếu vec=None → (1,0)."""
    if vec is None:
        return sp.csr_matrix((1, 0), dtype=np.float32)
    return vec.transform([text or ""]).tocsr()


class load_file:
    def __init__(
        self,
        clean_data_path,              # dict: mỗi key (bbox/class/color/tag/number) -> glob path đến thư mục dữ liệu sạch
        save_tfids_object_path,       # nơi lưu TF-IDF vectorizer + sparse matrix
        update: bool,
        all_datatpye,
        context_data=None,
        ngram_range=(1, 1),
        input_datatype='txt',
    ):
        tfidf_transform = {}
        context_matrix = {}

        for data_type in all_datatpye:
            tfidf_pkl = os.path.join(save_tfids_object_path, f'tfidf_transform_{data_type}.pkl')
            sparse_npz = os.path.join(save_tfids_object_path, f'sparse_context_matrix_{data_type}.npz')

            if (not os.path.exists(tfidf_pkl)) or update:
                # Load context: từ file (clean_data_path) hoặc từ context_data bên ngoài
                if context_data is None:
                    clean_paths = os.path.join(PROJECT_ROOT, clean_data_path[data_type])
                    context = self.load_context(clean_paths, input_datatype)
                    # Debug nhẹ
                    if len(context) > 0:
                        print(f"[{data_type}] sample:", context[0][:100])
                else:
                    context = context_data

                # Fit TF-IDF an toàn
                X, vec = _fit_tfidf_safe(context, ngram_range=ngram_range)
                tfidf_transform[data_type] = vec
                context_matrix[data_type] = X

                # Debug nhẹ
                if vec is not None:
                    try:
                        print(vec.get_feature_names_out()[:10])
                    except Exception:
                        pass
                print(f"[{data_type}] matrix shape:", context_matrix[data_type].shape)

                # Lưu
                os.makedirs(save_tfids_object_path, exist_ok=True)
                with open(tfidf_pkl, 'wb') as f:
                    pickle.dump(tfidf_transform[data_type], f)
                scipy.sparse.save_npz(sparse_npz, context_matrix[data_type])

    def load_context(self, clean_data_paths, input_datatype):
        """
        Đọc dữ liệu ngữ cảnh (TXT/JSON) và trả về list[str] đã được strip & preprocess (với JSON).
        """
        context = []

        if input_datatype == 'txt':
            data_paths = []
            cxx_data_paths = glob.glob(clean_data_paths)
            cxx_data_paths.sort()

            for cxx_data_path in cxx_data_paths:
                paths = glob.glob(os.path.join(cxx_data_path, '*.txt'))
                # nếu tên file có 3 số cuối (e.g. _123.txt) thì sort theo số; nếu không có thì cứ sort mặc định
                try:
                    paths.sort(reverse=False, key=lambda s: int(Path(s).stem[-3:]))
                except Exception:
                    paths.sort()

                data_paths += paths

            for path in data_paths:
                # Đọc TXT với UTF-8, fallback UTF-8-SIG
                try:
                    with io.open(path, 'r', encoding='utf-8') as f:
                        data = f.readlines()
                except UnicodeDecodeError:
                    with io.open(path, 'r', encoding='utf-8-sig') as f:
                        data = f.readlines()
                data = [item.strip() for item in data]
                context += data

        elif input_datatype == 'json':
            context_paths = glob.glob(clean_data_paths)
            context_paths.sort()

            for cxx_context_path in context_paths:
                paths = glob.glob(os.path.join(cxx_context_path, '*.json'))
                # nếu tên file có 3 số cuối (e.g. _123.json) thì sort theo số; nếu không có thì cứ sort mặc định
                try:
                    paths.sort(reverse=False, key=lambda x: int(Path(x).stem[-3:]))
                except Exception:
                    paths.sort()

                for path in paths:
                    # Đọc JSON với UTF-8, fallback UTF-8-SIG
                    try:
                        with io.open(path, 'r', encoding='utf-8') as f:
                            payload = json.load(f)
                    except UnicodeDecodeError:
                        with io.open(path, 'r', encoding='utf-8-sig') as f:
                            payload = json.load(f)

                    # payload: list[...] -> join rồi preprocess
                    context += [self.preprocess_text(' '.join(line)) for line in payload]
        else:
            print(f'Not support reading the {input_datatype}')
            sys.exit(1)

        return context

    @staticmethod
    def preprocess_text(text: str):
        text = text.lower()
        # Giữ chữ + số + khoảng trắng + các ký tự tiếng Việt, bỏ còn lại
        reg_pattern = r'[^a-z0-9A-Z_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễếệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ\s]'
        output = re.sub(reg_pattern, '', text)
        output = output.strip()
        return output


class object_retrieval(load_file):
    def __init__(
        self,
        clean_data_path={
            'bbox':   'dict/context_encoded/bboxes_encoded/*',
            'class':  'dict/context_encoded/classes_encoded/*',
            'color':  'dict/context_encoded/colors_encoded/*',
            'tag':    'dict/context_encoded/tags_encoded/*',
            'number': 'dict/context_encoded/number_encoded/*',
        },
        update: bool = False,
        save_tfids_object_path=os.path.join(PROJECT_ROOT, 'dict', 'bin', 'contexts_bin'),
        save_corpus_path=os.path.join(PROJECT_ROOT, 'dict', 'tag', 'tag_corpus.txt')
    ):
        # Bảo đảm thư mục tồn tại
        os.makedirs(os.path.join(PROJECT_ROOT, 'dict', 'bin'), exist_ok=True)
        os.makedirs(save_tfids_object_path, exist_ok=True)

        # Xác định các data type hợp lệ
        all_datatpye = [key for key in clean_data_path.keys() if (clean_data_path[key] is not None)]

        # Fit/Load TF-IDF và context matrix
        super().__init__(
            clean_data_path=clean_data_path,
            save_tfids_object_path=save_tfids_object_path,
            update=update,
            all_datatpye=all_datatpye,
        )

        # Nạp lại đối tượng TF-IDF và sparse matrix từ disk
        self.tfidf_transform = {}
        self.context_matrix = {}
        for data_type in all_datatpye:
            tfidf_pkl = os.path.join(save_tfids_object_path, f'tfidf_transform_{data_type}.pkl')
            sparse_npz = os.path.join(save_tfids_object_path, f'sparse_context_matrix_{data_type}.npz')
            with open(tfidf_pkl, 'rb') as f:
                self.tfidf_transform[data_type] = pickle.load(f)  # có thể là None
            self.context_matrix[data_type] = scipy.sparse.load_npz(sparse_npz)

        self.clean_data_path = clean_data_path

        # Xuất corpus tag (nếu có vocab)
        vec_tag = self.tfidf_transform.get('tag')
        try:
            self.tag_corpus = vec_tag.get_feature_names_out() if vec_tag is not None else np.array([], dtype=str)
        except Exception:
            self.tag_corpus = np.array([], dtype=str)

        if (self.tag_corpus.size > 0) and (not os.path.exists(save_corpus_path)):
            os.makedirs(os.path.dirname(save_corpus_path), exist_ok=True)
            corpus = [' '.join(words.split('_')) for words in self.tag_corpus]
            corpus = '\n'.join(corpus) + '\n'
            with open(save_corpus_path, 'w', encoding='utf-8') as f:
                f.write(corpus)

    def transform_input(self, input_query: str, transform_type: str):
        """
        Biến đổi query chuỗi thành vector TF-IDF theo từ vựng của từng kênh (bbox/class/color/tag/number).
        An toàn khi vectorizer is None.
        """
        if transform_type in ['bbox', 'class', 'color', 'tag', 'number']:
            vec = self.tfidf_transform[transform_type]
            vectorize = _transform_safe(vec, input_query)
        else:
            print('This transform_type does not support')
            sys.exit(1)
        return vectorize

    def __call__(self, texts, k=100, index=None):
        """
        texts: dict với các khoá 'bbox'/'class'/'color'/'tag'/'number' (chuỗi đã encode)
        Trả về: (scores, idx_image) sau khi merge bằng phép cộng.
        """
        scores, idx_image = [], []
        for input_type in ['bbox', 'color', 'class', 'tag', 'number']:
            if texts.get(input_type) is not None:
                s, idx = self.find_similar_score(texts[input_type], input_type, k, index=index)
                # Bỏ qua kênh không có dữ liệu/doc
                if s.size > 0 and idx.size > 0:
                    scores.append(s)
                    idx_image.append(idx)

        scores, idx_image = merge_searching_results_by_addition(scores, idx_image)
        return scores, idx_image

    def find_similar_score(self, text: str, transform_type: str, k: int, index):
        # Guard: nếu không có document ở kênh này → trả rỗng
        n_docs = self.context_matrix[transform_type].shape[0]
        if n_docs == 0:
            return np.array([], dtype=float), np.array([], dtype=int)

        vectorize = self.transform_input(text, transform_type)

        if index is None:
            # so khớp toàn bộ
            scores = vectorize.dot(self.context_matrix[transform_type].T).toarray()[0]
            sort_index = np.argsort(scores)[::-1][:k]
            scores = scores[sort_index]
        else:
            if len(index) == 0:
                return np.array([], dtype=float), np.array([], dtype=int)
            # so khớp trong tập con (index)
            scores = vectorize.dot(self.context_matrix[transform_type][index, :].T).toarray()[0]
            sort_index = np.argsort(scores)[::-1][:k]
            scores = scores[sort_index]
            sort_index = np.array(index)[sort_index]

        return scores, sort_index


if __name__ == '__main__':
    # Test nhẹ
    inputs = {
        'bbox': "a0kite b0kite",
        'class': "people1 tv1",
        'color': None,
        'tag': None,
        'number': None,
    }
    obj = object_retrieval()
    # scores, idx = obj(inputs, k=3)
    # print(scores, idx)
    pass
