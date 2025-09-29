import clip
import open_clip
import torch
import json
import faiss
import numpy as np
from utils.nlp_processing import Translation
from utils.combine_utils import merge_searching_results_by_addition
from utils.ocr_retrieval_engine.ocr_retrieval import ocr_retrieval
from utils.semantic_embed.speech_retrieval import speech_retrieval
from utils.object_retrieval_engine.object_retrieval import object_retrieval


class MyFaiss:
    def __init__(self, bin_clip_file: str, bin_clipv2_file: str, json_path: str, audio_json_path: str, img2audio_json_path: str):
        # FAISS indices
        self.index_clip = self.load_bin_file(bin_clip_file)
        self.index_clipv2 = self.load_bin_file(bin_clipv2_file)

        # Lưu luôn kích thước d cho tiện kiểm tra
        self.dim_clip = int(self.index_clip.d)
        self.dim_clipv2 = int(self.index_clipv2.d)

        # Retrieval engines
        self.object_retrieval = object_retrieval()
        self.ocr_retrieval = ocr_retrieval()
        self.asr_retrieval = speech_retrieval()

        # Mappings
        self.id2img_fps = self.load_json_file(json_path)                 # {int id_img: {"image_path": ..., ...}}
        self.audio_id2img_id = self.load_json_file(audio_json_path)      # {int id_audio: [int id_img, ...]}
        self.img_id2audio_id = self.load_json_file(img2audio_json_path)  # {int id_img: [int id_audio, ...]}

        # Text pre/post
        self.translater = Translation()

        # Models
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/16", device=self.__device)
        self.clipv2_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', device=self.__device, pretrained='datacomp_xl_s13b_b90k')
        self.clipv2_tokenizer = open_clip.get_tokenizer('ViT-L-14')

    # ----------------- Utils -----------------
    def load_json_file(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            js = json.load(f)
        # Chuẩn hoá key về int để tra cứu nhất quán
        return {int(k): v for k, v in js.items()}

    def load_bin_file(self, bin_file: str):
        return faiss.read_index(bin_file)

    def _lookup_info(self, sid):
        """Trả về info trong id2img_fps theo sid (thử ép về int)."""
        try:
            sid_int = int(sid)
        except Exception:
            return None
        return self.id2img_fps.get(sid_int, None)

    def _gather_infos(self, ids, scores=None):
        """
        Lọc bỏ các id không có trong mapping; giữ đồng bộ ids/scores/image_paths/infos.
        Trả về: (np_scores, np_ids, infos_list, image_paths_list)
        """
        kept_infos, kept_ids, kept_scores = [], [], []
        for i, sid in enumerate(ids):
            info = self._lookup_info(sid)
            if info is not None and 'image_path' in info:
                kept_infos.append(info)
                kept_ids.append(int(sid))
                if scores is not None:
                    kept_scores.append(float(scores[i]))
            # else: skip id bị thiếu

        if not kept_infos:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64), [], []

        np_ids = np.array(kept_ids, dtype=np.int64)
        np_scores = np.array(kept_scores, dtype=np.float32) if scores is not None else np.array([], dtype=np.float32)
        image_paths = [inf['image_path'] for inf in kept_infos]
        return np_scores, np_ids, kept_infos, image_paths

    def _ensure_id_selector(self, index_array):
        """Tạo IDSelectorArray an toàn từ list/np.array; trả None nếu mảng rỗng."""
        if index_array is None:
            return None
        idx = np.asarray(index_array, dtype=np.int64)
        if idx.size == 0:
            return None
        return faiss.IDSelectorArray(idx)

    # ---- Encode helpers (để kiểm soát kích thước) ----
    def _encode_text(self, text: str, model_type: str) -> np.ndarray:
        """Mã hoá text theo model_type ('clip' | 'clipv2'), trả np.float32 shape (1, d)."""
        if model_type == 'clip':
            toks = clip.tokenize([text]).to(self.__device)
            feats = self.clip_model.encode_text(toks)
        else:
            toks = self.clipv2_tokenizer([text]).to(self.__device)
            feats = self.clipv2_model.encode_text(toks)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype(np.float32)

    # ----------------- Searches -----------------
    def image_search(self, id_query, k):
        # reconstruct feature từ id_query
        query_feats = self.index_clip.reconstruct(int(id_query)).reshape(1, -1)

        scores, idx_image = self.index_clip.search(query_feats, k=k)
        scores = scores.flatten()
        idx_image = idx_image.flatten()

        # Lọc & gom thông tin (chống NoneType)
        scores, idx_image, infos_query, image_paths = self._gather_infos(idx_image, scores)
        return scores, idx_image, infos_query, image_paths

    def text_search(self, text, index, k, model_type):
        text = self.translater(text)

        # --- Encode theo model yêu cầu ---
        feats = self._encode_text(text, model_type)
        d_feat = feats.shape[1]

        # Chọn index theo model yêu cầu
        if model_type == 'clip':
            index_choosed = self.index_clip
            d_index = self.dim_clip
        else:
            index_choosed = self.index_clipv2
            d_index = self.dim_clipv2

        # --- Nếu dimension KHÔNG khớp, thử model/index còn lại ---
        if d_feat != d_index:
            alt_model = 'clipv2' if model_type == 'clip' else 'clip'
            feats_alt = self._encode_text(text, alt_model)
            d_feat_alt = feats_alt.shape[1]
            idx_alt = self.index_clipv2 if model_type == 'clip' else self.index_clip
            d_idx_alt = self.dim_clipv2 if model_type == 'clip' else self.dim_clip

            if d_feat_alt == d_idx_alt:
                # dùng phương án thay thế
                model_type = alt_model
                feats = feats_alt
                index_choosed = idx_alt
                d_index = d_idx_alt
            else:
                raise ValueError(
                    f"[FAISS] Dim mismatch: text_feat={d_feat} vs index({model_type})={d_index}; "
                    f"alt_feat={d_feat_alt} vs alt_index={d_idx_alt}. "
                    f"Kiểm tra lại cặp model-Index hoặc rebuild index cho khớp."
                )

        # --- ID selector (nếu có) ---
        id_sel = self._ensure_id_selector(index)
        if id_sel is None:
            scores, idx_image = index_choosed.search(feats, k=k)
        else:
            idx_arr = np.asarray(index, dtype=np.int64)
            if idx_arr.size == 0:
                return np.array([], dtype=np.float32), np.array([], dtype=np.int64), [], []
            scores, idx_image = index_choosed.search(
                feats,
                k=min(k, len(idx_arr)),
                params=faiss.SearchParametersIVF(sel=id_sel)
            )

        scores = scores.flatten()
        idx_image = idx_image.flatten()

        # --- GET INFOS KEYFRAMES_ID (robust) ---
        scores, idx_image, infos_query, image_paths = self._gather_infos(idx_image, scores)
        return scores, idx_image, infos_query, image_paths

    # ----------------- ASR helpers -----------------
    def asr_post_processing(self, tmp_asr_scores, tmp_asr_idx_image, k):
        result = dict()
        for asr_idx, asr_score in zip(tmp_asr_idx_image, tmp_asr_scores):
            lst_ids = self.audio_id2img_id.get(int(asr_idx))
            if not lst_ids:
                continue
            for idx in lst_ids:
                idx = int(idx)
                result[idx] = result.get(idx, 0.0) + float(asr_score)

        if not result:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        asr_idx_image = np.array([item[0] for item in result], dtype=np.int64)[:k]
        asr_scores = np.array([item[1] for item in result], dtype=np.float32)[:k]
        return asr_scores, asr_idx_image

    def asr_retrieval_helper(self, asr_input, k, index, semantic, keyword):
        if index is not None:
            # Map img_id -> audio_ids và gom ngược điểm về img_id
            audio_temp = dict()
            for idx in index:
                idx = int(idx)
                audio_idxes = self.img_id2audio_id.get(idx, [])
                for audio_idx in audio_idxes:
                    audio_idx = int(audio_idx)
                    audio_temp.setdefault(audio_idx, []).append(idx)

            if not audio_temp:
                return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

            audio_index = np.array(list(audio_temp.keys()), dtype=np.int64)
            tmp_asr_scores, tmp_asr_idx_image = self.asr_retrieval(
                asr_input, k=len(audio_index), index=audio_index, semantic=semantic, keyword=keyword
            )

            result = dict()
            for asr_idx, asr_score in zip(tmp_asr_idx_image, tmp_asr_scores):
                asr_idx = int(asr_idx)
                for idx in audio_temp.get(asr_idx, []):
                    result[idx] = result.get(idx, 0.0) + float(asr_score)

            if not result:
                return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

            result = sorted(result.items(), key=lambda x: x[1], reverse=True)
            asr_idx_image = np.array([item[0] for item in result], dtype=np.int64)[:k]
            asr_scores = np.array([item[1] for item in result], dtype=np.float32)[:k]
        else:
            tmp_asr_scores, tmp_asr_idx_image = self.asr_retrieval(
                asr_input, k=k, index=None, semantic=semantic, keyword=keyword
            )
            asr_scores, asr_idx_image = self.asr_post_processing(tmp_asr_scores, tmp_asr_idx_image, k)

        return asr_scores, asr_idx_image

    # ----------------- Context search -----------------
    def context_search(self, object_input, ocr_input, asr_input, k, semantic=False, keyword=True, index=None, useid=False):
        """
        inputs = {
            'bbox': "a0person",
            'class': "person0, person1",
            'color': None,
            'tag': None
        }
        """
        scores, idx_image = [], []

        # OBJECT
        if object_input is not None:
            object_scores, object_idx_image = self.object_retrieval(object_input, k=k, index=index)
            if object_scores.size and object_idx_image.size:
                scores.append(object_scores)
                idx_image.append(object_idx_image)

        # OCR
        if ocr_input is not None:
            ocr_scores, ocr_idx_image = self.ocr_retrieval(ocr_input, k=k, index=index)
            if ocr_scores.size and ocr_idx_image.size:
                scores.append(ocr_scores)
                idx_image.append(ocr_idx_image)

        # ASR
        if asr_input is not None:
            if not useid:
                asr_scores, asr_idx_image = self.asr_retrieval_helper(asr_input, k, None, semantic, keyword)
            else:
                asr_scores, asr_idx_image = self.asr_retrieval_helper(asr_input, k, index, semantic, keyword)
            if asr_scores.size and asr_idx_image.size:
                scores.append(asr_scores)
                idx_image.append(asr_idx_image)

        if not scores or not idx_image:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64), [], []

        # Hợp nhất điểm & id
        scores, idx_image = merge_searching_results_by_addition(scores, idx_image)

        # Lấy thông tin keyframes (robust)
        scores, idx_image, infos_query, image_paths = self._gather_infos(idx_image, scores)
        return scores, idx_image, infos_query, image_paths

    # ----------------- Reranking -----------------
    def reranking(self, prev_result, lst_pos_vote_idxs, lst_neg_vote_idxs, k):
        """
        Perform reranking using user feedback
        """
        lst_vote_idxs = []
        lst_vote_idxs.extend(lst_pos_vote_idxs or [])
        lst_vote_idxs.extend(lst_neg_vote_idxs or [])
        if not lst_vote_idxs:
            return [], [], [], []

        lst_vote_idxs = np.asarray(lst_vote_idxs, dtype=np.int64)
        len_pos = len(lst_pos_vote_idxs or [])

        # Tập ứng viên ban đầu (từ prev_result)
        result = {}
        for item in prev_result:
            for id_, score in zip(item['video_info']['lst_idxs'], item['video_info']['lst_scores']):
                result[int(id_)] = float(score)

        # Loại bỏ id bị vote âm khỏi tập ứng viên ban đầu
        for key in (lst_neg_vote_idxs or []):
            result.pop(int(key), None)

        if not result:
            return [], [], [], []

        # Giới hạn tìm trong tập ứng viên
        candidate_ids = np.array(list(result.keys()), dtype=np.int64)
        id_selector = faiss.IDSelectorArray(candidate_ids)

        # Query features từ các id vote
        query_feats = self.index_clip.reconstruct_batch(lst_vote_idxs)
        lst_scores, lst_idx_images = self.index_clip.search(
            query_feats, k=min(k, len(candidate_ids)), params=faiss.SearchParametersIVF(sel=id_selector)
        )

        # Cộng/trừ điểm theo vote dương/âm
        for i, (scores, idx_images) in enumerate(zip(lst_scores, lst_idx_images)):
            for score, idx_image in zip(scores, idx_images):
                idx_image = int(idx_image)
                if i < len_pos:
                    result[idx_image] = result.get(idx_image, 0.0) + float(score)
                else:
                    result[idx_image] = result.get(idx_image, 0.0) - float(score)

        # Sắp xếp & trả kết quả
        result_sorted = sorted(result.items(), key=lambda x: x[1], reverse=True)
        list_ids = [item[0] for item in result_sorted]
        lst_scores = [item[1] for item in result_sorted]
        _, list_ids_np, infos_query, list_image_paths = self._gather_infos(list_ids, lst_scores)

        # Đồng bộ lại scores theo list_ids_np đã được lọc
        list_ids_set = set(list_ids_np.tolist())
        lst_scores_filtered = [float(s) for (i, s) in result_sorted if i in list_ids_set]

        return lst_scores_filtered, list_ids_np.tolist(), infos_query, list_image_paths


# ================== Demo giữ nguyên (nếu cần) ==================
def main():
    # Ví dụ: đường dẫn placeholder. Cần thay bằng đường dẫn thật nếu chạy trực tiếp.
    bin_file = 'dict/faiss_cosine.bin'
    json_path = 'dict/keyframes_id.json'
    cosine_faiss = MyFaiss('data/TransNetDatabase/KeyFrames', bin_file, json_path, 'dict/audio2img.json', 'dict/img2audio.json')

    # IMAGE SEARCH
    i_scores, _, infos_query, i_image_paths = cosine_faiss.image_search(id_query=0, k=9)
    print(i_scores, i_image_paths)

    # TEXT SEARCH
    text = ('Người nghệ nhân đang tô màu cho chiếc mặt nạ một cách tỉ mỉ. '
            'Xung quanh ông là rất nhiều những chiếc mặt nạ. '
            'Người nghệ nhân đi đôi dép tổ ong rất giản dị. '
            'Sau đó là hình ảnh quay cận những chiếc mặt nạ. '
            'Loại mặt nạ này được gọi là mặt nạ giấy bồi Trung thu.')
    scores, _, infos_query, image_paths = cosine_faiss.text_search(text, index=None, k=9, model_type='clip')
    print(scores, image_paths)


if __name__ == "__main__":
    main()
