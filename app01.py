import os
import mimetypes
import copy
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from flask_cors import CORS
from flask import Flask, request, jsonify, send_file, abort
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

# ==== Project utils (giữ nguyên import cũ) ====
from utils.parse_frontend import parse_data
from utils.faiss_processing import MyFaiss
from utils.context_encoding import VisualEncoding
from utils.semantic_embed.tag_retrieval import tag_retrieval
from utils.combine_utils import merge_searching_results_by_addition
from utils.search_utils import group_result_by_video, search_by_filter
from flask import request, has_request_context

# ================= Helpers =================

# Thư mục vật lý chứa keyframes
BASE_KEYFRAMES_DIR = Path(os.environ.get(
    "KEYFRAMES_DIR",
    r"D:\AI HCM CHALLENGE\BE_FE_ver1\frontend\ai\public\data\Keyframes"
)).resolve()

# Base URL backend (để trả URL tuyệt đối cho <img src>)
BACKEND_BASE = os.environ.get("BACKEND_BASE_URL", "http://localhost:5000")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def _parse_keyframe_path(img_path: str):
    """Trả về (data_part, video_id, frame_id_stem) từ đường dẫn keyframe."""
    p = Path(img_path)
    parts = p.parts
    k = None
    for i, seg in enumerate(parts):
        if seg.lower() == "keyframes":
            k = i
            break
    tail = parts[k + 1:] if k is not None else parts
    stem = Path(tail[-1]).stem if tail else p.stem
    if len(tail) >= 3:
        data_part, video_id, frame_id = tail[-3], tail[-2], stem
    else:
        data_part = p.parent.parent.name if p.parent and p.parent.parent else ""
        video_id = p.parent.name if p.parent else ""
        frame_id = p.stem
    return data_part, video_id, frame_id


def _safe_map_frame_id(key: str, frame_id_stem: str, KeyframesMapper):
    """Giữ nguyên nếu không có mapper; chỉ dùng cho shot view."""
    try:
        if KeyframesMapper and key in KeyframesMapper:
            fid = str(int(frame_id_stem))
            return KeyframesMapper[key].get(fid, frame_id_stem)
    except Exception:
        pass
    return frame_id_stem


def _subpath_under_keyframes(img_path: str) -> str:
    """Lấy subpath dưới 'Keyframes' cho ảnh (chuẩn hoá Win/Linux)."""
    p = str(img_path).replace("\\", "/")
    if "Keyframes/" in p:
        return p.split("Keyframes/")[-1]
    parts = [seg for seg in p.strip("/").split("/") if seg]
    return "/".join(parts[-2:])  # fallback


def path_to_url(img_path: str) -> str:
    """Trả về absolute URL khớp đúng host/proto (kể cả khi qua ngrok)."""
    sub = _subpath_under_keyframes(img_path)
    if has_request_context():
        proto = request.headers.get("X-Forwarded-Proto", request.scheme)
        host = request.headers.get("X-Forwarded-Host", request.host)
        base = f"{proto}://{host}"
    else:
        # Fallback khi chạy ngoài request (hầu như không dùng đến)
        base = BACKEND_BASE
    return f"{base.rstrip('/')}/keyframe/{sub}"


def postprocess_result_urls(data_list):
    """Đổi toàn bộ lst_keyframe_paths sang URL để <img src> dùng được."""
    for item in data_list:
        vi = item.get("video_info", {})
        if "lst_keyframe_paths" in vi:
            vi["lst_keyframe_paths"] = [path_to_url(p) for p in vi["lst_keyframe_paths"]]
    return data_list


def enrich_groups_with_meta(groups):
    """Bơm sec & frame_idx (đúng từ id2img_fps.json) vào kết quả nhóm theo video."""
    for g in groups:
        vi = g.get('video_info', {})
        idxs = vi.get('lst_idxs', [])
        secs, frames = [], []
        for sid in idxs:
            info = DictImagePath.get(int(sid))
            secs.append(None if not info else info.get('sec'))
            frames.append(None if not info else info.get('frame_idx'))
        vi['lst_keyframe_secs'] = secs
        vi['lst_frame_numbers'] = frames
    return groups


# Gắn tham số thời gian đơn giản, giữ URL gốc (không đổi watch <-> youtu.be)
YT_TIME_KEYS = {'t', 'start', 'time_continue', 'timestart'}


def build_seek_url(video_url: str, start_sec=None):
    base = str(video_url)
    if start_sec is None:
        return base, None
    # giữ thập phân theo đúng json
    s_str = str(float(start_sec)).rstrip('0').rstrip('.') if '.' in str(start_sec) else str(start_sec)

    low = base.lower()
    scheme, netloc, path, query, frag = urlsplit(base)
        # embed chỉ nhận int
    # xoá tham số thời gian cũ
    q = [(k, v) for (k, v) in parse_qsl(query, keep_blank_values=True) if k not in YT_TIME_KEYS]

    if "youtube.com/embed" in low:
        q.append(("start", str(int(float(s_str)))))
    else:
        q.append(("t", f"{s_str}s"))

    return urlunsplit((scheme, netloc, path, urlencode(q, doseq=True), frag)), s_str


# ================== Config & Objects ==================

json_path = 'dict/id2img_fps.json'
audio_json_path = 'dict/audio_id2img_id.json'
img2audio_json_path = 'dict/img_id2audio_id.json'
scene_path = 'dict/scene_id2info.json'
map_keyframes_path = 'dict/map_keyframes.json'
video_division_path = 'dict/video_division_tag.json'
video_id2img_path = 'dict/video_id2img_id.json'
bin_clip_file = 'dict/faiss_clip_cosine.bin'
bin_clipv2_file = 'dict/faiss_clipv2_cosine.bin'

VisualEncoder = VisualEncoding()
CosineFaiss = MyFaiss(bin_clip_file, bin_clipv2_file, json_path, audio_json_path, img2audio_json_path)
TagRecommendation = tag_retrieval()
DictImagePath = CosineFaiss.id2img_fps
TotalIndexList = np.array(list(range(len(DictImagePath)))).astype('int64')

with open(scene_path, 'r', encoding='utf-8') as f:
    Sceneid2info = json.load(f)
with open(map_keyframes_path, 'r', encoding='utf-8') as f:
    KeyframesMapper = json.load(f)
with open(video_division_path, 'r', encoding='utf-8') as f:
    VideoDivision = json.load(f)
with open(video_id2img_path, 'r', encoding='utf-8') as f:
    Videoid2imgid = json.load(f)


def get_search_space(id):
    search_space = []
    video_space = VideoDivision[f'list_{id}']
    for video_id in video_space:
        search_space.extend(Videoid2imgid[video_id])
    return search_space


SearchSpace = {i: np.array(get_search_space(i)).astype('int64') for i in range(1, 5)}
SearchSpace[0] = TotalIndexList


def get_near_frame(idx):
    image_info = DictImagePath[idx]
    scene_idx = image_info['scene_idx'].split('/')
    return copy.deepcopy(
        Sceneid2info[scene_idx[0]][scene_idx[1]][scene_idx[2]][scene_idx[3]]['lst_keyframe_idxs']
    )


def get_related_ignore(ignore_index):
    total_ignore_index = []
    for idx in ignore_index:
        total_ignore_index.extend(get_near_frame(idx))
    return total_ignore_index


# ========= Helpers for VQA & TRAKE =========

CANDIDATE_TEXT_KEYS = [
    'ocr_text', 'ocr', 'asr_text', 'asr', 'subtitle', 'speech_text', 'text'
]


def _extract_shot_and_meta(idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Lấy shot node và video_info từ Sceneid2info dựa theo img idx."""
    info = DictImagePath[idx]
    col, vid, scene, shot = info['scene_idx'].split('/')
    video_info = Sceneid2info[col][vid]
    shot_node = video_info[scene][shot]
    return shot_node, video_info


def _collect_text_context(shot_node: Dict[str, Any], max_chars: int = 1200) -> str:
    """Gộp OCR/ASR/subtitle nếu có. Không chắc key tồn tại nên thử lần lượt."""
    buf = []
    for k in CANDIDATE_TEXT_KEYS:
        v = shot_node.get(k)
        if not v:
            continue
        if isinstance(v, list):
            # v có thể là list các câu (có hoặc không có timestamp)
            items = []
            for it in v:
                if isinstance(it, dict):
                    # thường dạng {"t": [...], "text": "..."}
                    txt = it.get('text') or it.get('t') or ''
                    if isinstance(txt, list):
                        txt = ' '.join(map(str, txt))
                    items.append(str(txt))
                else:
                    items.append(str(it))
            v = '\n'.join(items)
        else:
            v = str(v)
        if v.strip():
            buf.append(f"[{k}] {v.strip()}")
    text = '\n'.join(buf).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + '…'
    return text


def _merge_segments(segments: List[Dict[str, Any]], gap: float = 2.0) -> List[Dict[str, Any]]:
    """Gộp các khoảng thời gian giao/giáp nhau (khoảng cách < gap)."""
    if not segments:
        return []
    segs = sorted(segments, key=lambda x: (x['start_sec'], x['end_sec']))
    merged = [segs[0]]
    for s in segs[1:]:
        last = merged[-1]
        if s['start_sec'] <= last['end_sec'] + gap:
            # merge
            last['end_sec'] = max(last['end_sec'], s['end_sec'])
            if s.get('score', 0) > last.get('score', 0):
                # thay đại diện tốt hơn
                last['score'] = s['score']
                last['imgid'] = s['imgid']
                last['thumb_url'] = s['thumb_url']
                last['frame_idx'] = s.get('frame_idx')
        else:
            merged.append(s)
    return merged


# ================== Flask ==================
app = Flask(__name__, template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)


@app.after_request
def _add_cors_headers(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    # thêm header ngrok-skip-browser-warning để preflight pass
    resp.headers['Access-Control-Allow-Headers'] = (
        'Content-Type, Authorization, X-Requested-With, ngrok-skip-browser-warning'
    )
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return resp


# ---- Static serving for keyframes ----
@app.route("/keyframe/<path:subpath>")
def serve_keyframe(subpath):
    target = (BASE_KEYFRAMES_DIR / subpath).resolve()
    try:
        target.relative_to(BASE_KEYFRAMES_DIR)
    except Exception:
        return abort(403)
    if not target.exists() or not target.is_file():
        return abort(404)
    mt = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return send_file(str(target), mimetype=mt)


@app.route('/health')
def health():
    return jsonify({'ok': True, 'keyframes_root': str(BASE_KEYFRAMES_DIR)})


@app.route('/data')
def index():
    pagefile, count = [], 0
    for sid, value in DictImagePath.items():
        pagefile.append({'imgpath': path_to_url(value['image_path']), 'id': sid})
        count += 1
        if count >= 500:
            break
    return jsonify({'pagefile': pagefile})


@app.route('/imgsearch')
def image_search():
    k = int(request.args.get('k'))
    id_query = int(request.args.get('imgid'))
    lst_scores, list_ids, _, list_image_paths = CosineFaiss.image_search(id_query, k=k)
    data = group_result_by_video(lst_scores, list_ids, list_image_paths, KeyframesMapper)
    data = enrich_groups_with_meta(data)
    data = postprocess_result_urls(data)
    return jsonify(data)


@app.route('/getquestions', methods=['POST', 'OPTIONS'], strict_slashes=False)
def get_questions():
    if request.method == 'OPTIONS':
        return ('', 204)
    return jsonify([])  # FE setQuestions(res)


@app.route('/getignore', methods=['POST', 'OPTIONS'], strict_slashes=False)
def get_ignore():
    if request.method == 'OPTIONS':
        return ('', 204)
    return jsonify({'data': []})


@app.route('/socket.io/', methods=['GET', 'POST', 'OPTIONS'], strict_slashes=False)
def socketio_stub():
    if request.method == 'OPTIONS':
        return ('', 204)
    # Trả rỗng 200 là đủ để client ngừng báo lỗi, không implement Socket.IO thật
    return ('', 200)


@app.route('/textsearch', methods=['POST', 'OPTIONS'], strict_slashes=False)
def text_search():
    if request.method == 'OPTIONS':
        return ('', 204)
    data = request.get_json(silent=True) or {}

    search_space_index = int(data['search_space'])
    k = int(data['k'])
    clip = data['clip']
    clipv2 = data['clipv2']
    text_query = data['textquery']
    range_filter = int(data['range_filter'])

    index = None
    if data.get('filter'):
        index = np.array(data['id']).astype('int64')
        k = min(k, len(index))

    keep_index = None
    ignore_index = None
    if data.get('ignore'):
        ignore_index = get_related_ignore(np.array(data['ignore_idxs']).astype('int64'))
        keep_index = np.delete(TotalIndexList, ignore_index)

    if keep_index is not None:
        index = np.intersect1d(index, keep_index) if index is not None else keep_index

    index = SearchSpace[search_space_index] if index is None else np.intersect1d(index, SearchSpace[search_space_index])
    k = min(k, len(index))

    if clip and clipv2:
        model_type = 'both'
    elif clip:
        model_type = 'clip'
    else:
        model_type = 'clipv2'

    if data['filtervideo'] != 0:
        mode = data['filtervideo']
        prev_result = data['videos']
        data = search_by_filter(prev_result, text_query, k, mode, model_type, range_filter, ignore_index, keep_index,
                                Sceneid2info, DictImagePath, CosineFaiss, KeyframesMapper)
    else:
        if model_type == 'both':
            scores_clip, list_clip_ids, _, _ = CosineFaiss.text_search(text_query, index=index, k=k, model_type='clip')
            scores_clipv2, list_clipv2_ids, _, _ = CosineFaiss.text_search(text_query, index=index, k=k, model_type='clipv2')
            lst_scores, list_ids = merge_searching_results_by_addition([scores_clip, scores_clipv2],
                                                                       [list_clip_ids, list_clipv2_ids])
            kept_ids, kept_scores, kept_paths = [], [], []
            for sid, sc in zip(list_ids, lst_scores):
                info = DictImagePath.get(int(sid))
                if info and 'image_path' in info:
                    kept_ids.append(int(sid))
                    kept_scores.append(float(sc))
                    kept_paths.append(info['image_path'])
            lst_scores = np.array(kept_scores, dtype=np.float32)
            list_ids = np.array(kept_ids, dtype=np.int64)
            list_image_paths = kept_paths
        else:
            lst_scores, list_ids, _, list_image_paths = CosineFaiss.text_search(
                text_query, index=index, k=k, model_type=model_type
            )
        data = group_result_by_video(lst_scores, list_ids, list_image_paths, KeyframesMapper)

    data = enrich_groups_with_meta(data)
    data = postprocess_result_urls(data)
    return jsonify(data)


@app.route('/panel', methods=['POST', 'OPTIONS'], strict_slashes=False)
def panel():
    if request.method == 'OPTIONS':
        return ('', 204)
    search_items = request.get_json(silent=True) or {}
    k = int(search_items['k'])
    search_space_index = int(search_items['search_space'])

    index = None
    if search_items.get('useid'):
        index = np.array(search_items['id']).astype('int64')
        k = min(k, len(index))

    keep_index = None
    if search_items.get('ignore'):
        ignore_index = get_related_ignore(np.array(search_items['ignore_idxs']).astype('int64'))
        keep_index = np.delete(TotalIndexList, ignore_index)

    if keep_index is not None:
        index = np.intersect1d(index, keep_index) if index is not None else keep_index

    index = SearchSpace[search_space_index] if index is None else np.intersect1d(index, SearchSpace[search_space_index])
    k = min(k, len(index))

    object_input = parse_data(search_items, VisualEncoder)
    ocr_input = None if search_items['ocr'] == "" else search_items['ocr']
    asr_input = None if search_items['asr'] == "" else search_items['asr']

    semantic = False
    keyword = True
    lst_scores, list_ids, _, list_image_paths = CosineFaiss.context_search(
        object_input=object_input, ocr_input=ocr_input, asr_input=asr_input,
        k=k, semantic=semantic, keyword=keyword, index=index, useid=search_items['useid']
    )

    data = group_result_by_video(lst_scores, list_ids, list_image_paths, KeyframesMapper)
    data = enrich_groups_with_meta(data)
    data = postprocess_result_urls(data)
    return jsonify(data)


@app.route('/getrec', methods=['POST', 'OPTIONS'], strict_slashes=False)
def getrec():
    if request.method == 'OPTIONS':
        return ('', 204)
    k = 50
    text_query = request.get_json(silent=True)
    tag_outputs = TagRecommendation(text_query, k)
    return jsonify(tag_outputs)


@app.route('/relatedimg')
def related_img():
    # FE chỉ cần gửi imgid
    id_query = request.args.get('imgid', type=int)
    if id_query is None:
        return jsonify({})
    image_info = DictImagePath[id_query]           # có 'sec' & 'frame_idx'
    image_path = image_info['image_path']
    keyframe_sec = image_info.get('sec')

    scene_idx = image_info['scene_idx'].split('/')
    video_info = copy.deepcopy(Sceneid2info[scene_idx[0]][scene_idx[1]])
    video_url = video_info['video_metadata']['watch_url']
    shot_time = video_info[scene_idx[2]][scene_idx[3]]['shot_time']

    near_keyframes = [path_to_url(p) for p in video_info[scene_idx[2]][scene_idx[3]]['lst_keyframe_paths']]
    try:
        near_keyframes.remove(path_to_url(image_path))
    except ValueError:
        pass

    seek_url, sec_sent = build_seek_url(video_url, start_sec=keyframe_sec)

    return jsonify({
        'video_url': video_url,
        'video_url_seek': seek_url,
        'start_sec': sec_sent,
        'keyframe_sec': keyframe_sec,
        'frame_idx': image_info.get('frame_idx'),
        'video_range': shot_time,
        'near_keyframes': near_keyframes
    })


@app.route('/getvideoshot')
def get_video_shot():
    if request.args.get('imgid') == 'undefined':
        return jsonify({})
    id_query = int(request.args.get('imgid'))
    image_info = DictImagePath[id_query]
    scene_idx = image_info['scene_idx'].split('/')
    shots = copy.deepcopy(Sceneid2info[scene_idx[0]][scene_idx[1]][scene_idx[2]])

    selected_shot = int(scene_idx[3])
    total_n_shots = len(shots)
    new_shots = {}
    for select_id in range(max(0, selected_shot-5), min(selected_shot+6, total_n_shots)):
        new_shots[str(select_id)] = shots[str(select_id)]
    shots = new_shots

    for shot_key in list(shots.keys()):
        lst_keyframe_idxs = []
        url_paths = []
        for img_path in shots[shot_key]['lst_keyframe_paths']:
            data_part, video_id, frame_id = _parse_keyframe_path(img_path)
            key = f'{data_part}_{video_id}'.replace('_extra', '')
            if 'extra' not in data_part:
                frame_id = _safe_map_frame_id(key, frame_id, KeyframesMapper)
            try:
                frame_id_int = int(frame_id)
            except Exception:
                frame_id_int = frame_id
            lst_keyframe_idxs.append(frame_id_int)
            url_paths.append(path_to_url(img_path))

        shots[shot_key]['lst_idxs'] = shots[shot_key]['lst_keyframe_idxs']
        shots[shot_key]['lst_keyframe_secs'] = [DictImagePath[idx].get('sec') for idx in shots[shot_key]['lst_idxs']]
        shots[shot_key]['lst_frame_numbers'] = [DictImagePath[idx].get('frame_idx') for idx in shots[shot_key]['lst_idxs']]
        shots[shot_key]['lst_keyframe_idxs'] = lst_keyframe_idxs
        shots[shot_key]['lst_keyframe_paths'] = url_paths

    return jsonify({
        'collection': scene_idx[0],
        'video_id': scene_idx[1],
        'shots': shots,
        'selected_shot': scene_idx[3]
    })


@app.route('/feedback', methods=['POST', 'OPTIONS'], strict_slashes=False)
def feed_back():
    if request.method == 'OPTIONS':
        return ('', 204)
    data = request.get_json(silent=True) or {}
    k = int(data['k'])
    prev_result = data['videos']
    lst_pos_vote_idxs = data['lst_pos_idxs']
    lst_neg_vote_idxs = data['lst_neg_idxs']
    lst_scores, list_ids, _, list_image_paths = CosineFaiss.reranking(prev_result, lst_pos_vote_idxs, lst_neg_vote_idxs, k)
    data = group_result_by_video(lst_scores, list_ids, list_image_paths, KeyframesMapper)
    data = enrich_groups_with_meta(data)
    data = postprocess_result_urls(data)
    return jsonify(data)


@app.route('/translate', methods=['POST', 'OPTIONS'], strict_slashes=False)
def translate():
    if request.method == 'OPTIONS':
        return ('', 204)
    data = request.get_json(silent=True) or {}
    text_query = data['textquery']
    text_query_translated = CosineFaiss.translater(text_query)
    return jsonify(text_query_translated)


# ================== NEW ROUTES ==================
# ---- 1) /kis: Textual Known-Item Search ưu tiên keyword (OCR/ASR) ----
@app.route('/kis', methods=['POST', 'OPTIONS'], strict_slashes=False)
def kis_search():
    if request.method == 'OPTIONS':
        return ('', 204)
    payload = request.get_json(silent=True) or {}
    text_query: str = payload.get('textquery', '')
    k: int = int(payload.get('k', 50))
    search_space_index: int = int(payload.get('search_space', 0))
    use_semantic: bool = bool(payload.get('semantic', False))  # mặc định KIS = keyword-only

    index = SearchSpace.get(search_space_index, TotalIndexList)
    k = min(k, len(index))

    # KIS: keyword ưu tiên, semantic tuỳ chọn để "cứu" khi thiếu dữ liệu
    lst_scores, list_ids, _, list_image_paths = CosineFaiss.context_search(
        object_input=None, ocr_input=text_query, asr_input=text_query,
        k=k, semantic=use_semantic, keyword=True, index=index, useid=False
    )
    data = group_result_by_video(lst_scores, list_ids, list_image_paths, KeyframesMapper)
    data = enrich_groups_with_meta(data)
    data = postprocess_result_urls(data)
    return jsonify({
        'query': text_query,
        'semantic': use_semantic,
        'search_space': search_space_index,
        'results': data
    })


# ---- 2) /trake: Temporal Retrieval & Alignment of Key Events ----
@app.route('/trake', methods=['POST', 'OPTIONS'], strict_slashes=False)
def trake():
    if request.method == 'OPTIONS':
        return ('', 204)
    payload = request.get_json(silent=True) or {}
    text_query: str = payload.get('textquery', '')
    k: int = int(payload.get('k', 100))
    search_space_index: int = int(payload.get('search_space', 0))
    video_key: Optional[str] = payload.get('video_key')  # nếu muốn giới hạn 1 video (key y như JSON)
    min_gap_sec: float = float(payload.get('min_gap_sec', 2.0))

    # Xác định index
    if video_key:
        index = np.array(Videoid2imgid.get(video_key, []), dtype='int64')
        if index.size == 0:
            return jsonify({'error': f'video_key {video_key} không tồn tại hoặc không có keyframe'}), 400
    else:
        index = SearchSpace.get(search_space_index, TotalIndexList)

    k = min(k, len(index))

    # Dùng CLIP-V2 cho độ nhạy ngữ nghĩa, nếu cần có thể gộp với CLIP như /textsearch
    lst_scores, list_ids, _, _ = CosineFaiss.text_search(text_query, index=index, k=k, model_type='clipv2')

    # Gom theo video & dựng timeline bằng shot_time
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for sid, sc in zip(list_ids.tolist(), lst_scores.tolist()):
        info = DictImagePath[int(sid)]
        col, vid, scene, shot = info['scene_idx'].split('/')
        video_info = Sceneid2info[col][vid]
        shot_node = video_info[scene][shot]
        s, e = shot_node['shot_time']
        img_url = path_to_url(info['image_path'])
        key = (col, vid)
        if key not in groups:
            groups[key] = {
                'collection': col,
                'video_id': vid,
                'video_url': video_info['video_metadata']['watch_url'],
                'segments': []
            }
        groups[key]['segments'].append({
            'start_sec': float(s), 'end_sec': float(e), 'score': float(sc),
            'imgid': int(sid), 'thumb_url': img_url, 'frame_idx': info.get('frame_idx')
        })

    # Merge & build seek URLs
    out_videos = []
    for (col, vid), obj in groups.items():
        url = obj['video_url']
        merged = _merge_segments(obj['segments'], gap=min_gap_sec)
        for seg in merged:
            seek_url, _ = build_seek_url(url, start_sec=seg['start_sec'])
            seg['seek_url'] = seek_url
        out_videos.append({
            'collection': col,
            'video_id': vid,
            'video_url': url,
            'timeline': merged
        })

    return jsonify({
        'query': text_query,
        'search_space': search_space_index,
        'video_key': video_key,
        'videos': out_videos
    })


# ---- 3) /vqa: Visual Question Answering trên keyframe/shot ----
# Mặc định: trả về các keyframe liên quan + context OCR/ASR. Nếu có OPENAI_API_KEY thì sinh câu trả lời.
@app.route('/vqa', methods=['POST', 'OPTIONS'], strict_slashes=False)
def vqa():
    if request.method == 'OPTIONS':
        return ('', 204)
    payload = request.get_json(silent=True) or {}
    question: str = payload.get('question', '')
    topk: int = int(payload.get('k', 10))
    imgid: Optional[int] = payload.get('imgid')
    video_key: Optional[str] = payload.get('video_key')  # nếu chọn video nhưng chưa chọn khung hình
    strategy: str = (payload.get('strategy') or 'retrieval').lower()  # 'retrieval' | 'openai'

    # Xác định index ứng viên để truy xuất ngữ cảnh
    if imgid is not None:
        try:
            # ưu tiên các keyframe lân cận cùng shot
            index = np.array(get_near_frame(int(imgid)), dtype='int64')
        except Exception:
            index = np.array([int(imgid)], dtype='int64')
    elif video_key:
        index = np.array(Videoid2imgid.get(video_key, []), dtype='int64')
    else:
        index = TotalIndexList

    if index.size == 0:
        return jsonify({'error': 'Không có index ứng viên cho VQA'}), 400

    k = min(topk, len(index))

    # Lấy các khung hình/phân đoạn sát nghĩa với câu hỏi
    lst_scores, list_ids, _, _ = CosineFaiss.text_search(question, index=index, k=k, model_type='clipv2')

    candidates = []
    for sid, sc in zip(list_ids.tolist(), lst_scores.tolist()):
        info = DictImagePath[int(sid)]
        shot_node, video_info = _extract_shot_and_meta(int(sid))
        s, e = shot_node['shot_time']
        video_url = video_info['video_metadata']['watch_url']
        seek_url, _ = build_seek_url(video_url, start_sec=s)
        context_text = _collect_text_context(shot_node)
        candidates.append({
            'imgid': int(sid),
            'score': float(sc),
            'image_url': path_to_url(info['image_path']),
            'frame_idx': info.get('frame_idx'),
            'sec': info.get('sec'),
            'shot_time': [float(s), float(e)],
            'seek_url': seek_url,
            'context_text': context_text,
        })

    answer = None
    reasoning = None

    if strategy == 'openai' and OPENAI_API_KEY:
        # Gọi OpenAI nếu có key, dùng context top-3
        ctx = []
        for c in candidates[:3]:
            piece = f"Time {c['shot_time'][0]}-{c['shot_time'][1]}s\nContext: {c['context_text']}".strip()
            if piece:
                ctx.append(piece)
        ctx_text = "\n\n".join(ctx).strip() or "(no OCR/ASR context)"
        try:
            # dùng requests để không phụ thuộc phiên bản openai lib
            import requests
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            payload_oa = {
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": "You answer questions about a video shot using only the provided context. If unknown, say you are not sure succinctly."},
                    {"role": "user", "content": f"Question: {question}\n\nContext from OCR/ASR and metadata (may be partial):\n{ctx_text}\n\nAnswer in one or two sentences."}
                ],
                "temperature": 0.2,
            }
            r = requests.post("https://api.openai.com/v1/chat/completions", json=payload_oa, headers=headers, timeout=20)
            data = r.json()
            answer = (data.get('choices') or [{}])[0].get('message', {}).get('content')
            reasoning = 'llm'
        except Exception as e:
            answer = None
            reasoning = f"openai_error: {e}"

    # Nếu chưa có answer, fallback: trả về gợi ý keyframe + context để FE hiển thị
    return jsonify({
        'question': question,
        'strategy': strategy,
        'answer': answer,
        'reasoning': reasoning,
        'candidates': candidates
    })


# Running app
if __name__ == '__main__':
    print(f"[KEYFRAMES_DIR] Serving from: {BASE_KEYFRAMES_DIR}")
    print(f"[BACKEND_BASE] {BACKEND_BASE}")
    print("New endpoints: /kis (POST), /vqa (POST), /trake (POST), /health (GET)")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
