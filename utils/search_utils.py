import os
import copy
from pathlib import Path
import numpy as np
from utils.combine_utils import merge_searching_results_by_addition

# --------- Helpers chuẩn hoá & tách đường dẫn ---------
def _parse_keyframe_path(image_path: str):
    """
    Trả về (data_part, video_id, frame_id_stem) từ đường dẫn keyframe.
    Tự phát hiện đoạn sau thư mục 'KeyFrames' (không phân biệt hoa thường).
    Hoạt động trên cả Windows ('\\') và Linux ('/'), hỗ trợ mọi đuôi (.jpg/.jpeg/.png/.webp...).
    """
    if not image_path:
        return "", "", ""

    p = Path(image_path)
    parts = p.parts
    # tìm vị trí 'KeyFrames' (case-insensitive)
    k = None
    for i, seg in enumerate(parts):
        if seg.lower() == "keyframes":
            k = i
            break

    # phần đuôi sau 'KeyFrames'
    tail = parts[k + 1:] if k is not None else parts

    # kỳ vọng tối thiểu [data_part, video_id, file]
    if len(tail) >= 3:
        data_part = tail[-3]
        video_id = tail[-2]
        frame_id_stem = Path(tail[-1]).stem
    else:
        # fallback: lấy 3 phần cuối của toàn path
        if len(parts) >= 3:
            data_part = parts[-3]
            video_id = parts[-2]
            frame_id_stem = Path(parts[-1]).stem
        else:
            data_part, video_id, frame_id_stem = "", "", Path(image_path).stem

    return data_part, video_id, frame_id_stem


def _safe_map_frame_id(key: str, frame_id_stem: str, KeyframesMapper):
    """
    Nếu có KeyframesMapper và key tồn tại, thử map frame_id (dạng số).
    Nếu không map được, trả lại frame_id gốc.
    """
    try:
        if KeyframesMapper and key in KeyframesMapper:
            # Một số mapper dùng key là str(int(frame_id))
            fid = str(int(frame_id_stem))
            return KeyframesMapper[key].get(fid, frame_id_stem)
    except Exception:
        pass
    return frame_id_stem


# ----------------- HÀM ĐÃ SỬA -----------------
def group_result_by_video(lst_scores, list_ids, list_image_paths, KeyframesMapper):
    result_dict = dict()

    for i, image_path in enumerate(list_image_paths):
        data_part, video_id, frame_id = _parse_keyframe_path(image_path)
        key = f"{data_part}_{video_id}".replace("_extra", "")

        if "extra" not in data_part:
            frame_id = _safe_map_frame_id(key, frame_id, KeyframesMapper)

        # bảo đảm kiểu int nếu là số, nếu không để nguyên (tránh crash)
        try:
            frame_id_int = int(frame_id)
        except Exception:
            frame_id_int = frame_id  # giữ nguyên string

        if key not in result_dict:
            result_dict[key] = {
                "lst_keyframe_paths": [],
                "lst_idxs": [],
                "lst_keyframe_idxs": [],
                "lst_scores": [],
            }

        result_dict[key]["lst_keyframe_paths"].append(image_path)
        result_dict[key]["lst_idxs"].append(int(list_ids[i]))
        result_dict[key]["lst_keyframe_idxs"].append(frame_id_int)
        result_dict[key]["lst_scores"].append(float(lst_scores[i]))

    result = [{"video_id": key, "video_info": value} for key, value in result_dict.items()]
    result = sorted(result, key=lambda x: x["video_info"]["lst_scores"][0], reverse=True)
    return result


def search_by_filter(
    prev_result,
    text_query,
    k,
    mode,
    model_type,
    range_filter,
    ignore_index,
    keep_index,
    Sceneid2info,
    DictImagePath,
    CosineFaiss,
    KeyframesMapper,
):
    # --- gom các index cần ignore theo video ---
    ignore_videos = None
    if ignore_index is not None:
        ignore_videos = dict()
        for idx in ignore_index:
            image_path = DictImagePath[idx]["image_path"]
            data_part, video_id, frame_id = _parse_keyframe_path(image_path)
            key = f"{data_part}_{video_id}".replace("_extra", "")
            ignore_videos.setdefault(key, []).append(idx)

    filter_idx = []
    result_dict = dict()

    for item in prev_result:
        key = item["video_id"]
        ignore_video = ignore_videos.get(key) if ignore_videos is not None else None

        if key not in result_dict:
            result_dict[key] = {
                "video_info": {
                    "lst_keyframe_paths": [],
                    "lst_idxs": [],
                    "lst_keyframe_idxs": [],
                    "lst_scores": [],
                },
                "video_info_prev": item["video_info"],
            }

        lst_idxs = item["video_info"]["lst_idxs"]
        lst_shots = []
        for idx in lst_idxs:
            scene_idx = DictImagePath[idx]["scene_idx"]  # định dạng do dữ liệu quy ước (chuỗi có '/')
            if scene_idx in lst_shots:
                continue
            if ignore_video is not None and idx in ignore_video:
                continue

            lst_shots.append(scene_idx)
            scene_parts = scene_idx.split("/")  # scene_idx thường là "vid/xxx/yyy/zzz"
            video_info = copy.deepcopy(Sceneid2info[scene_parts[0]][scene_parts[1]][scene_parts[2]])

            if mode == 1:
                start, end = int(scene_parts[3]) + 1, int(scene_parts[3]) + range_filter
            else:
                start, end = int(scene_parts[3]) - range_filter, int(scene_parts[3]) - 1

            for i_shot in range(start, end):
                if 0 <= i_shot < len(video_info):
                    filter_idx.extend(video_info[str(i_shot)]["lst_keyframe_idxs"])

    # unique + kiểu int64
    filter_idx = np.array(sorted(set(filter_idx)), dtype="int64")

    if keep_index is not None:
        filter_idx = np.intersect1d(filter_idx, keep_index)

    if filter_idx.size == 0:
        return []  # không có gì để tìm

    k = min(k, int(filter_idx.size))

    # --- truy vấn ---
    if model_type == "both":
        scores_clip, list_clip_ids, _, _ = CosineFaiss.text_search(text_query, index=filter_idx, k=k, model_type="clip")
        scores_clipv2, list_clipv2_ids, _, _ = CosineFaiss.text_search(text_query, index=filter_idx, k=k, model_type="clipv2")
        lst_scores, list_ids = merge_searching_results_by_addition(
            [scores_clip, scores_clipv2],
            [list_clip_ids, list_clipv2_ids],
        )
        infos_query = list(map(CosineFaiss.id2img_fps.get, list(list_ids)))
        list_image_paths = [info["image_path"] for info in infos_query]
    else:
        lst_scores, list_ids, _, list_image_paths = CosineFaiss.text_search(
            text_query, index=filter_idx, k=k, model_type=model_type
        )

    # --- gom theo video ---
    for i, image_path in enumerate(list_image_paths):
        data_part, video_id, frame_id = _parse_keyframe_path(image_path)
        key = f"{data_part}_{video_id}".replace("_extra", "")

        if "extra" not in data_part:
            frame_id = _safe_map_frame_id(key, frame_id, KeyframesMapper)

        try:
            frame_id_int = int(frame_id)
        except Exception:
            frame_id_int = frame_id

        result_dict[key]["video_info"]["lst_keyframe_paths"].append(image_path)
        result_dict[key]["video_info"]["lst_idxs"].append(int(list_ids[i]))
        result_dict[key]["video_info"]["lst_keyframe_idxs"].append(frame_id_int)
        result_dict[key]["video_info"]["lst_scores"].append(float(lst_scores[i]))

    result = []
    for key, value in result_dict.items():
        if value["video_info"]["lst_keyframe_paths"]:
            result.append(
                {
                    "video_id": key,
                    "video_info": value["video_info"],
                    "video_info_prev": value["video_info_prev"],
                }
            )

    # sort theo tổng điểm hiện tại + trước đó
    result = sorted(
        result,
        key=lambda x: x["video_info"]["lst_scores"][0] + x["video_info_prev"]["lst_scores"][0],
        reverse=True,
    )
    return result
