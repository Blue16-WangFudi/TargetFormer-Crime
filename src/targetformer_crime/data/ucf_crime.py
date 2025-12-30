from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

VIDEO_EXTS_DEFAULT = (".mp4", ".avi", ".mkv", ".mov", ".m4v")
IMAGE_EXTS_DEFAULT = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class MediaRecord:
    split: str
    category: str
    label: int  # 0 normal, 1 abnormal
    uid: str
    storage: str  # "video" | "frames"
    video_path: Optional[Path] = None
    frames_dir: Optional[Path] = None
    frames_prefix: Optional[str] = None
    frame_ext: Optional[str] = None
    num_frames: Optional[int] = None  # frames storage: count of image files
    frame_idx_min: Optional[int] = None  # frames storage: parsed from filenames
    frame_idx_max: Optional[int] = None  # frames storage: parsed from filenames
    sample_path: Optional[Path] = None  # frames storage: one sample image path

    def key(self) -> str:
        return f"{self.split}/{self.category}/{self.uid}"


@dataclass(frozen=True)
class UcfCrimeIndex:
    root: Path
    train_dir: Path
    test_dir: Path
    normal_dirname: str
    records: Sequence[MediaRecord]

    @property
    def train_records(self) -> List[MediaRecord]:
        return [r for r in self.records if r.split.lower() == "train"]

    @property
    def test_records(self) -> List[MediaRecord]:
        return [r for r in self.records if r.split.lower() == "test"]


def _autodetect_root_with_splits(datasets_root: Path, train_dirname: str, test_dirname: str) -> Path:
    datasets_root = Path(datasets_root)
    train_dir = datasets_root / train_dirname
    test_dir = datasets_root / test_dirname
    if train_dir.exists() and test_dir.exists():
        return datasets_root

    # Try one-level nested: /datasets/<something>/{Train,Test}
    for sub in datasets_root.iterdir():
        if not sub.is_dir():
            continue
        train_dir = sub / train_dirname
        test_dir = sub / test_dirname
        if train_dir.exists() and test_dir.exists():
            return sub

    return datasets_root


def _detect_storage_type(split_dir: Path, video_exts: Sequence[str], image_exts: Sequence[str]) -> str:
    video_exts_l = {e.lower() for e in video_exts}
    image_exts_l = {e.lower() for e in image_exts}

    checked_dirs = 0
    for category_dir in split_dir.iterdir():
        if not category_dir.is_dir():
            continue
        checked_dirs += 1
        try:
            with os.scandir(category_dir) as it:
                for i, entry in enumerate(it):
                    if i >= 200:
                        break
                    if not entry.is_file():
                        continue
                    ext = Path(entry.name).suffix.lower()
                    if ext in video_exts_l:
                        return "video"
                    if ext in image_exts_l:
                        return "frames"
        except FileNotFoundError:
            continue
        if checked_dirs >= 3:
            break

    # Default: classic release uses videos.
    return "video"


def _iter_video_files_one_level(split_dir: Path, video_exts: Sequence[str]) -> Iterable[Tuple[str, str, Path]]:
    exts = {e.lower() for e in video_exts}
    for category_dir in split_dir.iterdir():
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        try:
            with os.scandir(category_dir) as it:
                for entry in it:
                    if not entry.is_file():
                        continue
                    p = Path(entry.path)
                    if p.suffix.lower() in exts:
                        yield category, p.stem, p
        except FileNotFoundError:
            continue


def _parse_frame_name(name: str, image_exts_l: set[str]) -> Optional[Tuple[str, int, str]]:
    p = Path(name)
    ext = p.suffix.lower()
    if ext not in image_exts_l:
        return None
    stem = p.stem
    if "_" not in stem:
        return None
    prefix, idx_str = stem.rsplit("_", 1)
    if not idx_str.isdigit():
        return None
    return prefix, int(idx_str), ext


def _scan_frame_category_full(category_dir: Path, image_exts_l: set[str]) -> Dict[str, Dict[str, object]]:
    # One pass over all frames in a category folder.
    stats: Dict[str, Dict[str, object]] = {}
    with os.scandir(category_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            parsed = _parse_frame_name(entry.name, image_exts_l=image_exts_l)
            if parsed is None:
                continue
            prefix, idx, ext = parsed
            st = stats.get(prefix)
            if st is None:
                st = {
                    "count": 0,
                    "ext": ext,
                    "sample": Path(entry.path),
                    "min_idx": int(idx),
                    "max_idx": int(idx),
                }
                stats[prefix] = st
            st["count"] = int(st["count"]) + 1
            st["min_idx"] = int(min(int(st.get("min_idx", idx)), int(idx)))
            st["max_idx"] = int(max(int(st.get("max_idx", idx)), int(idx)))
    return stats


def _iter_frame_prefixes_limited(category_dir: Path, image_exts_l: set[str], max_prefixes: int) -> Iterable[Tuple[str, str, Path]]:
    # Yield unique prefixes (with ext and a sample path) until max_prefixes reached.
    seen = set()
    with os.scandir(category_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            parsed = _parse_frame_name(entry.name, image_exts_l=image_exts_l)
            if parsed is None:
                continue
            prefix, _, ext = parsed
            if prefix in seen:
                continue
            seen.add(prefix)
            yield prefix, ext, Path(entry.path)
            if len(seen) >= max_prefixes:
                return


def discover_ucf_crime(
    datasets_root: Path,
    train_dirname: str = "Train",
    test_dirname: str = "Test",
    normal_dirname: str = "NormalVideos",
    video_exts: Optional[Sequence[str]] = None,
    image_exts: Optional[Sequence[str]] = None,
    max_videos: Optional[int] = None,
) -> UcfCrimeIndex:
    datasets_root = _autodetect_root_with_splits(Path(datasets_root), train_dirname=train_dirname, test_dirname=test_dirname)
    video_exts = list(video_exts or VIDEO_EXTS_DEFAULT)
    image_exts = list(image_exts or IMAGE_EXTS_DEFAULT)
    image_exts_l = {e.lower() for e in image_exts}

    train_dir = datasets_root / train_dirname
    test_dir = datasets_root / test_dirname
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"UCF-Crime root not found. Expected {train_dir} and {test_dir}. Override with TFC_DATASETS_ROOT."
        )

    storage_train = _detect_storage_type(train_dir, video_exts=video_exts, image_exts=image_exts)
    storage_test = _detect_storage_type(test_dir, video_exts=video_exts, image_exts=image_exts)
    storage = "video" if (storage_train == "video" or storage_test == "video") else "frames"

    records: List[MediaRecord] = []
    for split, split_dir in [("Train", train_dir), ("Test", test_dir)]:
        split_count = 0
        if storage == "video":
            for category, uid, video_path in _iter_video_files_one_level(split_dir, video_exts=video_exts):
                label = 0 if category.lower() == normal_dirname.lower() else 1
                records.append(
                    MediaRecord(
                        split=split,
                        category=category,
                        label=label,
                        uid=uid,
                        storage="video",
                        video_path=video_path,
                    )
                )
                split_count += 1
                if max_videos and split_count >= int(max_videos):
                    break
        else:
            category_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            category_dirs.sort(key=lambda p: (0 if p.name.lower() == normal_dirname.lower() else 1, p.name))

            for category_dir in category_dirs:
                if not category_dir.is_dir():
                    continue
                category = category_dir.name
                label = 0 if category.lower() == normal_dirname.lower() else 1

                if max_videos:
                    if split_count >= int(max_videos):
                        break
                    # Smoke/limited scan: take at most 1 video per category to
                    # ensure both normal/abnormal are covered.
                    gen = _iter_frame_prefixes_limited(category_dir, image_exts_l=image_exts_l, max_prefixes=1)
                    first = next(gen, None)
                    if first is None:
                        continue
                    prefix, ext, sample = first
                    records.append(
                        MediaRecord(
                            split=split,
                            category=category,
                            label=label,
                            uid=prefix,
                            storage="frames",
                            frames_dir=category_dir,
                            frames_prefix=prefix,
                            frame_ext=ext,
                            num_frames=None,
                            sample_path=sample,
                        )
                    )
                    split_count += 1
                else:
                    stats = _scan_frame_category_full(category_dir, image_exts_l=image_exts_l)
                    for prefix in sorted(stats.keys()):
                        st = stats[prefix]
                        records.append(
                            MediaRecord(
                                split=split,
                                category=category,
                                label=label,
                                uid=prefix,
                                storage="frames",
                                frames_dir=category_dir,
                                frames_prefix=prefix,
                                frame_ext=str(st["ext"]),
                                num_frames=int(st["count"]),
                                frame_idx_min=int(st.get("min_idx")) if "min_idx" in st else None,
                                frame_idx_max=int(st.get("max_idx")) if "max_idx" in st else None,
                                sample_path=Path(str(st["sample"])),
                            )
                        )

    records = sorted(records, key=lambda r: (r.split, r.category, r.uid))
    return UcfCrimeIndex(
        root=datasets_root,
        train_dir=train_dir,
        test_dir=test_dir,
        normal_dirname=normal_dirname,
        records=records,
    )
