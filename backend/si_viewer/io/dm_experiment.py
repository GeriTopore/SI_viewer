"""What a Digital Micrograph file says about the experiment it belongs to.

GMS saves one acquisition as several files (the survey image, the ADF image
and the spectrum images of a spectrum imaging run, ...). Their tags tie them
together:

- ``ImageTags.Meta Data.Experiment keywords``: entries with an
  ``Experiment ID`` (e.g. "Spectrum Imaging_27/01/2026_15:43:07", shared by
  every file of the run) and a last entry naming what the file is ("Survey",
  "Scan Signal", "EELS", ...);
- ``ImageTags.EELS.Acquisition.Dual acquire sibling.Unique Image ID``: a
  dual EELS spectrum image names its sibling (``ImageList.<k>.UniqueID``);
- the survey's ``DocumentObjectList`` holds the rectangles GMS drew on it:
  where the spectrum image was taken ("Spectrum Image") and the spatial drift
  reference area. A rectangle is (top, left, bottom, right) in pixels.

Only tags are read (nothing for other formats).
"""

from __future__ import annotations

from numbers import Integral

_RECTANGLES = (5, 23)  # rectangle annotation, rectangle ROI


def _om(signal) -> dict:
    try:
        return signal.original_metadata.as_dictionary()
    except Exception:
        return {}


def _image(signal) -> dict:
    """The tags of the file's (first) image."""
    images = _om(signal).get("ImageList") or {}
    for key in sorted(images):
        if isinstance(images[key], dict) and "ImageTags" in images[key]:
            return images[key]
    return {}


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _keywords(signal) -> list[dict]:
    kw = ((_image(signal).get("ImageTags") or {}).get("Meta Data") or {}).get("Experiment keywords") or {}
    return [kw[k] for k in sorted(kw, key=lambda k: int(k[8:]) if k[8:].isdigit() else 0) if isinstance(kw[k], dict)]


def experiment_ids(signal) -> list[str]:
    """The DM experiment IDs a file was recorded under (e.g. its spectrum
    imaging run and in-situ record)."""
    return [i for i in (_text(k.get("Experiment ID")) for k in _keywords(signal)) if i]


def acquisition_label(signal) -> str:
    """What GMS says the file is: "Survey", "Scan Signal", "EELS", ... ("" if
    it doesn't say)."""
    labels = [_text(k.get("Label")) for k in _keywords(signal) if not _text(k.get("Experiment ID"))]
    return next((label for label in reversed(labels) if label), "")


def is_survey(signal) -> bool:
    """The survey image of a spectrum imaging run."""
    title = signal.metadata.General.get_item("title", "") or ""
    return acquisition_label(signal) == "Survey" or "(SI Survey)" in title


def scan_signal_name(signal) -> str:
    """The STEM detector of a scanned image ("ADF", "HAADF", "BF", ...)."""
    tags = _image(signal).get("ImageTags") or {}
    return _text((tags.get("DataBar") or {}).get("Signal Name"))


def _unique_id(value) -> tuple | None:
    if isinstance(value, dict):
        value = tuple(value.get(f"Data{i}") for i in range(4))
    if isinstance(value, (list, tuple)) and len(value) == 4 and all(isinstance(v, Integral) for v in value):
        return tuple(int(v) for v in value)
    return None


def unique_id(signal) -> tuple | None:
    """GMS's id of the image."""
    return _unique_id(_image(signal).get("UniqueID"))


def dual_sibling_id(signal) -> tuple | None:
    """The id of the other spectrum image of a dual EELS acquisition."""
    acq = (((_image(signal).get("ImageTags") or {}).get("EELS") or {}).get("Acquisition") or {})
    return _unique_id((acq.get("Dual acquire sibling") or {}).get("Unique Image ID"))


def _region_kind(label: str) -> str:
    low = label.lower()
    if low.startswith("spectrum image"):
        return "si"
    if "drift" in low:
        return "drift"
    return "other"


def survey_regions(signal) -> list[dict]:
    """The rectangles drawn on a survey image: {label, kind ("si", "drift" or
    "other"), x, y, width, height (pixels)}."""
    docs = _om(signal).get("DocumentObjectList") or {}
    out = []
    for doc in docs.values():
        if not isinstance(doc, dict):
            continue
        for ann in (doc.get("AnnotationGroupList") or {}).values():
            if not isinstance(ann, dict) or ann.get("AnnotationType") not in _RECTANGLES:
                continue
            rect = ann.get("Rectangle")
            if not isinstance(rect, (list, tuple)) or len(rect) != 4:
                continue
            top, left, bottom, right = (float(v) for v in rect)
            if bottom <= top or right <= left:
                continue
            label = _text(ann.get("Label")) or _text(ann.get("Name"))
            out.append({
                "label": label,
                "kind": _region_kind(label),
                "x": left,
                "y": top,
                "width": right - left,
                "height": bottom - top,
            })
    # The spectrum image first.
    return sorted(out, key=lambda r: r["kind"] != "si")
