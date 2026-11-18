from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st
import torch
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "best.pt"
DEFAULT_IMAGE_SIZE = 640


TRANSLATIONS = {
    "English": {
        "hero_kicker": "Computer Vision · YOLO26s",
        "title": "Road Damage Detector",
        "intro": (
            "Upload a road image to identify potholes, cracks, debris, manholes, "
            "speed breakers, and road barricades. The trained model runs once the "
            "image is uploaded and returns an annotated result immediately."
        ),
        "card_classes_title": "6 detection classes",
        "card_classes_text": "Road damage and road hazards",
        "card_model_title": "YOLO26s checkpoint",
        "card_model_text": "Automatically selected best.pt",
        "card_privacy_title": "Privacy-conscious",
        "card_privacy_text": "Images are processed only for inference",
        "language": "Language / 语言",
        "settings": "Detection settings",
        "confidence": "Confidence threshold",
        "confidence_help": "Lower values show more detections but may increase false positives.",
        "iou": "IoU threshold",
        "iou_help": "Controls suppression of overlapping bounding boxes.",
        "classes_heading": "Recognized classes",
        "classes_list": "Pothole · Crack · Debris · Manhole · Speed breaker · Road barricade",
        "academic_note": (
            "This application is an academic demonstration. Predictions should be "
            "reviewed before being used for maintenance or safety decisions."
        ),
        "upload": "Upload a road image",
        "upload_help": "Supported formats: JPG, JPEG, PNG, BMP, and WEBP.",
        "upload_prompt": "Upload an image above to start detection.",
        "invalid_image": "The uploaded file could not be read as a valid image.",
        "large_image": "This image is too large. Please upload an image below 40 megapixels.",
        "spinner": "Loading the model and analyzing the image...",
        "model_missing": "The model file is unavailable. Check that models/best.pt exists.",
        "prediction_failed": "Prediction failed. Check the application logs for details.",
        "detections": "Detections",
        "detected_classes": "Detected classes",
        "highest_confidence": "Highest confidence",
        "inference_time": "Inference time",
        "prediction_result": "Prediction result",
        "original_image": "Original image",
        "model_output": "Model output",
        "no_detection": (
            "No road damage or hazard was detected at the selected confidence threshold. "
            "Try a lower threshold, or review the image manually."
        ),
        "detected_summary": "Detected {objects} object(s) across {classes} class(es).",
        "details": "Detection details",
        "column_detection": "Detection",
        "column_class": "Class",
        "column_confidence": "Confidence",
        "column_left": "Left",
        "column_top": "Top",
        "column_right": "Right",
        "column_bottom": "Bottom",
        "class_totals": "Class totals",
        "download": "Download annotated result",
        "footer": (
            "Group F road-damage detection project · Model predictions are informational "
            "and may include false positives or false negatives."
        ),
    },
    "中文": {
        "hero_kicker": "计算机视觉 · YOLO26s",
        "title": "道路损伤检测系统",
        "intro": (
            "上传道路图片，识别坑洞、裂缝、道路碎片、井盖、减速带和道路路障。"
            "图片上传后，训练好的模型会立即完成推理并返回带检测框的结果。"
        ),
        "card_classes_title": "6 类检测目标",
        "card_classes_text": "道路损伤与道路障碍物",
        "card_model_title": "YOLO26s 模型权重",
        "card_model_text": "验证集自动选择的 best.pt",
        "card_privacy_title": "仅用于推理",
        "card_privacy_text": "上传图片仅用于本次模型检测",
        "language": "Language / 语言",
        "settings": "检测参数",
        "confidence": "置信度阈值",
        "confidence_help": "降低阈值会显示更多检测结果，但也可能增加误检。",
        "iou": "IoU 阈值",
        "iou_help": "控制重叠检测框的抑制程度。",
        "classes_heading": "可识别类别",
        "classes_list": "坑洞 · 裂缝 · 道路碎片 · 井盖 · 减速带 · 道路路障",
        "academic_note": "本应用为学术项目演示。用于维护或安全决策前，请人工复核模型结果。",
        "upload": "上传道路图片",
        "upload_help": "支持 JPG、JPEG、PNG、BMP 和 WEBP 格式。",
        "upload_prompt": "请上传图片以开始检测。",
        "invalid_image": "无法将上传的文件读取为有效图片。",
        "large_image": "图片尺寸过大，请上传低于 4000 万像素的图片。",
        "spinner": "正在加载模型并分析图片……",
        "model_missing": "模型文件不可用，请检查 models/best.pt 是否存在。",
        "prediction_failed": "预测失败，请查看应用日志了解详细信息。",
        "detections": "检测目标数",
        "detected_classes": "检出类别数",
        "highest_confidence": "最高置信度",
        "inference_time": "推理时间",
        "prediction_result": "预测结果",
        "original_image": "原始图片",
        "model_output": "模型输出",
        "no_detection": "当前置信度阈值下未检测到道路损伤或障碍物。可尝试降低阈值，或人工检查图片。",
        "detected_summary": "共检测到 {objects} 个目标，涉及 {classes} 个类别。",
        "details": "检测详情",
        "column_detection": "序号",
        "column_class": "类别",
        "column_confidence": "置信度",
        "column_left": "左边界",
        "column_top": "上边界",
        "column_right": "右边界",
        "column_bottom": "下边界",
        "class_totals": "分类统计",
        "download": "下载标注结果",
        "footer": "Group F 道路损伤检测项目 · 模型预测仅供参考，可能出现误检或漏检。",
    },
}


CLASS_NAMES = {
    "English": {
        "pothole": "Pothole",
        "crack": "Crack",
        "debris": "Debris",
        "manhole": "Manhole",
        "speed_breaker": "Speed breaker",
        "road_barricade": "Road barricade",
    },
    "中文": {
        "pothole": "坑洞",
        "crack": "裂缝",
        "debris": "道路碎片",
        "manhole": "井盖",
        "speed_breaker": "减速带",
        "road_barricade": "道路路障",
    },
}


st.set_page_config(
    page_title="Road Damage Detector | 道路损伤检测",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        :root {
            --road-ink: #132238;
            --road-muted: #5f6f82;
            --road-accent: #ee6c4d;
            --road-blue: #2463a8;
            --road-panel: #f5f7fa;
        }
        .stApp { background: #ffffff; }
        .block-container { max-width: 1220px; padding-top: 2rem; }
        .hero {
            border: 1px solid #dfe6ee;
            border-radius: 18px;
            padding: 1.7rem 1.9rem;
            background: linear-gradient(135deg, #f7f9fc 0%, #edf4fb 100%);
            margin-bottom: 1.4rem;
        }
        .hero-kicker {
            color: var(--road-accent);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }
        .hero h1 {
            color: var(--road-ink);
            font-size: clamp(2rem, 4vw, 3.2rem);
            line-height: 1.05;
            margin: 0.45rem 0 0.65rem;
        }
        .hero p {
            color: var(--road-muted);
            font-size: 1.02rem;
            line-height: 1.65;
            margin: 0;
            max-width: 820px;
        }
        .info-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.5rem;
        }
        .info-card {
            background: var(--road-panel);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            color: var(--road-ink);
            border-left: 4px solid var(--road-blue);
        }
        .info-card strong { display: block; margin-bottom: 0.2rem; }
        .info-card span { color: var(--road-muted); font-size: 0.88rem; }
        [data-testid="stMetric"] {
            background: #f7f9fc;
            border: 1px solid #e1e7ee;
            padding: 0.8rem 1rem;
            border-radius: 12px;
        }
        @media (max-width: 760px) {
            .info-strip { grid-template-columns: 1fr; }
            .block-container { padding-top: 1rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model() -> YOLO:
    """Load the trained model once per Streamlit server process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found at {MODEL_PATH}. "
            "Place the trained checkpoint at models/best.pt."
        )
    return YOLO(str(MODEL_PATH))


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=94)
    return buffer.getvalue()


def run_inference(
    model: YOLO,
    image: Image.Image,
    confidence: float,
    iou_threshold: float,
) -> tuple[Image.Image, pd.DataFrame, dict[str, int], float]:
    """Run YOLO inference and return the annotated image and detections."""
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    rgb_array = np.asarray(image.convert("RGB"))

    start_time = time.perf_counter()
    result = model.predict(
        source=rgb_array,
        conf=confidence,
        iou=iou_threshold,
        imgsz=DEFAULT_IMAGE_SIZE,
        device=device,
        verbose=False,
    )[0]
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Ultralytics returns the plotted NumPy array in BGR order.
    annotated_bgr = result.plot(line_width=2, labels=True, conf=True)
    annotated_rgb = annotated_bgr[:, :, ::-1]
    annotated_image = Image.fromarray(annotated_rgb)

    rows: list[dict[str, object]] = []
    if result.boxes is not None and len(result.boxes) > 0:
        coordinates = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

        for index, (xyxy, score, class_id) in enumerate(
            zip(coordinates, confidences, class_ids), start=1
        ):
            x1, y1, x2, y2 = xyxy.tolist()
            rows.append(
                {
                    "Detection": index,
                    "ClassKey": str(result.names[class_id]).lower().replace(" ", "_"),
                    "Confidence": round(float(score), 3),
                    "Left": round(float(x1), 1),
                    "Top": round(float(y1), 1),
                    "Right": round(float(x2), 1),
                    "Bottom": round(float(y2), 1),
                }
            )

    columns = [
        "Detection",
        "ClassKey",
        "Confidence",
        "Left",
        "Top",
        "Right",
        "Bottom",
    ]
    detections = pd.DataFrame(rows, columns=columns)
    class_counts = dict(Counter(detections["ClassKey"])) if not detections.empty else {}
    return annotated_image, detections, class_counts, elapsed_ms


with st.sidebar:
    language = st.selectbox(
        "Language / 语言",
        options=["English", "中文"],
        index=0,
    )

text = TRANSLATIONS[language]
class_names = CLASS_NAMES[language]


st.markdown(
    f"""
    <section class="hero">
        <div class="hero-kicker">{text['hero_kicker']}</div>
        <h1>{text['title']}</h1>
        <p>{text['intro']}</p>
    </section>
    <div class="info-strip">
        <div class="info-card"><strong>{text['card_classes_title']}</strong><span>{text['card_classes_text']}</span></div>
        <div class="info-card"><strong>{text['card_model_title']}</strong><span>{text['card_model_text']}</span></div>
        <div class="info-card"><strong>{text['card_privacy_title']}</strong><span>{text['card_privacy_text']}</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header(text["settings"])
    confidence_threshold = st.slider(
        text["confidence"],
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help=text["confidence_help"],
    )
    iou_threshold = st.slider(
        text["iou"],
        min_value=0.10,
        max_value=0.90,
        value=0.70,
        step=0.05,
        help=text["iou_help"],
    )
    st.divider()
    st.subheader(text["classes_heading"])
    st.caption(text["classes_list"])
    st.divider()
    st.caption(text["academic_note"])


uploaded_file = st.file_uploader(
    text["upload"],
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help=text["upload_help"],
)


if uploaded_file is None:
    st.info(text["upload_prompt"], icon="📤")
else:
    try:
        uploaded_image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error(text["invalid_image"])
        st.stop()

    if uploaded_image.width * uploaded_image.height > 40_000_000:
        st.error(text["large_image"])
        st.stop()

    try:
        with st.spinner(text["spinner"]):
            detector = load_model()
            output_image, detection_table, counts, inference_ms = run_inference(
                detector,
                uploaded_image,
                confidence_threshold,
                iou_threshold,
            )
    except FileNotFoundError as error:
        st.error(text["model_missing"])
        st.stop()
    except Exception as error:
        st.error(text["prediction_failed"])
        st.exception(error)
        st.stop()

    total_detections = len(detection_table)
    highest_confidence = (
        float(detection_table["Confidence"].max()) if total_detections else 0.0
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric(text["detections"], total_detections)
    metric_2.metric(text["detected_classes"], len(counts))
    metric_3.metric(
        text["highest_confidence"],
        f"{highest_confidence:.1%}" if total_detections else "—",
    )
    metric_4.metric(text["inference_time"], f"{inference_ms:.0f} ms")

    st.subheader(text["prediction_result"])
    original_column, result_column = st.columns(2)
    with original_column:
        st.caption(text["original_image"])
        st.image(uploaded_image, use_container_width=True)
    with result_column:
        st.caption(text["model_output"])
        st.image(output_image, use_container_width=True)

    if total_detections == 0:
        st.warning(text["no_detection"], icon="⚠️")
    else:
        st.success(
            text["detected_summary"].format(
                objects=total_detections, classes=len(counts)
            ),
            icon="✅",
        )
        st.subheader(text["details"])

        display_table = detection_table.copy()
        display_table["ClassKey"] = display_table["ClassKey"].map(
            lambda name: class_names.get(name, name.replace("_", " ").title())
        )
        display_table = display_table.rename(
            columns={
                "Detection": text["column_detection"],
                "ClassKey": text["column_class"],
                "Confidence": text["column_confidence"],
                "Left": text["column_left"],
                "Top": text["column_top"],
                "Right": text["column_right"],
                "Bottom": text["column_bottom"],
            }
        )
        st.dataframe(
            display_table,
            hide_index=True,
            use_container_width=True,
            column_config={
                text["column_confidence"]: st.column_config.ProgressColumn(
                    text["column_confidence"],
                    min_value=0.0,
                    max_value=1.0,
                    format="%.3f",
                )
            },
        )

        count_text = " · ".join(
            f"{class_names.get(label, label.replace('_', ' ').title())}: {count}"
            for label, count in counts.items()
        )
        st.caption(f"{text['class_totals']} — {count_text}")

    safe_stem = Path(uploaded_file.name).stem or "road_image"
    st.download_button(
        label=text["download"],
        data=image_to_jpeg_bytes(output_image),
        file_name=f"{safe_stem}_prediction.jpg",
        mime="image/jpeg",
        use_container_width=True,
    )


st.divider()
st.caption(text["footer"])

