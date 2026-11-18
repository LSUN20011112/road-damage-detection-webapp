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


st.set_page_config(
    page_title="Road Damage Detector",
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
                    "Class": result.names[class_id].replace("_", " ").title(),
                    "Confidence": round(float(score), 3),
                    "Left": round(float(x1), 1),
                    "Top": round(float(y1), 1),
                    "Right": round(float(x2), 1),
                    "Bottom": round(float(y2), 1),
                }
            )

    columns = [
        "Detection",
        "Class",
        "Confidence",
        "Left",
        "Top",
        "Right",
        "Bottom",
    ]
    detections = pd.DataFrame(rows, columns=columns)
    class_counts = dict(Counter(detections["Class"])) if not detections.empty else {}
    return annotated_image, detections, class_counts, elapsed_ms


st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">Computer Vision · YOLO26s</div>
        <h1>Road Damage Detector</h1>
        <p>
            Upload a road image to identify potholes, cracks, debris, manholes,
            speed breakers, and road barricades. The trained model runs once the
            image is uploaded and returns an annotated result immediately.
        </p>
    </section>
    <div class="info-strip">
        <div class="info-card"><strong>6 detection classes</strong><span>Road damage and road hazards</span></div>
        <div class="info-card"><strong>YOLO26s checkpoint</strong><span>Automatically selected best.pt</span></div>
        <div class="info-card"><strong>Privacy-conscious</strong><span>Images are processed only for inference</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Detection settings")
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Lower values show more detections but may increase false positives.",
    )
    iou_threshold = st.slider(
        "IoU threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.70,
        step=0.05,
        help="Controls suppression of overlapping bounding boxes.",
    )
    st.divider()
    st.subheader("Recognized classes")
    st.caption(
        "Pothole · Crack · Debris · Manhole · Speed breaker · Road barricade"
    )
    st.divider()
    st.caption(
        "This application is an academic demonstration. Predictions should be "
        "reviewed before being used for maintenance or safety decisions."
    )


uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supported formats: JPG, JPEG, PNG, BMP, and WEBP.",
)


if uploaded_file is None:
    st.info("Upload an image above to start detection.", icon="📤")
else:
    try:
        uploaded_image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("The uploaded file could not be read as a valid image.")
        st.stop()

    if uploaded_image.width * uploaded_image.height > 40_000_000:
        st.error("This image is too large. Please upload an image below 40 megapixels.")
        st.stop()

    try:
        with st.spinner("Loading the model and analyzing the image..."):
            detector = load_model()
            output_image, detection_table, counts, inference_ms = run_inference(
                detector,
                uploaded_image,
                confidence_threshold,
                iou_threshold,
            )
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
    except Exception as error:
        st.error("Prediction failed. Check the application logs for details.")
        st.exception(error)
        st.stop()

    total_detections = len(detection_table)
    highest_confidence = (
        float(detection_table["Confidence"].max()) if total_detections else 0.0
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Detections", total_detections)
    metric_2.metric("Detected classes", len(counts))
    metric_3.metric(
        "Highest confidence",
        f"{highest_confidence:.1%}" if total_detections else "—",
    )
    metric_4.metric("Inference time", f"{inference_ms:.0f} ms")

    st.subheader("Prediction result")
    original_column, result_column = st.columns(2)
    with original_column:
        st.caption("Original image")
        st.image(uploaded_image, use_container_width=True)
    with result_column:
        st.caption("Model output")
        st.image(output_image, use_container_width=True)

    if total_detections == 0:
        st.warning(
            "No road damage or hazard was detected at the selected confidence "
            "threshold. Try a lower threshold, or review the image manually.",
            icon="⚠️",
        )
    else:
        st.success(
            f"Detected {total_detections} object(s) across {len(counts)} class(es).",
            icon="✅",
        )
        st.subheader("Detection details")
        st.dataframe(
            detection_table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.3f"
                )
            },
        )

        count_text = " · ".join(f"{label}: {count}" for label, count in counts.items())
        st.caption(f"Class totals — {count_text}")

    safe_stem = Path(uploaded_file.name).stem or "road_image"
    st.download_button(
        label="Download annotated result",
        data=image_to_jpeg_bytes(output_image),
        file_name=f"{safe_stem}_prediction.jpg",
        mime="image/jpeg",
        use_container_width=True,
    )


st.divider()
st.caption(
    "Group F road-damage detection project · Model predictions are informational "
    "and may include false positives or false negatives."
)
