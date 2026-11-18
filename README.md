# Road Damage Detection Web App

A Streamlit application that uses the trained Group F YOLO26s `best.pt` checkpoint to detect six road-damage and road-hazard classes in a user-uploaded image.

**Live application:**  
[https://6-class-road-damage-detector.streamlit.app](https://6-class-road-damage-detector.streamlit.app)

## Features

- Upload JPG, JPEG, PNG, BMP, or WEBP images.
- Detect potholes, cracks, debris, manholes, speed breakers, and road barricades.
- Compare the original and annotated images side by side.
- Adjust confidence and IoU thresholds.
- Review detection classes, confidence scores, and bounding-box coordinates.
- Download the annotated prediction image.
- Receive an explicit message when no object is detected.

## Project structure

```text
Road_Damage_Detection_WebApp/
├── app.py
├── models/
│   └── best.pt
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── packages.txt
├── .gitignore
└── README.md
```

## Run locally

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies and launch the application:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local address, normally `http://localhost:8501`.

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository and upload the contents of this folder. Keep `app.py`, `requirements.txt`, and `models/best.pt` in the paths shown above.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
3. Select **Create app**, choose the repository and branch, and set the entrypoint to `app.py`.
4. Deploy the app. Streamlit will install the Python packages and provide a public `streamlit.app` URL.

The checkpoint is approximately 20 MB, so it is below GitHub's per-file limit and can be committed directly. The model is cached with `st.cache_resource`, so it loads once for each server process rather than once per uploaded image.

## Model configuration

- Architecture: YOLO26s object detector
- Checkpoint: `models/best.pt`
- Input image size: 640 pixels
- Default confidence threshold: 0.25
- Default IoU threshold: 0.70
- Classes: pothole, crack, debris, manhole, speed breaker, road barricade

## Important limitations

This is an academic demonstration, not a certified road-safety system. Predictions can include false positives or false negatives, especially under unusual lighting, occlusion, small-object, or domain-shift conditions. Results should be reviewed before maintenance or safety decisions are made.

## Licensing note

Ultralytics code and trained models are subject to the Ultralytics licensing terms. The AGPL-3.0 option is suitable when the complete application is released as open source under compatible terms. A separate Ultralytics enterprise license may be required for proprietary or closed-source commercial deployment. Review the current terms before commercial use.
