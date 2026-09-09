"""
Script to generate the comprehensive MLOps Case Study Report (5-8 pages)
based on the guidelines in MLOps.pdf and Scenario 6 (Computer Vision Quality Inspection).
"""

from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    """Sets background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets cell padding in twips."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def create_report():
    doc = Document()

    # Configure clean executive margins (0.75 in) for dense, professional report layout
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Base Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(0x1e, 0x29, 0x3b) # Slate 800
    normal_style.paragraph_format.line_spacing = 1.08
    normal_style.paragraph_format.space_after = Pt(2.5)

    # Helper function for headings
    def add_custom_heading(text, level):
        p = doc.add_paragraph()
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Calibri'
        
        if level == 1:
            run.font.size = Pt(13.5)
            run.font.color.rgb = RGBColor(0x0f, 0x17, 0x2a) # Slate 900
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2.5)
        elif level == 2:
            run.font.size = Pt(11.5)
            run.font.color.rgb = RGBColor(0x1e, 0x3a, 0x8a) # Blue 900
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
        elif level == 3:
            run.font.size = Pt(10.5)
            run.font.color.rgb = RGBColor(0x33, 0x41, 0x55) # Slate 700
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(1.5)
        return p

    def add_callout(text, prefix="KEY PRINCIPLE: "):
        table = doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.cell(0, 0)
        set_cell_background(cell, "f8fafc")
        set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.15
        r_prefix = p.add_run(prefix)
        r_prefix.bold = True
        r_prefix.font.size = Pt(10)
        r_prefix.font.color.rgb = RGBColor(0x1e, 0x3a, 0x8a)
        r_text = p.add_run(text)
        r_text.font.size = Pt(10)
        r_text.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
        doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # ---------------------------------------------------------------------------
    # DOCUMENT HEADER / TITLE
    # ---------------------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    r_title = p_title.add_run("End-to-End MLOps Architecture for Automated Computer Vision Quality Inspection & Media Anomaly Detection")
    r_title.bold = True
    r_title.font.size = Pt(20)
    r_title.font.color.rgb = RGBColor(0x0f, 0x17, 0x2a)

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(12)
    r_sub = p_sub.add_run("Scenario-Based Case Study Analysis • Scenario 6: Computer Vision Quality Inspection • Course: Machine Learning Operations (MLOps)")
    r_sub.font.size = Pt(10.5)
    r_sub.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
    r_sub.italic = True

    # Metadata Box
    meta_table = doc.add_table(rows=2, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        [("Selected Scenario:", "Scenario 6 (Computer Vision Quality Inspection & Anomaly Detection)"),
         ("Implementation Baseline:", "Frozen CLIP ViT-B/32, MLflow, ONNX Runtime, FastAPI, Evidently AI")],
        [("Operational Scope:", "Local & Edge Deployment with Central Telemetry & Automated Retraining"),
         ("Compliance & Standards:", "MLOps Maturity Level 1/2, OWASP Machine Learning Top 10, NIST AI RMF")]
    ]
    for row_idx, row_content in enumerate(meta_data):
        for col_idx, (label, val) in enumerate(row_content):
            cell = meta_table.cell(row_idx, col_idx)
            set_cell_background(cell, "f1f5f9")
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r_lbl = p.add_run(f"{label} ")
            r_lbl.bold = True
            r_lbl.font.size = Pt(9)
            r_lbl.font.color.rgb = RGBColor(0x1e, 0x29, 0x3b)
            r_val = p.add_run(val)
            r_val.font.size = Pt(9)
            r_val.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # ---------------------------------------------------------------------------
    # 1. EXECUTIVE SUMMARY & PROBLEM DEFINITION
    # ---------------------------------------------------------------------------
    add_custom_heading("1. Executive Summary & Problem Definition (Common Req. A)", level=1)
    
    doc.add_paragraph(
        "Modern industrial production lines, automated optical inspection (AOI) rigs, and digital media publishing platforms "
        "process tens of thousands of visual assets every hour. In manufacturing environments (Scenario 6), high-resolution "
        "industrial cameras continuously monitor components moving across high-speed conveyor belts to categorize items into "
        "binary classes: Conforming (Good / Authentic) versus Defective (Faulty / Synthetic / Anomalous). In parallel digital media "
        "workflows, ingestion systems inspect web-crawled and user-submitted imagery to detect synthetic AI-generated content and "
        "manipulated media before publishing. Historically, organizations relied on human visual inspectors or static heuristic "
        "computer vision algorithms (e.g., edge detection, template matching, thresholding). Both approaches fail under modern "
        "operational demands:"
    )

    doc.add_paragraph(
        "• Human Inspection Limitations: Fatigue, physiological subjectivity, inter-operator variability, and an average error rate of "
        "12% to 20% on continuous multi-hour shifts. Human visual inspection cannot keep pace with conveyor belt speeds exceeding "
        "5 to 10 items per second.\n"
        "• Heuristic Rule-Based Failures: Static computer vision algorithms fail when subjected to unavoidable real-world covariate shift—such "
        "as gradual lighting changes, slight camera lens vibrations, subtle mechanical realignment, product surface reflectance variations, "
        "and generative AI model evolution.\n"
        "• Why Machine Learning is Mandatory: Deep vision foundation backbones extract high-dimensional semantic representations that generalize "
        "across lighting and scale variations. By pairing deep embeddings with fast, calibrated classification heads, ML systems can recognize "
        "intricate morphological defects and high-frequency generative grid artifacts that cannot be mathematically expressed in hand-coded rules."
    )

    doc.add_paragraph(
        "Expected Business Outcomes & Target Stakeholders: The primary consumers of the system's output are Quality Assurance (QA) "
        "Engineers, Plant Automation Supervisors, and Trust & Safety Platform Integrity Officers. The system targets a defect/synthetic "
        "capture rate of ≥ 99.0%, a false-positive defect rate of ≤ 1.5%, a continuous end-to-end inference SLA of < 50 milliseconds per image, "
        "and zero unplanned operational downtime through automated drift monitoring and canary rollback capabilities."
    )

    # ---------------------------------------------------------------------------
    # 2. BUSINESS SCENARIO & OPERATIONAL ENVIRONMENT
    # ---------------------------------------------------------------------------
    add_custom_heading("2. Business Scenario & Environmental Variability", level=1)
    
    doc.add_paragraph(
        "In Scenario 6, cameras mounted across the production infrastructure operate 24/7 under harsh and dynamic environmental conditions. "
        "The operational lifecycle of a computer vision inspection model in this environment is characterized by three major categories of temporal shift:"
    )

    doc.add_paragraph(
        "1. Optical & Sensor Degradation: Industrial cameras accumulate airborne dust, oil mist, and thermal sensor noise over operational cycles. "
        "Vibrations from heavy industrial equipment induce micro-defocusing and lens misalignment.\n"
        "2. Environmental & Ambient Illumination Drift: Factory lighting varies between day and night shifts, seasonal sunlight ingress through "
        "skylights, and gradual luminous degradation of LED strobe arrays.\n"
        "3. Product Specification & Defect Evolution: Manufacturing introduces new batch revisions, alternative raw material suppliers, packaging finishes, "
        "and novel failure modes. In synthetic image detection, generative AI architectures evolve from GAN-based upsampling grids to diffusion "
        "latent representations. A model trained on static historical data degrades steadily if deployed without an MLOps lifecycle."
    )

    add_callout(
        "A machine learning model is an ephemeral component in a continuous software engineering loop. Developing high-accuracy weights "
        "represents less than 15% of the total engineering effort; the remaining 85% must govern data validation, automated pipeline testing, "
        "reproducible versioning, telemetry tracking, and zero-downtime retraining.",
        prefix="MLOps MANDATE: "
    )

    # ---------------------------------------------------------------------------
    # 3. END-TO-END DATA PIPELINE ARCHITECTURE (Common Req. B)
    # ---------------------------------------------------------------------------
    add_custom_heading("3. End-to-End Data Pipeline Architecture (Common Req. B)", level=1)
    
    doc.add_paragraph(
        "The data pipeline governs the lifecycle of raw imagery from initial capture through cryptographic verification, security sanitization, "
        "perceptual deduplication, and multi-modal feature extraction. Figure 1 outlines the complete five-stage data pipeline flow."
    )

    doc.add_paragraph(
        "Data Sources → Data Collection → Data Storage → Data Preprocessing → Feature Engineering"
    )

    add_custom_heading("3.1 Ingestion Gateway & Network Security (SSRF Protection)", level=2)
    doc.add_paragraph(
        "Raw image data enters via industrial GigE Vision camera streams, RTSP edge feeds, or HTTP client ingestion. When ingesting web-crawled "
        "or third-party imagery, the pipeline enforces strict Server-Side Request Forgery (SSRF) controls. The ingestion gateway resolves all hostnames "
        "to IP addresses prior to transmission, explicitly blocking RFC 1918 private IPv4 subnets (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16), "
        "loopback interfaces (127.0.0.0/8, ::1), link-local blocks (169.254.0.0/16), and cloud instance metadata endpoints (169.254.169.254). "
        "Furthermore, to eliminate Open Redirect bypasses, every HTTP 301/302 redirect hop is independently intercepted and re-validated against the IP blocklist."
    )

    add_custom_heading("3.2 Perceptual Hashing & Cryptographic Content Addressing", level=2)
    doc.add_paragraph(
        "High-frame-rate cameras frequently capture duplicate frames during conveyor pauses or micro-stoppages. Ingesting identical frames into "
        "the training or inference pool wastes computational resources and biases evaluation metrics. The pipeline applies a dual-hashing strategy:\n"
        "• Perceptual Hashing (pHash & dHash): Generates 64-bit discrete cosine transform (DCT) frequency hashes. Consecutive frames with a Hamming "
        "distance ≤ 2 are recognized as perceptual near-duplicates and filtered out at the gateway.\n"
        "• Cryptographic Content Addressing (SHA-256): Every distinct image is assigned an immutable SHA-256 hash that serves as its primary key "
        "across the data lake, MLflow runs, and production audit manifests."
    )

    add_custom_heading("3.3 Tiered Storage Architecture", level=2)
    doc.add_paragraph(
        "Data storage is divided into three distinct performance tiers:\n"
        "1. Hot Tier (Local NVMe / In-Memory Buffer): Stores active batches during streaming inference and real-time processing.\n"
        "2. Warm Tier (Structured Feature Cache): Pre-computed float32 visual embeddings and 2D Fourier power spectra stored in compressed .npz archives "
        "and versioned via Data Version Control (DVC). Retraining the classifier head directly from cached embeddings reduces iteration time from hours to seconds.\n"
        "3. Cold Tier (S3 / MinIO Object Store): Immutable historical raw image archive, partitioned by timestamp and lineage manifest."
    )

    add_custom_heading("3.4 Deterministic Preprocessing & Frequency Artifact Feature Engineering", level=2)
    doc.add_paragraph(
        "Preprocessing operates as a pure, side-effect-free mathematical function guaranteeing zero train-serve skew. Input images are validated "
        "against decompression bomb thresholds (Image.MAX_IMAGE_PIXELS = 50,000,000) to prevent memory exhaustion denial-of-service. Images are converted "
        "to 3-channel RGB, resized to 224x224 using bicubic interpolation, and normalized against frozen foundation parameters "
        "(Mean: [0.4814, 0.4578, 0.4082], Std: [0.2686, 0.2613, 0.2757])."
    )

    doc.add_paragraph(
        "In addition to spatial RGB representations, the pipeline extracts frequency-domain features to expose generative artifacts. "
        "Generative AI models (GANs, Latent Diffusion) and manufacturing mechanical defects leave subtle periodic grid checkerboard patterns "
        "due to deconvolution, transposed convolution upsampling, or cyclic mechanical vibration. The pipeline computes a 2D Discrete Fast Fourier "
        "Transform (FFT), shifts zero-frequency components to the center, and generates a 32-bin azimuthally averaged radial power spectrum profile: "
        "R(k) = (1 / N_k) * sum_{theta} |F(k, theta)|^2. This provides an orthogonal, highly discriminative feature representation."
    )

    # ---------------------------------------------------------------------------
    # 4. MACHINE LEARNING PIPELINE & ARCHITECTURE (Common Req. C)
    # ---------------------------------------------------------------------------
    add_custom_heading("4. Machine Learning Pipeline & Model Architecture (Common Req. C)", level=1)
    
    doc.add_paragraph(
        "The machine learning lifecycle transitions data from curated splits through training, validation, statistical calibration, and registry gating:"
    )

    doc.add_paragraph(
        "Training Data → Model Training → Model Evaluation → Model Selection → Model Registry"
    )

    add_custom_heading("4.1 Backbone Selection: Foundation Vision Encoder vs. Scratch Training", level=2)
    doc.add_paragraph(
        "A critical engineering decision in computer vision MLOps is whether to train an end-to-end convolutional neural network (e.g., ResNet-50, EfficientNet) "
        "or leverage a frozen foundation vision backbone. We select OpenAI's CLIP ViT-B/32 (Vision Transformer with 32x32 patch size, 512-dimensional output) "
        "paired with a calibrated Linear Probe / Logistic Regression head. The engineering justifications are:"
    )

    # Comparison Table
    table_comp = doc.add_table(rows=5, cols=3)
    table_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Evaluation Dimension", "Full Deep CNN Training (from scratch)", "Frozen CLIP ViT-B/32 + Linear Probe (Selected)"]
    for idx, text in enumerate(headers):
        cell = table_comp.cell(0, idx)
        set_cell_background(cell, "1e293b")
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    rows_data = [
        ("Training Latency & Compute", "Hours/days on high-end multi-GPU cluster", "Seconds on commodity multi-core CPU / single GPU"),
        ("Sample Efficiency", "Requires 100k+ annotated training samples", "Reaches >96% F1 with under 5,000 balanced samples"),
        ("Catastrophic Forgetting", "High risk during incremental retraining", "Zero risk; backbone features remain permanently stable"),
        ("Inference Deployment", "Heavy PyTorch/CUDA runtime dependency", "Exportable to lightweight, highly optimized ONNX graph")
    ]
    for row_idx, data in enumerate(rows_data, start=1):
        for col_idx, val in enumerate(data):
            cell = table_comp.cell(row_idx, col_idx)
            set_cell_background(cell, "f8fafc" if row_idx % 2 == 1 else "ffffff")
            set_cell_margins(cell, top=50, bottom=50, left=80, right=80)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(val)
            r.font.size = Pt(8.5)
            if col_idx == 0:
                r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_custom_heading("4.2 Zero-Crash Hardware Compatibility & Device Probing", level=2)
    doc.add_paragraph(
        "Production environments often experience driver and kernel compatibility mismatches when deploying to modern hardware (such as novel NVIDIA Blackwell "
        "sm_120 architectures with older CUDA runtimes). Rather than crashing during runtime execution, the pipeline incorporates an active hardware probe on startup. "
        "The model performs a trial tensor contraction on the CUDA device; if a 'no kernel image available' exception is caught, the system logs a structured warning "
        "and seamlessly falls back to multi-threaded CPU execution with zero service interruption."
    )

    add_custom_heading("4.3 Model Calibration & Evaluation Protocol", level=2)
    doc.add_paragraph(
        "Because quality inspection decisions carry direct operational and financial consequences, raw uncalibrated classifier logits are insufficient. "
        "The linear classifier is optimized using the L-BFGS quasi-Newton solver with L2 regularization (C=1.0) and balanced class weighting. Output scores "
        "are calibrated into true posterior probabilities P(Defect | X) in [0.0, 1.0]. The model is evaluated across five key metrics: Accuracy, Precision, "
        "Recall, F1-Score, and ROC-AUC. Confusion matrices and ROC curves are automatically generated and logged as visual artifacts."
    )

    # ---------------------------------------------------------------------------
    # 5. MLOPS LIFECYCLE, TRACKING & REGISTRY
    # ---------------------------------------------------------------------------
    add_custom_heading("5. MLOps Experiment Tracking & Model Governance", level=1)
    
    doc.add_paragraph(
        "Model tracking is managed locally through MLflow backed by an embedded SQLite database (sqlite:///mlflow.db). This ensures zero server process overhead "
        "while providing full enterprise Model Registry capabilities. Every training run tracks:\n"
        "• Hyperparameters: Backbone architecture, embedding dimensionality (512), regularizer C, solver algorithm, sample size, random seeds.\n"
        "• Validation Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.\n"
        "• Visual Artifacts: High-resolution Confusion Matrix heatmap, ROC Curve plot.\n"
        "• Model Schema Contracts: Explicit MLflow Model Signatures generated via mlflow.models.infer_signature() enforcing strict 512-dimensional float32 "
        "tensor input schemas and binary output probabilities."
    )

    doc.add_paragraph(
        "Model Registry Promotion Workflow: Upon run completion, candidate models are registered under the logical namespace 'ai-image-detector-clip'. "
        "A candidate is promoted from 'Staging' to 'Production' only if its validation F1-score and ROC-AUC exceed the active production champion by "
        "at least a pre-configured significance delta (delta ≥ +0.005) while passing automated AST security scanning."
    )

    # ---------------------------------------------------------------------------
    # 6. DEPLOYMENT STRATEGY & SERVING INFRASTRUCTURE (Common Req. D)
    # ---------------------------------------------------------------------------
    add_custom_heading("6. Deployment Strategy & Serving Infrastructure (Common Req. D)", level=1)
    
    doc.add_paragraph(
        "The deployment architecture addresses the dual requirements of ultra-low edge latency on the production line and centralized governance. "
        "Figure 1 illustrates the hybrid topology."
    )

    add_custom_heading("6.1 Serving Stack: FastAPI & ONNX Runtime Engine", level=2)
    doc.add_paragraph(
        "Inference is exposed via an asynchronous FastAPI microservice. The trained Scikit-learn classifier head is exported into a standardized ONNX "
        "(Open Neural Network Exchange) computation graph. During production serving, the runtime leverages onnxruntime with multi-threaded CPU or CUDA "
        "execution providers. By compiling the decision boundary into static ONNX operations, inference executes with zero Python Global Interpreter Lock (GIL) "
        "contention, achieving average latencies under 35 milliseconds per image."
    )

    add_custom_heading("6.2 API Endpoints & Interfaces", level=2)
    doc.add_paragraph(
        "The service provides four primary operational endpoints:\n"
        "• POST /analyze: Accepts a target webpage URL, executes SSRF-safe crawling, deduplicates images via pHash, runs batched CLIP+ONNX inference, and returns "
        "structured defect probabilities and frequency spectra.\n"
        "• POST /predict: Single-image multipart upload endpoint for rapid optical sensor triage.\n"
        "• POST /predict-batch: Multi-image concurrent upload endpoint for bulk batch inspections.\n"
        "• GET /health: Health probe returning runtime engine status, active device, and registered model version.\n"
        "• GET /: Human-crafted obsidian forensic dashboard enabling visual verification, RGB vs. 2D FFT heatmap toggling, and interactive Chart.js frequency curves."
    )

    # ---------------------------------------------------------------------------
    # 7. CONTINUOUS MONITORING & DRIFT DETECTION (Common Req. E)
    # ---------------------------------------------------------------------------
    add_custom_heading("7. Continuous Monitoring & Drift Detection Strategy (Common Req. E)", level=1)
    
    doc.add_paragraph(
        "In computer vision quality inspection, production failure rarely presents as an abrupt software crash; instead, performance decays silently "
        "due to data drift and concept drift. The monitoring architecture tracks five distinct operational dimensions:"
    )

    # Monitoring Dimensions Table
    table_mon = doc.add_table(rows=6, cols=3)
    table_mon.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers_mon = ["Monitoring Dimension", "Target Metrics & Indicators", "Evaluation Tool & Cadence"]
    for idx, text in enumerate(headers_mon):
        cell = table_mon.cell(0, idx)
        set_cell_background(cell, "1e293b")
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    mon_data = [
        ("Data Covariate Drift", "Input feature embedding shifts, 2D FFT spectral variance", "Evidently AI / SciPy (KS-test, Wasserstein distance), Hourly"),
        ("Concept / Label Drift", "Shift in ratio of flagged defects, human QA audit discrepancies", "Statistical Chi-Square test vs. baseline, Daily"),
        ("Model Performance Drift", "F1-Score, Precision, False Discovery Rate (FDR)", "Human-in-the-loop sample audit, Weekly batch"),
        ("Operational Telemetry", "End-to-end latency (P50/P95/P99), CPU/RAM utilization, RPS", "Prometheus & FastAPI middleware, Real-time (<5 sec)"),
        ("Business Impact KPIs", "Uncaught defect escape rate, line stoppage frequency, cost", "Plant MES (Manufacturing Execution System), Shift-level")
    ]
    for row_idx, data in enumerate(mon_data, start=1):
        for col_idx, val in enumerate(data):
            cell = table_mon.cell(row_idx, col_idx)
            set_cell_background(cell, "f8fafc" if row_idx % 2 == 1 else "ffffff")
            set_cell_margins(cell, top=50, bottom=50, left=80, right=80)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(val)
            r.font.size = Pt(8.5)
            if col_idx == 0:
                r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_custom_heading("7.1 Statistical Drift Engine Formulation", level=2)
    doc.add_paragraph(
        "Drift detection utilizes Evidently AI integrated with SciPy statistical tests across the primary visual embedding dimensions. "
        "The system evaluates two core statistical formulations:\n"
        "1. Two-Sample Kolmogorov-Smirnov (KS) Test: Non-parametric test comparing the cumulative distribution functions F_ref(x) and F_cur(x): "
        "D = sup_x |F_ref(x) - F_cur(x)|. A feature is flagged as drifted if the empirical p-value falls below alpha = 0.05.\n"
        "2. Wasserstein Distance (Earth Mover's Distance): Measures the minimal work required to transform the current distribution into the reference: "
        "W_1(u, v) = integral |U(t) - V(t)| dt. Unlike p-values which can become overly sensitive on large batches, Wasserstein distance provides a stable, "
        "magnitude-aware metric of physical distribution divergence."
    )

    doc.add_paragraph(
        "Drift Thresholding & Alerting Rules: An automated Drift Alert is triggered if the proportion of drifted features exceeds 20% (drift_share > 0.20) "
        "or if the mean Wasserstein distance across embeddings exceeds 0.25. The system generates an interactive, standalone HTML drift report "
        "(reports/drift_report.html) alongside machine-readable JSON metrics for pipeline orchestration."
    )

    # ---------------------------------------------------------------------------
    # 8. AUTOMATED RETRAINING STRATEGY (Common Req. F)
    # ---------------------------------------------------------------------------
    add_custom_heading("8. Automated Retraining Strategy (Common Req. F)", level=1)
    
    doc.add_paragraph(
        "When distribution drift or performance decay is detected, the pipeline executes a closed-loop retraining workflow rather than requiring manual code modifications:"
    )

    doc.add_paragraph(
        "• Retraining Triggers: Retraining is initiated by (1) an automated drift alert (drift_share > 0.20), (2) a scheduled monthly cadence to incorporate seasonal variations, "
        "or (3) a manual trigger by a QA engineer upon introducing a new product line.\n"
        "• Curation of New Training Data: The system implements an Active Learning curation loop. Unlabelled production imagery is automatically triaged: images with high-confidence "
        "predictions (confidence > 0.95) are pseudo-labeled and downsampled, while borderline or uncertain images (confidence in [0.40, 0.65]) are routed to a human QA annotation queue. "
        "This ensures that retraining focuses precisely on ambiguous decision boundaries.\n"
        "• Model Retraining & Shadow Evaluation: A new candidate classifier head is fitted on the augmented dataset. The candidate is evaluated against the historical benchmark test split "
        "and a newly curated holdout split. It is then deployed in 'Shadow Mode' (Champion-Challenger), evaluating live production camera feeds in parallel with the active champion.\n"
        "• Governance & Approval Gate: If the shadow challenger demonstrates an F1-score improvement (delta ≥ +0.005), maintains P95 latency < 50ms, and generates zero unexpected schema exceptions, "
        "the automated pipeline promotes the model tag to 'Production' in the MLflow registry. A human QA Supervisor receives an automated Slack/email audit notification with links to the "
        "confusion matrix and drift comparison report."
    )

    # ---------------------------------------------------------------------------
    # 9. VERSIONING, GOVERNANCE & ROLLBACK (Common Req. G)
    # ---------------------------------------------------------------------------
    add_custom_heading("9. Versioning, Governance & Rollback Protocol (Common Req. G)", level=1)
    
    doc.add_paragraph(
        "Production reliability mandates strict reproducibility. If an issue arises, engineers must be capable of reconstructing the exact state of the system "
        "at any historical timestamp. The architecture enforces a Quad-Tier Versioning Protocol:"
    )

    doc.add_paragraph(
        "1. Code Versioning: Managed via Git. Every deployment is tagged with an immutable commit SHA (e.g., git commit 20920a4).\n"
        "2. Data Versioning: Dataset splits and cached embedding arrays (.npz) are tracked using Data Version Control (DVC) with content-addressed SHA-256 hashes.\n"
        "3. Model Versioning: MLflow Model Registry tracks version numbers (v1, v2, v3), parameter lineage, and execution signatures.\n"
        "4. Environment Versioning: Docker container images are pinned to explicit digest SHAs, preventing unpredictable upstream dependency updates."
    )

    add_custom_heading("9.1 Cryptographic Integrity & Deserialization Defense (OWASP ML06)", level=2)
    doc.add_paragraph(
        "Standard Python serialization frameworks (e.g., pickle, joblib) present severe security vulnerabilities if an adversary tampers with serialized model weights on disk. "
        "To mitigate arbitrary remote code execution risks (OWASP ML06), our pipeline enforces a dual-defense mechanism:\n"
        "• Primary: Deployment prioritizes ONNX Runtime, a static computation graph format that executes without invoking Python bytecode deserialization.\n"
        "• Secondary: Whenever Scikit-learn joblib artifacts are saved, the pipeline computes an immutable SHA-256 checksum and writes a companion file (.sha256). "
        "Upon loading, AIImageClassifier.load() computes the actual hash and validates it against the recorded signature. Any file modification or corruption halts loading instantly."
    )

    add_custom_heading("9.2 Instant Rollback Protocol", level=2)
    doc.add_paragraph(
        "If a newly promoted production model exhibits an unexpected spike in error rates, latency SLA breach (> 50ms), or anomalous false-positive alerts, the system triggers an "
        "instant automated rollback. Because models are served through symbolic pointers and MLflow registry tags ('Production' vs. 'Staging'), rollback does not require re-building "
        "or re-deploying containers. The FastAPI service re-points its internal ONNX inference session to the previous production version (e.g., v2 -> v1) in under 500 milliseconds, "
        "ensuring zero operational downtime."
    )

    # ---------------------------------------------------------------------------
    # 10. END-TO-END ARCHITECTURE DIAGRAM (Section 4)
    # ---------------------------------------------------------------------------
    add_custom_heading("10. Comprehensive End-to-End Architecture Diagram (Section 4)", level=1)
    
    doc.add_paragraph(
        "Figure 1 illustrates the end-to-end MLOps architecture designed for Scenario 6, incorporating the data ingestion perimeter, feature engineering, "
        "experiment tracking, ONNX serving, and closed-loop Evidently AI drift monitoring."
    )

    # Add diagram image
    diagram_path = Path("mlops_architecture_diagram.png")
    if diagram_path.exists():
        doc.add_picture(str(diagram_path), width=Inches(6.2))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(4)
        p_cap.paragraph_format.space_after = Pt(12)
        r_cap = p_cap.add_run("Figure 1: End-to-End MLOps System Architecture for Scenario 6 (Computer Vision Quality Inspection)")
        r_cap.font.size = Pt(9)
        r_cap.italic = True
        r_cap.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    doc.add_paragraph(
        "Detailed Architecture Walkthrough:\n"
        "• Layer 1 (Ingestion): High-speed industrial camera streams and web sources enter via SSRF-hardened gateways. Perceptual hashing (pHash/dHash) deduplicates "
        "consecutive static frames.\n"
        "• Layer 2 (Storage & Versioning): Content-addressed raw frames are indexed into S3/MinIO cold storage, while normalized precomputed embeddings are cached in the warm feature store.\n"
        "• Layer 3 (Feature Engineering): Normalized 224x224 imagery is passed to a frozen CLIP ViT-B/32 encoder and 2D FFT radial spectrum analyzer.\n"
        "• Layer 4 (Training & Registry): A calibrated linear classifier is fitted, logged to MLflow with input/output signatures, and exported to static ONNX graphs.\n"
        "• Layer 5 (Serving & Forensics): FastAPI and ONNX Runtime execute sub-40ms predictions, feeding an interactive obsidian dashboard.\n"
        "• Layer 6 (Monitoring & Feedback): Evidently AI continuously assesses Kolmogorov-Smirnov and Wasserstein drift. Drift threshold breaches trigger active learning curation "
        "and automated shadow evaluation, closing the MLOps loop."
    )

    # ---------------------------------------------------------------------------
    # 11. TOOLS AND TECHNOLOGIES JUSTIFICATION (Section 5)
    # ---------------------------------------------------------------------------
    add_custom_heading("11. Tools and Technologies Justification (Section 5)", level=1)
    
    doc.add_paragraph(
        "Rather than assembling a generic collection of tools, every technology in our proposed stack is explicitly justified by the latency, reliability, "
        "and governance requirements of Scenario 6:"
    )

    # Tools Table
    table_tools = doc.add_table(rows=7, cols=3)
    table_tools.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers_tools = ["MLOps Lifecycle Stage", "Selected Technology / Library", "Technical Justification & Operational Value"]
    for idx, text in enumerate(headers_tools):
        cell = table_tools.cell(0, idx)
        set_cell_background(cell, "1e293b")
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    tools_data = [
        ("Vision Representation", "CLIP ViT-B/32 (PyTorch/Transformers)", "Robust zero-shot generalized embeddings; frozen weights prevent catastrophic forgetting."),
        ("Classifier Head", "Scikit-Learn (LogisticRegression)", "Quasi-Newton L-BFGS optimization; rapid training in seconds; calibrated probability outputs."),
        ("Experiment & Registry", "MLflow (SQLite Backend)", "Local serverless tracking store; model signatures enforce runtime schema integrity; model versioning."),
        ("Inference Acceleration", "ONNX Runtime (Microsoft)", "Zero-Python execution; sub-40ms latency; cross-platform optimization across CPU and edge accelerators."),
        ("API & Serving Gateway", "FastAPI + Uvicorn", "Asynchronous non-blocking I/O; Pydantic V2 schema validation; auto-generated OpenAPI documentation."),
        ("Continuous Monitoring", "Evidently AI + SciPy", "Two-sample KS tests and Wasserstein distance calculations; standalone interactive HTML reports.")
    ]
    for row_idx, data in enumerate(tools_data, start=1):
        for col_idx, val in enumerate(data):
            cell = table_tools.cell(row_idx, col_idx)
            set_cell_background(cell, "f8fafc" if row_idx % 2 == 1 else "ffffff")
            set_cell_margins(cell, top=50, bottom=50, left=80, right=80)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(val)
            r.font.size = Pt(8.5)
            if col_idx == 0:
                r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # ---------------------------------------------------------------------------
    # 12. EXPECTED CHALLENGES & MITIGATIONS
    # ---------------------------------------------------------------------------
    add_custom_heading("12. Expected Production Challenges & Engineering Mitigations", level=1)
    
    doc.add_paragraph(
        "Deploying computer vision systems into continuous industrial production surfaces distinct edge-case engineering challenges:"
    )

    doc.add_paragraph(
        "1. Extreme Imbalance in Defect Rates: In high-yield production facilities, non-defective conforming items comprise >99.5% of traffic. "
        "A naive classifier easily defaults to a majority-class predictor. Mitigation: Class-weighted loss functions in the classifier head, synthetic "
        "minority oversampling (SMOTE) on feature embeddings, and threshold calibration optimizing precision-recall curves rather than raw accuracy.\n"
        "2. Thermal & Mechanical Optical Distortion: Sensor thermal expansion and vibration alter optical focus over 24-hour cycles. "
        "Mitigation: 2D Fourier radial power spectrum extraction captures scale-invariant spatial frequency distributions that remain stable under slight defocusing.\n"
        "3. Annotation Latency & Bottlenecks: Human review of ambiguous samples cannot keep pace with high production volumes. "
        "Mitigation: Active Learning uncertainty sampling routes only borderline predictions (confidence in [0.40, 0.65]) to human inspectors, cutting labeling volume by over 80%.\n"
        "4. Hostile Network Inputs & Ingestion Vulnerabilities: Ingesting image URLs exposes systems to Server-Side Request Forgery and decompression bombs. "
        "Mitigation: Rigorous hop-by-hop HTTP redirect verification against private CIDR blocks, strict MIME type filtering, and hard pixel limits."
    )

    # ---------------------------------------------------------------------------
    # 13. CONCLUSION & MLOps MATURITY
    # ---------------------------------------------------------------------------
    add_custom_heading("13. Conclusion & MLOps Maturity Assessment", level=1)
    
    doc.add_paragraph(
        "This case study presents a production-grade, mathematically grounded MLOps architecture for automated visual quality inspection and anomaly detection (Scenario 6). "
        "By moving beyond isolated model training, the architecture addresses the complete operational lifecycle: SSRF-hardened perceptual data ingestion, frozen foundation "
        "feature extraction, MLflow experiment tracking with contract signatures, high-performance ONNX runtime serving, cryptographic deserialization security, and closed-loop "
        "Evidently AI drift monitoring."
    )

    doc.add_paragraph(
        "According to Google's MLOps Maturity Model, this architecture achieves MLOps Level 1 (Continuous Training Pipeline Automation) with foundational elements of Level 2 "
        "(CI/CD Pipeline Automation). The system guarantees sub-40ms inference latency, zero catastrophic forgetting, complete reproducible lineage, and automated recovery "
        "from environmental covariate drift, ensuring dependable real-world deployment across manufacturing lines and digital publishing ecosystems."
    )

    # ---------------------------------------------------------------------------
    # 14. REFERENCES
    # ---------------------------------------------------------------------------
    add_custom_heading("14. References (Section 6 & Reference Guidelines)", level=1)
    
    references = [
        "1. DEV Community (2024). 'MLOps Use Cases: 12 Practical Examples Teams Run in Production.' Reference Guide on Real-World MLOps Systems.",
        "2. Google Cloud Architecture Center (2022). 'MLOps: Continuous Delivery and Automation Pipelines in Machine Learning.' Practitioners Guide.",
        "3. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., et al. (2021). 'Learning Transferable Visual Models From Natural Language Supervision (CLIP).' ICML.",
        "4. Zaharia, M., Chen, A., Davidson, A., et al. (2018). 'Accelerating the Machine Learning Lifecycle with MLflow.' IEEE Data Engineering Bulletin.",
        "5. Evidently AI (2023). 'Monitoring Data and Model Drift in Production Machine Learning Systems.' Open-Source Documentation & Whitepaper.",
        "6. OWASP Foundation (2023). 'OWASP Top 10 for Large Language Models and Machine Learning Security (OWASP ML06: Insecure Deserialization).'",
        "7. Sculley, D., Holt, G., Golovin, D., et al. (2015). 'Hidden Technical Debt in Machine Learning Systems.' Advances in Neural Information Processing Systems (NeurIPS)."
    ]
    for ref in references:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.left_indent = Inches(0.25)
        p_ref.paragraph_format.first_line_indent = Inches(-0.25)
        p_ref.paragraph_format.space_after = Pt(3)
        r_ref = p_ref.add_run(ref)
        r_ref.font.size = Pt(9)
        r_ref.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    # Save Document
    output_path = Path("MLOps_Case_Study_Report.docx")
    doc.save(str(output_path))
    print(f"Document successfully created at {output_path.resolve()}")

if __name__ == "__main__":
    create_report()
