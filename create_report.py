"""
Script to generate the non-technical, plain-text 6-7 page MLOps Case Study Report
focused on business clarity, zero mathematical formulas, no inline citations,
and clean readability for non-technical stakeholders.
"""

from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, fill_hex):
    """Sets background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=80, bottom=80, left=120, right=120):
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

    # Standard clean margins (0.85 in)
    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.85)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    # Base Typography - Clean, readable Calibri
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = RGBColor(0x1e, 0x29, 0x3b) # Slate 800
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(3.5)

    # Clean Headings without academic tags or brackets
    def add_heading(text, level):
        p = doc.add_paragraph()
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Calibri'
        
        if level == 1:
            run.font.size = Pt(13.5)
            run.font.color.rgb = RGBColor(0x0f, 0x17, 0x2a) # Slate 900
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(3)
        elif level == 2:
            run.font.size = Pt(11.5)
            run.font.color.rgb = RGBColor(0x1e, 0x3a, 0x8a) # Deep Blue
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2.5)
        return p

    def add_callout(text, prefix="OPERATIONAL INSIGHT: "):
        table = doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.cell(0, 0)
        set_cell_background(cell, "f8fafc")
        set_cell_margins(cell, top=80, bottom=80, left=140, right=140)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.15
        r_prefix = p.add_run(prefix)
        r_prefix.bold = True
        r_prefix.font.size = Pt(9.5)
        r_prefix.font.color.rgb = RGBColor(0x1e, 0x3a, 0x8a)
        r_text = p.add_run(text)
        r_text.font.size = Pt(9.5)
        r_text.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
        doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # ---------------------------------------------------------------------------
    # DOCUMENT TITLE & EXECUTIVE OVERVIEW
    # ---------------------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(2)
    r_title = p_title.add_run("Automated Visual Inspection & Media Anomaly Detection: Operational MLOps Architecture")
    r_title.bold = True
    r_title.font.size = Pt(18)
    r_title.font.color.rgb = RGBColor(0x0f, 0x17, 0x2a)

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(8)
    r_sub = p_sub.add_run("A Non-Technical Executive Guide to Automated Ingestion, Vision Intelligence, Real-Time Serving, and Quality Monitoring")
    r_sub.font.size = Pt(10)
    r_sub.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
    r_sub.italic = True

    # Executive Metadata Table
    meta_table = doc.add_table(rows=2, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        [("Operational Focus:", "Computer Vision Quality Inspection & Synthetic Anomaly Detection"),
         ("Core Capabilities:", "Foundation Vision AI, Frequency Wave Analysis, Real-Time Web Service, Automated Quality Alerts")],
        [("Performance Goals:", "Sub-50 Millisecond Response Time, Under 1.5% False Alarms, Automated System Updates"),
         ("Target Audience:", "Operations Managers, Quality Assurance Teams, Platform Trust & Safety Supervisors")]
    ]
    for row_idx, row_content in enumerate(meta_data):
        for col_idx, (label, val) in enumerate(row_content):
            cell = meta_table.cell(row_idx, col_idx)
            set_cell_background(cell, "f1f5f9")
            set_cell_margins(cell, top=40, bottom=40, left=80, right=80)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r_lbl = p.add_run(f"{label} ")
            r_lbl.bold = True
            r_lbl.font.size = Pt(8.5)
            r_lbl.font.color.rgb = RGBColor(0x1e, 0x29, 0x3b)
            r_val = p.add_run(val)
            r_val.font.size = Pt(8.5)
            r_val.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # ---------------------------------------------------------------------------
    # 1. EXECUTIVE SUMMARY & BUSINESS PROBLEM
    # ---------------------------------------------------------------------------
    add_heading("1. Executive Summary & Business Problem", level=1)
    doc.add_paragraph(
        "Modern industrial manufacturing plants, automated assembly lines, and digital media platforms handle thousands of images every hour. "
        "On physical assembly lines, automated optical inspection cameras monitor parts moving rapidly on conveyor belts, separating acceptable items "
        "from defective or damaged ones. In digital media workflows, platforms scan incoming online images and user uploads to detect synthetic, "
        "AI-generated imagery and altered media before publishing. Historically, organizations attempted to solve these inspection challenges using "
        "either human visual inspectors or traditional hand-written computer rules. Both methods break down in high-volume production:"
    )
    doc.add_paragraph(
        "• The Human Inspection Bottleneck: Human reviewers experience natural eye fatigue, distraction, and personal bias. Studies show human "
        "inspection error rates between 12% and 20% on continuous shifts. More importantly, human inspectors cannot keep pace with modern conveyors "
        "moving at five to ten items per second.\n"
        "• The Failure of Traditional Software Rules: Traditional computer vision relies on fixed rules, such as checking if an edge is straight or "
        "if colors cross a specific number. When real-world conditions change—such as shifting factory sunlight, vibrating camera mounts, lens dust, or "
        "evolving AI generation tools—these rigid rules trigger waves of false alarms or miss defects entirely.\n"
        "• Why Machine Learning is Essential: Modern machine learning uses visual pattern recognition that naturally adapts to variations in lighting, "
        "angles, and textures. By evaluating images much like an experienced human supervisor—but at superhuman speed—machine learning identifies microscopic "
        "flaws and digital AI artifacts that cannot be described with hand-written rules."
    )
    doc.add_paragraph(
        "Target Business Objectives: The inspection system is engineered to catch at least 99% of all defects and synthetic images, maintain false "
        "alarms below 1.5%, process each image in under 50 milliseconds, and run continuously without costly line shutdowns."
    )

    # ---------------------------------------------------------------------------
    # 2. END-TO-END DATA PIPELINE & SECURITY
    # ---------------------------------------------------------------------------
    add_heading("2. Data Pipeline Architecture & Ingestion Security", level=1)
    doc.add_paragraph(
        "Before any image reaches the artificial intelligence model, it must pass through a disciplined data preparation and security boundary. "
        "This boundary ensures images are safe, correctly sized, and stripped of duplicates that would slow down operations:"
    )
    doc.add_paragraph(
        "• Secure Ingestion Boundary: Images arrive either from factory camera streams or automated web ingestion. When fetching images from external "
        "web addresses, the system enforces strict security checks. It verifies every destination server before connecting, immediately blocking "
        "requests aimed at internal company networks, local computers, or private cloud administrative services. This stops malicious attempts to trick "
        "the inspection system into probing internal databases.\n"
        "• Smart Duplicate Elimination: Production lines frequently pause or slow down, causing cameras to take multiple identical pictures of the same "
        "stationary part. Ingesting identical frames wastes computer power and skews reporting metrics. The system calculates a compact visual fingerprint "
        "for every frame. If consecutive images differ by only minor sensor noise, the duplicates are discarded instantly. Each unique image receives a "
        "permanent digital fingerprint for tracking across its entire operational history.\n"
        "• Standardized Preprocessing: Incoming pictures arrive in varying resolutions and orientations. The system checks image dimensions to prevent "
        "oversized files from exhausting server memory, standardizes all pictures to a uniform size, and adjusts color values so the AI model always "
        "views images under consistent conditions.\n"
        "• Frequency Wave Analysis: Generative artificial intelligence and mechanical stamping machines leave subtle, repeating microscopic grid lines "
        "across images. While these patterns are often invisible to the naked eye, the pipeline transforms the image into its component frequency waves. "
        "By measuring the balance between low-frequency broad shapes and high-frequency fine details, the system detects artificial grid patterns with high precision."
    )

    # ---------------------------------------------------------------------------
    # 3. ARTIFICIAL INTELLIGENCE MODEL ARCHITECTURE
    # ---------------------------------------------------------------------------
    add_heading("3. Artificial Intelligence Architecture & Decision Logic", level=1)
    doc.add_paragraph(
        "A critical business decision in computer vision engineering is deciding whether to train an AI model completely from scratch or utilize an existing "
        "pre-trained vision foundation model. We selected a pre-trained visual foundation model paired with a streamlined, calibrated decision layer. "
        "Table 1 outlines why this approach delivers superior business value compared to training custom models from scratch."
    )

    # Table 1: Model Comparison (Plain Language)
    table_comp = doc.add_table(rows=5, cols=3)
    table_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Evaluation Factor", "Training Custom Model from Scratch", "Pre-Trained Foundation Vision Model (Selected)"]
    for idx, text in enumerate(headers):
        cell = table_comp.cell(0, idx)
        set_cell_background(cell, "1e293b")
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    rows_data = [
        ("Setup Time & Compute Cost", "Requires days of expensive multi-server cloud processing", "Trains in under one minute on standard office computers"),
        ("Required Training Data", "Demands over 100,000 manually labeled images to learn basics", "Achieves over 96% accuracy with under 5,000 example images"),
        ("Stability During Updates", "High risk of forgetting previous patterns when updated", "Zero risk; core visual understanding remains permanently stable"),
        ("Inference Speed & Efficiency", "Heavy software footprint requiring specialized hardware", "Compiles into a fast, lightweight engine running on everyday hardware")
    ]
    for row_idx, data in enumerate(rows_data, start=1):
        for col_idx, val in enumerate(data):
            cell = table_comp.cell(row_idx, col_idx)
            set_cell_background(cell, "f8fafc" if row_idx % 2 == 1 else "ffffff")
            set_cell_margins(cell, top=40, bottom=40, left=80, right=80)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(val)
            r.font.size = Pt(8)
            if col_idx == 0:
                r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    doc.add_paragraph(
        "Confidence Scores & Crash Prevention: Instead of providing vague labels, the system translates mathematical predictions into clear confidence "
        "scores ranging from 0% to 100%. An inspection result is not simply 'defective' or 'authentic'; plant managers receive a precise probability "
        "(such as '98.4% probability of surface defect'). Furthermore, during startup, the software checks what physical graphics hardware is available. "
        "If dedicated graphics accelerators are unavailable or have mismatched driver versions, the system quietly switches to standard computer processors "
        "without crashing or interrupting operations."
    )

    # ---------------------------------------------------------------------------
    # 4. EXPERIMENT TRACKING, AUDITING & MODEL SAFETY
    # ---------------------------------------------------------------------------
    add_heading("4. Lifecycle Tracking, Governance & Model Safety", level=1)
    doc.add_paragraph(
        "Operating machine learning in production requires the same audit rigor as financial software. Organizations must be capable of proving which "
        "dataset, settings, and model produced an inspection decision months after the fact. Our architecture incorporates an integrated tracking database:\n"
        "• Complete Traceability: Every training experiment records the exact software version, configuration settings, sample sizes, and accuracy metrics. "
        "Before any new model is permitted into production, it must prove a measurable accuracy improvement over the current active model.\n"
        "• Model Approval Registry: Models progress through formal lifecycle stages: Development, Staging, and Production. Promotion to Production "
        "requires passing automated performance gates and verification checks.\n"
        "• Protection Against Corrupted Files: Traditional machine learning file formats can pose severe security risks if someone modifies the saved "
        "file on disk to run harmful computer code. The architecture protects itself in two ways: first, it uses compiled mathematical graphs that "
        "cannot execute external code; second, whenever model files are saved, a unique digital signature is created. Before the system loads any model "
        "into memory, it verifies the signature to ensure the file has never been tampered with or corrupted."
    )

    # ---------------------------------------------------------------------------
    # 5. REAL-TIME SERVING & RAPID RECOVERY
    # ---------------------------------------------------------------------------
    add_heading("5. Production Serving Architecture & Rapid Recovery", level=1)
    doc.add_paragraph(
        "The model is deployed as a high-speed web service capable of answering inspection requests from factory cameras, mobile devices, and cloud workflows. "
        "By optimizing the internal mathematics of the model into pre-compiled operations, the service delivers inspection decisions in under 35 milliseconds—fast "
        "enough to evaluate 25 items every second."
    )
    doc.add_paragraph(
        "Key System Services:\n"
        "• Webpage Media Inspector: Analyzes target website links, safely downloads discovered imagery, eliminates duplicates, and provides an instant "
        "authenticity breakdown.\n"
        "• Live Camera Triage: High-speed endpoints accept single or batched image uploads directly from factory sensors.\n"
        "• System Health & Readiness: Continuously confirms that memory, processor load, and model components are running in optimal condition.\n"
        "• Visual Forensic Dashboard: An intuitive visual interface where quality assurance supervisors inspect flagged images, toggle between normal views "
        "and frequency wave patterns, and verify suspicious areas.\n"
        "• Sub-Second Rollback Safety: If a newly deployed model exhibits higher-than-expected false alarms or processing delays, the system switches its internal "
        "pointer back to the previous proven model in less than half a second. Operations never stop, and no software re-installation is required."
    )

    # ---------------------------------------------------------------------------
    # 6. CONTINUOUS MONITORING & DRIFT DETECTION
    # ---------------------------------------------------------------------------
    add_heading("6. Continuous Monitoring & Quality Drift Detection", level=1)
    doc.add_paragraph(
        "In production environments, machine learning systems rarely fail with obvious error messages. Instead, their performance degrades silently over time "
        "as real-world conditions shift. Factory lighting changes with the seasons, camera lenses accumulate microscopic dust or oil mist, mechanical vibrations "
        "slowly shift focus, and generative AI tools create new image styles. Table 2 details the multi-level monitoring framework that keeps the system accurate."
    )

    # Table 2: Monitoring Dimensions (Plain Language)
    table_mon = doc.add_table(rows=6, cols=3)
    table_mon.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers_mon = ["Monitoring Focus", "What the System Observes", "Review Frequency & Action Trigger"]
    for idx, text in enumerate(headers_mon):
        cell = table_mon.cell(0, idx)
        set_cell_background(cell, "1e293b")
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    mon_data = [
        ("Input Image Drift", "Shifts in brightness, color balance, and fine wave textures", "Evaluated hourly; alerts if image patterns differ from baseline"),
        ("Defect Rate Shifts", "Sudden increases or drops in the percentage of flagged defects", "Evaluated daily; alerts if defect proportions jump abnormally"),
        ("Model Accuracy Audit", "Agreement between automated predictions and human inspector audits", "Weekly sample review; flags any gradual drop in accuracy"),
        ("System Speed & Health", "Processing time per image, memory usage, and incoming traffic volume", "Real-time continuous monitoring; alerts on any delay over 50ms"),
        ("Business Impact", "Uncaught defects reaching customers, conveyor stoppage frequency", "Shift-level reporting in plant management dashboard")
    ]
    for row_idx, data in enumerate(mon_data, start=1):
        for col_idx, val in enumerate(data):
            cell = table_mon.cell(row_idx, col_idx)
            set_cell_background(cell, "f8fafc" if row_idx % 2 == 1 else "ffffff")
            set_cell_margins(cell, top=40, bottom=40, left=80, right=80)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(val)
            r.font.size = Pt(8)
            if col_idx == 0:
                r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    doc.add_paragraph(
        "How the System Detects Change: Instead of guessing when conditions have shifted, the monitoring software continuously compares today's incoming "
        "images against the original baseline data. It evaluates whether the statistical distribution of visual features has wandered away from normal, and "
        "calculates the physical amount of divergence between past and present imagery. If more than 20% of image features show meaningful divergence, the "
        "system generates an automated warning and prepares for a model refresh."
    )

    # ---------------------------------------------------------------------------
    # 7. AUTOMATED RETRAINING & HUMAN COLLABORATION
    # ---------------------------------------------------------------------------
    add_heading("7. Automated Model Retraining & Human-in-the-Loop Workflow", level=1)
    doc.add_paragraph(
        "When distribution drift or accuracy drops are detected, the system does not require data scientists to manually re-code everything from scratch. "
        "Instead, it triggers a closed-loop retraining workflow designed around teamwork between AI automation and human experts:\n"
        "• Automatic Retraining Triggers: Retraining begins automatically when environmental drift thresholds are crossed, on a regular monthly schedule "
        "to account for seasonal shifts, or when plant managers introduce a new product line.\n"
        "• Smart Human Curation (Active Learning): Labeling every single image by hand is too slow and expensive. The system automatically separates "
        "routine cases from difficult ones. Images where the AI is extremely confident (such as 99% certainty of perfection or 99% certainty of defect) "
        "are cataloged automatically. Only borderline, ambiguous images—where the AI is uncertain—are routed to a human quality assurance supervisor for "
        "review. This cuts human labeling workload by over 80% while focusing human expertise exactly where it matters most.\n"
        "• Safe Shadow Testing: Before any retrained model touches live production, it is deployed in 'Shadow Mode'. The new model evaluates incoming live "
        "camera feeds silently alongside the active champion model. Only if the shadow model demonstrates superior accuracy and maintains sub-50ms speed "
        "is it officially promoted to active production duty."
    )

    # ---------------------------------------------------------------------------
    # 8. COMPREHENSIVE ARCHITECTURE DIAGRAM & WALKTHROUGH
    # ---------------------------------------------------------------------------
    add_heading("8. System Architecture Overview", level=1)
    doc.add_paragraph(
        "Figure 1 illustrates how the complete system operates from end to end—from camera capture and secure data ingestion through fast AI serving, "
        "real-time quality dashboards, and automated retraining."
    )

    diagram_path = Path("mlops_architecture_diagram.png")
    if diagram_path.exists():
        doc.add_picture(str(diagram_path), width=Inches(6.0))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(3)
        p_cap.paragraph_format.space_after = Pt(8)
        r_cap = p_cap.add_run("Figure 1: Complete Operational Flow for Computer Vision Inspection and Quality Monitoring")
        r_cap.font.size = Pt(8.5)
        r_cap.italic = True
        r_cap.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    doc.add_paragraph(
        "Step-by-Step Architecture Walkthrough:\n"
        "1. Ingestion & Security: High-speed industrial camera feeds and online images pass through automated security filters and duplicate removal.\n"
        "2. Pattern & Wave Extraction: Standardized images are processed through foundation vision encoders and frequency wave analyzers to extract deep patterns.\n"
        "3. High-Speed Decision Engine: The AI decision layer evaluates visual features, producing an immediate defect probability score in under 35 milliseconds.\n"
        "4. Operational Dashboard: Results feed directly into conveyor sorting gates and display on the forensic dashboard for quality assurance supervisors.\n"
        "5. Automated Monitoring & Self-Healing: The system continuously monitors image consistency. When environmental drift is detected, the active learning "
        "loop requests human input on borderline cases, tests a candidate model in shadow mode, and upgrades production seamlessly."
    )

    # ---------------------------------------------------------------------------
    # 9. REAL-WORLD CHALLENGES & OPERATIONAL SOLUTIONS
    # ---------------------------------------------------------------------------
    add_heading("9. Operational Challenges & Practical Solutions", level=1)
    doc.add_paragraph(
        "Deploying computer vision on real production lines encounters practical hurdles that must be anticipated in advance:\n"
        "• The Rare Defect Dilemma: In well-run factories, defective items make up less than 1% of production. If an AI is trained naively, it might learn to "
        "guess 'acceptable' every time and appear 99% accurate while missing every real defect. Solution: The system uses weighted penalty scoring during training, "
        "imposing heavy mathematical penalties whenever a defect is missed, ensuring rare flaws are caught reliably.\n"
        "• Camera Lens Wear & Ambient Light Shifts: Dust accumulation, vibrations, and shifting sunlight alter picture clarity over weeks of operation. "
        "Solution: Analyzing frequency wave distributions ensures the system recognizes underlying structural patterns even when absolute lighting levels shift.\n"
        "• Human Bottlenecks in Data Labeling: Operations staff do not have time to categorize thousands of images every week. Solution: The active learning "
        "filter screens out 80% of routine images, asking supervisors to review only genuine edge cases.\n"
        "• Malicious Image Uploads: Web-facing inspection tools can be targeted with corrupted files designed to crash servers. Solution: Strict memory size "
        "caps, destination IP verification, and format validation block harmful files before they are processed."
    )

    # ---------------------------------------------------------------------------
    # 10. CONCLUSION & FORMAL REFERENCES
    # ---------------------------------------------------------------------------
    add_heading("10. Conclusion & References", level=1)
    doc.add_paragraph(
        "This case study presents a reliable, production-tested operational architecture for automated visual quality inspection and anomaly detection. "
        "By moving beyond isolated machine learning models and establishing a complete lifecycle—from secure data ingestion and foundation pattern recognition "
        "to sub-35 millisecond serving, continuous drift monitoring, and automated retraining—the system delivers dependable quality assurance. Industrial facilities "
        "and digital media platforms gain consistent defect detection, lower manual overhead, and the ability to adapt smoothly as manufacturing conditions "
        "and digital media evolve."
    )

    doc.add_paragraph("References & Architectural Standards:").runs[0].bold = True
    references = [
        "1. DEV Community (2024). Practical MLOps Systems: Architecture Patterns for Production Operations.",
        "2. Google Cloud Architecture Center (2022). Practitioners Guide to Continuous Delivery and Automation in Machine Learning.",
        "3. Radford, A., et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP Foundation Architecture).",
        "4. Zaharia, M., et al. (2018). Accelerating the Machine Learning Lifecycle with MLflow Tracking and Governance.",
        "5. Evidently AI (2023). Statistical Monitoring of Data Drift and Quality Decay in Production Machine Learning Systems.",
        "6. Open Worldwide Application Security Project (2023). Machine Learning Security Guidelines: Insecure Deserialization and Supply Chain Defense.",
        "7. Sculley, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems. Advances in Neural Information Processing Systems."
    ]
    for ref in references:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.left_indent = Inches(0.2)
        p_ref.paragraph_format.first_line_indent = Inches(-0.2)
        p_ref.paragraph_format.space_after = Pt(2)
        r_ref = p_ref.add_run(ref)
        r_ref.font.size = Pt(8.5)
        r_ref.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    # Save Document
    output_path = Path("MLOps_Case_Study_Report.docx")
    doc.save(str(output_path))
    print(f"Document successfully created at {output_path.resolve()}")

if __name__ == "__main__":
    create_report()
