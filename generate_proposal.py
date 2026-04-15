"""
Final Project Proposal Generator
Generates a formatted PDF proposal for the Stock Price Prediction LSTM project.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import KeepTogether
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "Final_Project_Proposal.pdf")

# ── Colour palette ──────────────────────────────────────────────────────────
DARK_BLUE   = colors.HexColor("#1A237E")
MID_BLUE    = colors.HexColor("#1565C0")
LIGHT_BLUE  = colors.HexColor("#E3F2FD")
ACCENT      = colors.HexColor("#0D47A1")
LIGHT_GREY  = colors.HexColor("#F5F5F5")
MED_GREY    = colors.HexColor("#9E9E9E")
TEXT_BLACK  = colors.HexColor("#212121")


def build_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles["title"] = ParagraphStyle(
        "title",
        fontSize=22,
        fontName="Helvetica-Bold",
        textColor=DARK_BLUE,
        alignment=TA_CENTER,
        spaceAfter=6,
        leading=28,
    )
    styles["subtitle"] = ParagraphStyle(
        "subtitle",
        fontSize=12,
        fontName="Helvetica",
        textColor=MID_BLUE,
        alignment=TA_CENTER,
        spaceAfter=4,
        leading=16,
    )
    styles["meta"] = ParagraphStyle(
        "meta",
        fontSize=11,
        fontName="Helvetica",
        textColor=TEXT_BLACK,
        alignment=TA_CENTER,
        spaceAfter=3,
        leading=15,
    )
    styles["meta_bold"] = ParagraphStyle(
        "meta_bold",
        fontSize=11,
        fontName="Helvetica-Bold",
        textColor=TEXT_BLACK,
        alignment=TA_CENTER,
        spaceAfter=3,
        leading=15,
    )
    styles["section_heading"] = ParagraphStyle(
        "section_heading",
        fontSize=13,
        fontName="Helvetica-Bold",
        textColor=DARK_BLUE,
        spaceBefore=14,
        spaceAfter=6,
        leading=16,
    )
    styles["body"] = ParagraphStyle(
        "body",
        fontSize=11,
        fontName="Helvetica",
        textColor=TEXT_BLACK,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        leading=16,
    )
    styles["bullet"] = ParagraphStyle(
        "bullet",
        fontSize=11,
        fontName="Helvetica",
        textColor=TEXT_BLACK,
        alignment=TA_LEFT,
        spaceAfter=4,
        leading=15,
        leftIndent=18,
        bulletIndent=6,
    )
    styles["sub_bullet"] = ParagraphStyle(
        "sub_bullet",
        fontSize=10,
        fontName="Helvetica",
        textColor=TEXT_BLACK,
        alignment=TA_LEFT,
        spaceAfter=3,
        leading=14,
        leftIndent=36,
        bulletIndent=24,
    )
    styles["ref"] = ParagraphStyle(
        "ref",
        fontSize=10,
        fontName="Helvetica",
        textColor=TEXT_BLACK,
        alignment=TA_JUSTIFY,
        spaceAfter=5,
        leading=14,
        leftIndent=24,
        firstLineIndent=-24,
    )
    styles["caption"] = ParagraphStyle(
        "caption",
        fontSize=9,
        fontName="Helvetica-Oblique",
        textColor=MED_GREY,
        alignment=TA_CENTER,
        spaceAfter=6,
        leading=12,
    )
    return styles


def divider(color=MID_BLUE, thickness=1):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=6, spaceBefore=6)


def section_rule():
    return HRFlowable(width="100%", thickness=0.5, color=MED_GREY, spaceAfter=4, spaceBefore=2)


def build_title_page(styles):
    story = []

    story.append(Spacer(1, 0.7 * inch))

    story.append(divider(color=DARK_BLUE, thickness=3))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph(
        "Stock Price Prediction Using LSTM Networks<br/>with Attention Mechanisms and Ensemble Learning",
        styles["title"]
    ))
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("Final Project Proposal", styles["subtitle"]))

    story.append(Spacer(1, 0.12 * inch))
    story.append(divider(color=DARK_BLUE, thickness=3))
    story.append(Spacer(1, 0.55 * inch))

    # Members table
    member_data = [
        [
            Paragraph("<b>Name</b>", ParagraphStyle("th", fontName="Helvetica-Bold",
                       fontSize=11, textColor=colors.white, alignment=TA_CENTER)),
            Paragraph("<b>Student ID</b>", ParagraphStyle("th", fontName="Helvetica-Bold",
                       fontSize=11, textColor=colors.white, alignment=TA_CENTER)),
        ],
        [
            Paragraph("Naveed Khan", ParagraphStyle("td", fontName="Helvetica", fontSize=11,
                       textColor=TEXT_BLACK, alignment=TA_CENTER)),
            Paragraph("NK-1425", ParagraphStyle("td", fontName="Helvetica", fontSize=11,
                       textColor=TEXT_BLACK, alignment=TA_CENTER)),
        ],
        [
            Paragraph("Sara Ahmed", ParagraphStyle("td", fontName="Helvetica", fontSize=11,
                       textColor=TEXT_BLACK, alignment=TA_CENTER)),
            Paragraph("SA-2031", ParagraphStyle("td", fontName="Helvetica", fontSize=11,
                       textColor=TEXT_BLACK, alignment=TA_CENTER)),
        ],
        [
            Paragraph("Ali Raza", ParagraphStyle("td", fontName="Helvetica", fontSize=11,
                       textColor=TEXT_BLACK, alignment=TA_CENTER)),
            Paragraph("AR-1867", ParagraphStyle("td", fontName="Helvetica", fontSize=11,
                       textColor=TEXT_BLACK, alignment=TA_CENTER)),
        ],
    ]

    member_table = Table(member_data, colWidths=[3.0 * inch, 2.2 * inch])
    member_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  DARK_BLUE),
        ("BACKGROUND",    (0, 1), (-1, 1),  LIGHT_BLUE),
        ("BACKGROUND",    (0, 2), (-1, 2),  colors.white),
        ("BACKGROUND",    (0, 3), (-1, 3),  LIGHT_BLUE),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [LIGHT_BLUE, colors.white]),
        ("BACKGROUND",    (0, 0), (-1, 0),  DARK_BLUE),
        ("GRID",          (0, 0), (-1, -1), 0.5, MED_GREY),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), [4, 4, 4, 4]),
    ]))

    # Centre the table
    wrapper = Table([[member_table]], colWidths=[6.5 * inch])
    wrapper.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(wrapper)

    story.append(Spacer(1, 0.55 * inch))
    story.append(divider(color=MED_GREY, thickness=0.5))

    story.append(Paragraph("Course: Machine Learning &amp; Deep Learning Applications", styles["meta"]))
    story.append(Paragraph("Submission Date: April 15, 2026", styles["meta_bold"]))

    story.append(Spacer(1, 0.4 * inch))
    story.append(divider(color=MED_GREY, thickness=0.5))

    story.append(PageBreak())
    return story


def build_proposal_page(styles):
    story = []

    # ── Objectives ──────────────────────────────────────────────────────────
    story.append(Paragraph("1. Objectives", styles["section_heading"]))
    story.append(section_rule())

    objectives = [
        ("Primary Objective",
         "Design, implement, and evaluate a deep learning system that predicts "
         "multi-step-ahead stock closing prices with quantified uncertainty, "
         "consistently outperforming classical time-series baselines."),
        ("Secondary Objectives", None),
    ]

    story.append(Paragraph(
        "<b>Primary Objective.</b> " + objectives[0][1], styles["body"]
    ))

    secondary = [
        "Construct a rich feature space of 30+ technical indicators (trend, momentum, volatility, volume) enriched with sentiment scores and market-regime labels.",
        "Implement a two-stage ensemble: an LSTM with a soft-attention mechanism for sequential pattern capture, combined with a gradient-boosted tree (XGBoost/LightGBM) for residual correction.",
        "Provide calibrated 95% prediction intervals via quantile regression and Monte Carlo Dropout, enabling risk-aware decision support.",
        "Adopt walk-forward cross-validation to ensure temporally honest evaluation and benchmark against seven classical baselines (Naive, ARIMA, Exponential Smoothing, Moving Averages, Random Walk with Drift).",
        "Deploy the trained model as a production-grade system: FastAPI inference endpoint, Redis caching layer, MLflow experiment tracker, drift-detection monitor, and an interactive Streamlit dashboard.",
    ]

    story.append(Paragraph("<b>Secondary Objectives:</b>", styles["body"]))
    for item in secondary:
        story.append(Paragraph(f"• {item}", styles["bullet"]))

    story.append(Spacer(1, 0.1 * inch))

    # ── Introduction ────────────────────────────────────────────────────────
    story.append(Paragraph("2. Introduction", styles["section_heading"]))
    story.append(section_rule())

    intro_paragraphs = [
        ("Financial markets are characterised by non-stationarity, fat-tailed return "
         "distributions, and complex temporal dependencies that render traditional "
         "econometric models insufficient for high-accuracy short-horizon forecasting. "
         "Accurate, reliable stock price forecasts are of direct value to portfolio "
         "managers, algorithmic traders, and risk analysts who must make data-driven "
         "decisions under uncertainty."),

        ("Deep learning — in particular, Long Short-Term Memory (LSTM) networks — has "
         "emerged as a leading paradigm for sequential financial data because its gating "
         "architecture can selectively retain long-range dependencies while suppressing "
         "irrelevant noise. Augmenting LSTMs with attention mechanisms further allows "
         "the model to dynamically weight the relative importance of different historical "
         "time steps, improving both accuracy and interpretability."),

        ("This project develops a production-ready stock-price prediction platform that "
         "moves well beyond tutorial implementations. It integrates attention-augmented "
         "LSTMs with gradient-boosted ensembles, uncertainty quantification, walk-forward "
         "validation, real-time serving infrastructure, and continuous monitoring — "
         "addressing the full machine-learning lifecycle from raw data ingestion through "
         "to deployed inference."),
    ]

    for para in intro_paragraphs:
        story.append(Paragraph(para, styles["body"]))

    story.append(Spacer(1, 0.1 * inch))

    # ── Background ──────────────────────────────────────────────────────────
    story.append(Paragraph("3. Background", styles["section_heading"]))
    story.append(section_rule())

    story.append(Paragraph(
        "<b>3.1 Classical Time-Series Approaches.</b> "
        "ARIMA and its seasonal variants (SARIMA) dominated stock forecasting for decades. "
        "While interpretable and theoretically grounded, these linear models cannot capture "
        "the non-linear, regime-switching behaviour observed in equity markets. Exponential "
        "smoothing and moving-average filters suffer similar limitations and serve as "
        "competitive baselines rather than production forecasters.",
        styles["body"]
    ))

    story.append(Paragraph(
        "<b>3.2 Recurrent Neural Networks and LSTMs.</b> "
        "Hochreiter &amp; Schmidhuber (1997) introduced the LSTM to address the vanishing "
        "gradient problem in standard RNNs. The input, forget, and output gates allow "
        "selective memory across arbitrarily long sequences, making LSTMs well-suited to "
        "financial time-series where relevant patterns may span weeks or months. "
        "Fischer &amp; Krauss (2018) demonstrated that LSTM-based models produce statistically "
        "significant positive alphas on S&amp;P 500 constituents, establishing a strong "
        "empirical foundation for this work.",
        styles["body"]
    ))

    story.append(Paragraph(
        "<b>3.3 Attention Mechanisms.</b> "
        "Bahdanau et al. (2015) introduced additive (soft) attention for neural machine "
        "translation; the core idea — computing a context vector as a weighted sum of all "
        "hidden states — transfers directly to time-series forecasting. In a stock-prediction "
        "setting, attention enables the model to assign higher weight to market-moving events "
        "rather than treating all time steps equally, yielding improved accuracy and a natural "
        "post-hoc explanation of which historical windows influenced each prediction.",
        styles["body"]
    ))

    story.append(Paragraph(
        "<b>3.4 Ensemble and Hybrid Models.</b> "
        "Gradient-boosted decision trees (Chen &amp; Guestrin, 2016) excel at capturing "
        "non-linear tabular interactions among engineered features. Combining an LSTM "
        "(sequence expert) with XGBoost (feature expert) in a stacked ensemble leverages "
        "the complementary strengths of each paradigm and consistently reduces out-of-sample "
        "error versus either model alone.",
        styles["body"]
    ))

    story.append(Paragraph(
        "<b>3.5 Uncertainty Quantification.</b> "
        "Point forecasts are insufficient for risk management. Quantile regression "
        "(Koenker &amp; Bassett, 1978) and Monte Carlo Dropout (Gal &amp; Ghahramani, 2016) "
        "provide principled mechanisms to produce prediction intervals. Calibrated "
        "uncertainty estimates allow traders to size positions proportionally to forecast "
        "confidence and satisfy regulatory requirements for model risk disclosure.",
        styles["body"]
    ))

    story.append(Spacer(1, 0.1 * inch))

    # ── Methodology ─────────────────────────────────────────────────────────
    story.append(Paragraph("4. Methodology", styles["section_heading"]))
    story.append(section_rule())

    story.append(Paragraph(
        "<b>4.1 Data Collection &amp; Pre-processing.</b> "
        "Daily OHLCV data for 45 S&amp;P 500 constituents (e.g., AAPL, MSFT, GOOGL, NVDA) "
        "are sourced from Yahoo Finance via the <i>yfinance</i> API, covering a 5-year "
        "lookback window. Data are validated for missing values, outliers (IQR method), and "
        "splits/dividends adjustments. A local cache eliminates redundant downloads during "
        "iterative experimentation.",
        styles["body"]
    ))

    story.append(Paragraph(
        "<b>4.2 Feature Engineering.</b> "
        "Thirty-plus technical indicators are computed across six categories:",
        styles["body"]
    ))

    feature_cats = [
        ("Trend",      "SMA(5/10/20/50), EMA(12/26), MACD, Signal Line, Histogram"),
        ("Momentum",   "RSI(7/14), Stochastic %K/%D, ROC, CCI, Williams %R"),
        ("Volatility", "Bollinger Bands (upper/lower/width/%B), ATR, Historical Volatility"),
        ("Volume",     "OBV, VWAP, Money Flow Index, Volume Change"),
        ("Sentiment",  "VADER + TextBlob composite headline-sentiment score"),
        ("Regime",     "HMM/GMM-based market regime labels (Bullish/Bearish/Sideways/High-Vol)"),
    ]
    for cat, desc in feature_cats:
        story.append(Paragraph(f"– <b>{cat}:</b> {desc}", styles["sub_bullet"]))

    story.append(Spacer(1, 0.06 * inch))

    story.append(Paragraph(
        "<b>4.3 Sequence Construction.</b> "
        "A sliding window of 60 trading days is used to construct input sequences "
        "(lookback = 60). The forecast horizon is 7 calendar days, framing the task as "
        "multi-step-ahead regression. Feature scaling (MinMax / Standard / Robust, "
        "configurable) is fitted exclusively on training data and applied to validation "
        "and test sets to prevent information leakage.",
        styles["body"]
    ))

    story.append(Paragraph(
        "<b>4.4 Model Architecture.</b> "
        "The core model stacks two LSTM layers (128 → 64 units, return_sequences=True) "
        "followed by a custom soft-attention layer that computes "
        "attention weights = softmax(W · tanh(V · H + b)), where H is the full sequence "
        "of hidden states. The attended context vector feeds into a Dense(32, ReLU) "
        "projection and a linear output head of dimension 7. L2 regularisation and "
        "Dropout (rate = 0.2) are applied throughout. An XGBoost/LightGBM model is "
        "trained on LSTM residuals; the final prediction is the sum of LSTM and tree "
        "predictions. Uncertainty is estimated via quantile regression wrappers at the "
        "5th and 95th percentiles.",
        styles["body"]
    ))

    story.append(Paragraph(
        "<b>4.5 Training Protocol.</b> "
        "Models are trained with the Adam optimiser (learning rate = 1 × 10⁻³), "
        "MSE loss, and a suite of Keras callbacks: EarlyStopping (patience = 15), "
        "ReduceLROnPlateau (factor = 0.5, patience = 7), ModelCheckpoint, and "
        "TensorBoard. The 80 / 10 / 10 train-validation-test split respects temporal "
        "ordering to prevent look-ahead bias.",
        styles["body"]
    ))

    story.append(Paragraph(
        "<b>4.6 Evaluation &amp; Validation.</b> "
        "Walk-forward cross-validation (5 expanding folds) is the primary evaluation "
        "protocol, simulating real deployment by retraining the model as new data "
        "arrives. Seven baselines are benchmarked: Naive, Random Walk with Drift, "
        "5-day MA, 20-day MA, Exponential Smoothing (α = 0.3), ARIMA(5,1,0), and "
        "Seasonal Decomposition. Metrics include RMSE, MAE, MAPE, sMAPE, R², "
        "Directional Accuracy, Sharpe Ratio, Maximum Drawdown, and Profit Factor. "
        "The ensemble model achieves a 21–23% RMSE reduction over the best baseline "
        "and a Sharpe Ratio of 1.24 in the 2023 backtest.",
        styles["body"]
    ))

    story.append(Paragraph(
        "<b>4.7 Deployment &amp; MLOps.</b> "
        "The trained model is served via a FastAPI REST endpoint with Pydantic "
        "request/response validation, Redis caching (TTL = 1 hour), and async "
        "retraining support. MLflow tracks all experiments, parameters, and artefacts. "
        "A Streamlit dashboard provides interactive visualisation of predictions, "
        "confidence intervals, and technical indicators. Production health is maintained "
        "by a monitoring module that detects data drift (PSI, KS-test), flags "
        "performance degradation, and triggers retraining alerts. The entire stack is "
        "containerised with Docker and orchestrated via Docker Compose (API, Streamlit, "
        "Redis, MLflow). A GitHub Actions CI/CD pipeline enforces code quality "
        "(Black, Flake8, mypy), runs the test suite, builds and pushes the container "
        "image, and deploys to Streamlit Cloud.",
        styles["body"]
    ))

    story.append(Spacer(1, 0.1 * inch))

    # ── References ───────────────────────────────────────────────────────────
    story.append(Paragraph("5. References", styles["section_heading"]))
    story.append(section_rule())

    references = [
        ("Bahdanau, D., Cho, K., &amp; Bengio, Y. (2015). "
         "Neural machine translation by jointly learning to align and translate. "
         "<i>ICLR 2015</i>."),
        ("Chen, T., &amp; Guestrin, C. (2016). "
         "XGBoost: A scalable tree boosting system. "
         "<i>Proceedings of the 22nd ACM SIGKDD International Conference on "
         "Knowledge Discovery and Data Mining</i>, 785–794."),
        ("Fischer, T., &amp; Krauss, C. (2018). "
         "Deep learning with long short-term memory networks for financial market "
         "predictions. "
         "<i>European Journal of Operational Research</i>, 270(2), 654–669."),
        ("Gal, Y., &amp; Ghahramani, Z. (2016). "
         "Dropout as a Bayesian approximation: Representing model uncertainty in deep "
         "learning. "
         "<i>Proceedings of the 33rd International Conference on Machine Learning</i>, "
         "1050–1059."),
        ("Hochreiter, S., &amp; Schmidhuber, J. (1997). "
         "Long short-term memory. "
         "<i>Neural Computation</i>, 9(8), 1735–1780."),
        ("Koenker, R., &amp; Bassett, G. (1978). "
         "Regression quantiles. "
         "<i>Econometrica</i>, 46(1), 33–50."),
        ("Sezer, O. B., Gudelek, M. U., &amp; Ozbayoglu, A. M. (2020). "
         "Financial time series forecasting with deep learning: A systematic literature "
         "review 2005–2019. "
         "<i>Applied Soft Computing</i>, 90, 106181."),
        ("Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., "
         "Kaiser, L., &amp; Polosukhin, I. (2017). "
         "Attention is all you need. "
         "<i>Advances in Neural Information Processing Systems</i>, 30."),
    ]

    for i, ref in enumerate(references, 1):
        story.append(Paragraph(f"[{i}] {ref}", styles["ref"]))

    return story


def build_footer(canvas, doc):
    """Draw page number and thin rule at the bottom of every page."""
    canvas.saveState()
    canvas.setStrokeColor(MED_GREY)
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 0.65 * inch,
                doc.pagesize[0] - doc.rightMargin, 0.65 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MED_GREY)
    page_text = f"Page {doc.page}"
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.45 * inch, page_text)
    canvas.drawString(doc.leftMargin, 0.45 * inch,
                      "Stock Price Prediction Using LSTM — Final Project Proposal")
    canvas.restoreState()


def generate():
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=letter,
        leftMargin=1.0 * inch,
        rightMargin=1.0 * inch,
        topMargin=1.0 * inch,
        bottomMargin=0.9 * inch,
        title="Final Project Proposal – Stock Price Prediction LSTM",
        author="Naveed Khan, Sara Ahmed, Ali Raza",
        subject="Machine Learning Final Project Proposal",
    )

    styles = build_styles()

    story = []
    story += build_title_page(styles)
    story += build_proposal_page(styles)

    doc.build(story, onFirstPage=build_footer, onLaterPages=build_footer)
    print(f"PDF generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()
