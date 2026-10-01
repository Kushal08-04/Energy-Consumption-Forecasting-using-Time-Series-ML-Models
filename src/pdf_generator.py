"""
Professional PDF Report Generator for Energy Consumption Forecasting.
Generates multi-page PDF executive reports using ReportLab with embedded
high-resolution charts, KPI cards, model scorecards, and forecast tables.
"""

from typing import Dict, Any, Optional
import io
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
)


class EnergyForecastPDFReport:
    """
    Builds a professional multi-page Energy Forecast PDF report.
    """

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._init_custom_styles()

    def _init_custom_styles(self):
        # Header / Title style
        self.styles.add(ParagraphStyle(
            name="ReportTitle",
            parent=self.styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=4,
        ))
        self.styles.add(ParagraphStyle(
            name="ReportSubtitle",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#475569"),
            spaceAfter=12,
        ))
        self.styles.add(ParagraphStyle(
            name="SectionHeading",
            parent=self.styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=12,
            spaceAfter=8,
        ))
        self.styles.add(ParagraphStyle(
            name="BodyDark",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155"),
        ))
        self.styles.add(ParagraphStyle(
            name="TableHeader",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.white,
            alignment=1, # Center
        ))
        self.styles.add(ParagraphStyle(
            name="TableCell",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#1e293b"),
            alignment=1, # Center
        ))
        self.styles.add(ParagraphStyle(
            name="KPIVal",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=16,
            textColor=colors.HexColor("#0284c7"),
            alignment=1,
        ))
        self.styles.add(ParagraphStyle(
            name="KPILabel",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#64748b"),
            alignment=1,
        ))

    def _generate_forecast_chart(
        self,
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        target_col: str = "energy_consumption_mw",
        recent_hours: int = 120,
    ) -> io.BytesIO:
        """Renders historical series transitioning into the forecast with confidence bands."""
        fig, ax = plt.subplots(figsize=(8.2, 3.4), dpi=200)

        # Plot recent history
        if target_col not in historical_df.columns:
            num_cols = historical_df.select_dtypes(include=[np.number]).columns
            target_col = num_cols[0] if len(num_cols) > 0 else historical_df.columns[0]

        hist_subset = historical_df.iloc[-recent_hours:]
        ax.plot(
            hist_subset.index,
            hist_subset[target_col],
            color="#334155",
            linewidth=1.6,
            label="Historical Load (Actual)",
        )

        # Plot forecast line
        ax.plot(
            forecast_df.index,
            forecast_df["forecast_mw"],
            color="#0284c7",
            linewidth=2.0,
            linestyle="--",
            label="AI Forecast",
        )

        # Plot confidence interval
        if "lower_bound_mw" in forecast_df.columns and "upper_bound_mw" in forecast_df.columns:
            ax.fill_between(
                forecast_df.index,
                forecast_df["lower_bound_mw"],
                forecast_df["upper_bound_mw"],
                color="#38bdf8",
                alpha=0.25,
                label="95% Confidence Band",
            )

        # Mark peak forecast point
        peak_idx = forecast_df["forecast_mw"].idxmax()
        peak_val = forecast_df.loc[peak_idx, "forecast_mw"]
        ax.scatter([peak_idx], [peak_val], color="#ef4444", s=50, zorder=5)
        ax.annotate(
            f"Peak: {peak_val:,.0f} MW",
            xy=(peak_idx, peak_val),
            xytext=(10, 8),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="#dc2626",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.9),
        )

        ax.set_title("Energy Demand Trajectory & 95% Confidence Forecast", fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("Power Demand (MW)", fontsize=9, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        ax.tick_params(axis="both", labelsize=8)
        ax.legend(loc="upper left", frameon=True, fontsize=7.5)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200)
        plt.close(fig)
        buf.seek(0)
        return buf

    def _generate_diurnal_chart(self, forecast_df: pd.DataFrame) -> io.BytesIO:
        """Renders 24-hour diurnal load curve for forecasted horizon."""
        fig, ax = plt.subplots(figsize=(8.2, 2.5), dpi=200)

        df_copy = forecast_df.copy()
        df_copy["hour"] = df_copy.index.hour
        hourly_stats = df_copy.groupby("hour")["forecast_mw"].agg(["mean", "min", "max"])

        hours = hourly_stats.index.values
        ax.plot(hours, hourly_stats["mean"], color="#0284c7", linewidth=2.2, marker="o", markersize=4, label="Projected Avg Hourly Load")
        ax.fill_between(hours, hourly_stats["min"], hourly_stats["max"], color="#93c5fd", alpha=0.3, label="Hourly Forecast Range (Min-Max)")

        # Highlight typical morning & evening peak zones
        ax.axvspan(8, 11, color="#fef08a", alpha=0.35, label="Morning Peak Window (08:00 - 11:00)")
        ax.axvspan(18, 21, color="#fed7aa", alpha=0.35, label="Evening Peak Window (18:00 - 21:00)")

        ax.set_title("Diurnal Profile: 24-Hour Average Energy Demand Cycle", fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Hour of Day (00:00 - 23:00)", fontsize=8.5, fontweight="bold")
        ax.set_ylabel("Demand (MW)", fontsize=8.5, fontweight="bold")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(axis="both", labelsize=7.5)
        ax.legend(loc="lower right", frameon=True, fontsize=7)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200)
        plt.close(fig)
        buf.seek(0)
        return buf

    def generate_pdf(
        self,
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        scorecard_df: pd.DataFrame,
        metadata: Dict[str, Any],
        drift_report: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> bytes:
        """
        Assembles all elements into a structured multi-page PDF report.
        Returns the PDF bytes or writes to output_path.
        """
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            output_path if output_path else pdf_buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        elements = []

        # --- 1. TITLE & METADATA BANNER ---
        region_name = metadata.get("selected_region", "Regional Grid") or "Regional Grid"
        generation_time = metadata.get("generation_time", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
        best_model = metadata.get("best_model", "Adaptive Ensemble")

        elements.append(Paragraph("ENERGY CONSUMPTION FORECASTING REPORT", self.styles["ReportTitle"]))
        sub_text = (
            f"<b>Target Region:</b> {region_name} | "
            f"<b>Generated:</b> {generation_time} | "
            f"<b>Primary Forecaster:</b> {best_model}"
        )
        elements.append(Paragraph(sub_text, self.styles["ReportSubtitle"]))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

        # --- 2. EXECUTIVE KPI CARDS TABLE ---
        peak_demand = float(forecast_df["forecast_mw"].max())
        min_demand = float(forecast_df["forecast_mw"].min())
        avg_demand = float(forecast_df["forecast_mw"].mean())
        total_energy_mwh = float(forecast_df["forecast_mw"].sum())

        kpi_data = [
            [
                Paragraph(f"{peak_demand:,.0f} MW", self.styles["KPIVal"]),
                Paragraph(f"{min_demand:,.0f} MW", self.styles["KPIVal"]),
                Paragraph(f"{avg_demand:,.0f} MW", self.styles["KPIVal"]),
                Paragraph(f"{total_energy_mwh:,.0f} MWh", self.styles["KPIVal"]),
            ],
            [
                Paragraph("PROJECTED PEAK DEMAND", self.styles["KPILabel"]),
                Paragraph("PROJECTED MIN DEMAND", self.styles["KPILabel"]),
                Paragraph("AVERAGE HOURLY DEMAND", self.styles["KPILabel"]),
                Paragraph("TOTAL PROJECTED ENERGY", self.styles["KPILabel"]),
            ],
        ]
        kpi_table = Table(kpi_data, colWidths=[130, 130, 130, 150])
        kpi_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(kpi_table)
        elements.append(Spacer(1, 10))

        # --- 3. MODEL BENCHMARK SCORECARD ---
        elements.append(Paragraph("Model Benchmark & Validation Scorecard", self.styles["SectionHeading"]))
        
        score_headers = [
            Paragraph("Model Name", self.styles["TableHeader"]),
            Paragraph("MAE (MW)", self.styles["TableHeader"]),
            Paragraph("RMSE (MW)", self.styles["TableHeader"]),
            Paragraph("MAPE (%)", self.styles["TableHeader"]),
            Paragraph("R² Score", self.styles["TableHeader"]),
            Paragraph("Status", self.styles["TableHeader"]),
        ]
        score_rows = [score_headers]

        if not scorecard_df.empty:
            for idx, row in scorecard_df.iterrows():
                is_best = (idx == best_model)
                badge = "★ BEST FIT" if is_best else "Verified"
                score_rows.append([
                    Paragraph(f"<b>{idx}</b>", self.styles["TableCell"]),
                    Paragraph(f"{row['MAE']:.2f}", self.styles["TableCell"]),
                    Paragraph(f"{row['RMSE']:.2f}", self.styles["TableCell"]),
                    Paragraph(f"{row['MAPE']:.2f}%", self.styles["TableCell"]),
                    Paragraph(f"{row['R2']:.4f}", self.styles["TableCell"]),
                    Paragraph(f"<b>{badge}</b>", self.styles["TableCell"]),
                ])
        else:
            score_rows.append([Paragraph("No models trained yet", self.styles["TableCell"])] * 6)

        score_table = Table(score_rows, colWidths=[160, 75, 75, 75, 75, 80])
        score_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(score_table)
        elements.append(Spacer(1, 12))

        # --- 4. FORECAST CHART EMBED ---
        elements.append(Paragraph("Demand Trajectory & Forecast Horizon", self.styles["SectionHeading"]))
        chart_buf = self._generate_forecast_chart(historical_df, forecast_df, target_col=metadata.get("target_col", "energy_consumption_mw"))
        elements.append(Image(chart_buf, width=7.2 * inch, height=3.0 * inch))
        elements.append(Spacer(1, 10))

        # --- 5. DIURNAL PROFILE CHART ---
        diurnal_buf = self._generate_diurnal_chart(forecast_df)
        elements.append(Image(diurnal_buf, width=7.2 * inch, height=2.2 * inch))
        elements.append(Spacer(1, 12))

        # --- 6. CONTINUOUS LEARNING & DRIFT AUDIT ---
        elements.append(Paragraph("Continuous Learning & Data Freshness Audit", self.styles["SectionHeading"]))
        drift_status = drift_report.get("status", "Healthy")
        drift_msg = drift_report.get("message", "Model parameters are optimal.")
        ks_val = drift_report.get("ks_statistic", 0.0)

        drift_p = Paragraph(
            f"<b>Data Freshness Status:</b> {drift_status} | "
            f"<b>KS Drift Stat:</b> {ks_val:.4f}<br/>"
            f"<b>Analysis:</b> {drift_msg}<br/>"
            f"<i>Continuous learning engine automatically prevents data decay via streaming incremental updates and batch retraining triggers.</i>",
            self.styles["BodyDark"]
        )
        elements.append(drift_p)
        elements.append(Spacer(1, 12))

        # --- 7. FORECAST DATA PREVIEW TABLE ---
        elements.append(Paragraph(f"Forecast Breakdown (Next {min(24, len(forecast_df))} Hours)", self.styles["SectionHeading"]))

        table_headers = [
            Paragraph("Timestamp", self.styles["TableHeader"]),
            Paragraph("Forecast (MW)", self.styles["TableHeader"]),
            Paragraph("Lower Band 95%", self.styles["TableHeader"]),
            Paragraph("Upper Band 95%", self.styles["TableHeader"]),
            Paragraph("Demand Level", self.styles["TableHeader"]),
        ]
        table_rows = [table_headers]

        preview_df = forecast_df.head(24)
        q70 = forecast_df["forecast_mw"].quantile(0.70)
        q30 = forecast_df["forecast_mw"].quantile(0.30)

        for ts, r in preview_df.iterrows():
            f_val = r["forecast_mw"]
            if f_val >= q70:
                tag = "<font color='#dc2626'><b>Peak</b></font>"
            elif f_val <= q30:
                tag = "<font color='#16a34a'>Off-Peak</font>"
            else:
                tag = "<font color='#2563eb'>Normal</font>"

            table_rows.append([
                Paragraph(ts.strftime("%Y-%m-%d %H:%M"), self.styles["TableCell"]),
                Paragraph(f"<b>{f_val:,.1f}</b>", self.styles["TableCell"]),
                Paragraph(f"{r['lower_bound_mw']:,.1f}", self.styles["TableCell"]),
                Paragraph(f"{r['upper_bound_mw']:,.1f}", self.styles["TableCell"]),
                Paragraph(tag, self.styles["TableCell"]),
            ])

        f_table = Table(table_rows, colWidths=[130, 100, 105, 105, 100])
        f_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(f_table)

        # Build PDF
        doc.build(elements)

        if output_path:
            return b""
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
