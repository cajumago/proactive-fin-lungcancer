"""
report_generator.py - Generate downloadable reports (CSV, Excel, PDF)
for predicted hospital charge estimates.

All generators return in-memory bytes suitable for Streamlit's
st.download_button, avoiding temporary files on the server.
"""

import io
from datetime import datetime

import pandas as pd
from fpdf import FPDF


def _build_report_dataframe(input_data, estimated_charge, oop_cost=None,
                            oop_method=None, inflation_pct=None):
    """Build a summary DataFrame from prediction inputs and results.

    Parameters
    ----------
    input_data : dict
        The raw input dictionary passed to predict_single.
    estimated_charge : float
        The predicted total charge.
    oop_cost : float, optional
        Calculated out-of-pocket cost.
    oop_method : str, optional
        "Health Insurance" or "Self-Pay".
    inflation_pct : float, optional
        Inflation percentage used (e.g. 7.17).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with all report fields.
    """
    report = {
        "Report Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Age Group": input_data.get("Age Group", ""),
        "Type of Admission": input_data.get("Type of Admission", ""),
        "Patient Disposition": input_data.get("Patient Disposition", ""),
        "APR Severity of Illness": input_data.get(
            "APR Severity of Illness Description", ""
        ),
        "APR Medical/Surgical": input_data.get(
            "APR Medical Surgical Description", ""
        ),
        "APR DRG Description": input_data.get("APR DRG Description", ""),
        "CCSR Procedure": input_data.get("CCSR Procedure Description", ""),
        "Payment Typology": input_data.get("Payment Typology 1", ""),
        "Length of Stay (days)": input_data.get("Length of Stay", ""),
        "Facility Id": input_data.get("Permanent Facility Id", ""),
        "Inflation Adjustment (%)": (
            f"{inflation_pct:.2f}" if inflation_pct is not None else ""
        ),
        "Estimated Total Charge ($)": f"{estimated_charge:,.2f}",
    }

    if oop_cost is not None:
        report["Payment Method"] = oop_method or ""
        report["Out-of-Pocket Cost ($)"] = f"{oop_cost:,.2f}"

    return pd.DataFrame([report])


def generate_csv(input_data, estimated_charge, oop_cost=None,
                 oop_method=None, inflation_pct=None):
    """Generate a CSV report as bytes.

    Returns
    -------
    bytes
        UTF-8 encoded CSV content.
    """
    df = _build_report_dataframe(
        input_data, estimated_charge, oop_cost, oop_method, inflation_pct
    )
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False, encoding="utf-8")
    return buffer.getvalue()


def generate_excel(input_data, estimated_charge, oop_cost=None,
                   oop_method=None, inflation_pct=None):
    """Generate an Excel (.xlsx) report as bytes.

    Returns
    -------
    bytes
        Excel file content.
    """
    df = _build_report_dataframe(
        input_data, estimated_charge, oop_cost, oop_method, inflation_pct
    )
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Charge Estimate")
    return buffer.getvalue()


def generate_pdf(input_data, estimated_charge, oop_cost=None,
                 oop_method=None, inflation_pct=None):
    """Generate a professional PDF report as bytes.

    Layout:
    - Title header
    - Report timestamp
    - Patient/discharge information table (alternating row colors)
    - Highlighted prediction result
    - OOP section (if calculated)
    - Footer with model metadata and disclaimer

    Returns
    -------
    bytes
        PDF file content.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Title ---
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(
        0, 12,
        "NY Respiratory Cancer - Charge Estimate Report",
        new_x="LMARGIN", new_y="NEXT", align="C",
    )
    pdf.ln(4)

    # --- Date ---
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0, 6,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        new_x="LMARGIN", new_y="NEXT", align="R",
    )
    pdf.ln(6)

    # --- Patient Information Table ---
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(
        0, 8, "Patient and Discharge Information",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(2)

    fields = [
        ("Age Group", input_data.get("Age Group", "")),
        ("Type of Admission", input_data.get("Type of Admission", "")),
        ("Patient Disposition", input_data.get("Patient Disposition", "")),
        ("APR Severity of Illness",
         input_data.get("APR Severity of Illness Description", "")),
        ("APR Medical/Surgical",
         input_data.get("APR Medical Surgical Description", "")),
        ("APR DRG Description",
         input_data.get("APR DRG Description", "")),
        ("CCSR Procedure",
         input_data.get("CCSR Procedure Description", "")),
        ("Payment Typology", input_data.get("Payment Typology 1", "")),
        ("Length of Stay (days)",
         str(input_data.get("Length of Stay", ""))),
        ("Facility Id",
         str(input_data.get("Permanent Facility Id", ""))),
    ]

    col_w_label = 70
    col_w_value = 115

    for i, (label, value) in enumerate(fields):
        if i % 2 == 0:
            pdf.set_fill_color(240, 240, 240)
        else:
            pdf.set_fill_color(255, 255, 255)

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(col_w_label, 7, label, border=0, fill=True)
        pdf.set_font("Helvetica", "", 10)
        display_value = str(value)[:65]
        pdf.cell(
            col_w_value, 7, display_value,
            border=0, new_x="LMARGIN", new_y="NEXT", fill=True,
        )

    pdf.ln(6)

    # --- Prediction Result (highlighted) ---
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(
        0, 8, "Prediction Result",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(2)

    if inflation_pct is not None:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(
            0, 6, f"Inflation Adjustment: {inflation_pct:.2f}%",
            new_x="LMARGIN", new_y="NEXT",
        )

    pdf.set_fill_color(34, 139, 34)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(
        0, 10,
        f"  Estimated Total Charge: ${estimated_charge:,.2f}",
        new_x="LMARGIN", new_y="NEXT", fill=True,
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # --- OOP Section (conditional) ---
    if oop_cost is not None:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(
            0, 8, "Out-of-Pocket Cost Estimate",
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.ln(2)

        pdf.set_font("Helvetica", "", 10)
        pdf.cell(
            0, 6, f"Payment Method: {oop_method or 'N/A'}",
            new_x="LMARGIN", new_y="NEXT",
        )

        pdf.set_fill_color(30, 100, 180)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(
            0, 10,
            f"  Estimated Out-of-Pocket: ${oop_cost:,.2f}",
            new_x="LMARGIN", new_y="NEXT", fill=True,
        )
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

    # --- Footer ---
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(
        0, 5,
        "Model: XGBRegressor (1000 trees, depth 4, lr 0.1) | "
        "Target: log(Total Charges + 3000) | "
        "Data: SPARCS 2018-2021",
        new_x="LMARGIN", new_y="NEXT", align="C",
    )
    pdf.cell(
        0, 5,
        "Disclaimer: This estimate is for informational purposes only "
        "and should not be used for billing decisions.",
        new_x="LMARGIN", new_y="NEXT", align="C",
    )

    buffer = io.BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()
