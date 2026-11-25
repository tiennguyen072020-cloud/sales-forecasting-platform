import pandas as pd

import ml.data.loader as data_loader
import ml.data.preprocessing as preprocessing
from ml.data.preprocessing import (
    CleaningResult,
    calculate_line_revenue,
    normalize_invoice_no,
    save_processed_data,
    standardize_columns,
)

# Test columns
def test_standardize_columns():

    df = pd.DataFrame({
        "Invoice": ["123"],
        "StockCode": ["A01"],
        "Description": ["Product"],
        "Quantity": [2],
        "InvoiceDate": ["2026-01-01"],
        "Price": [10],
        "Customer ID": [1],
        "Country": ["UK"],
    })

    result = standardize_columns(
        df
    )

    assert (
        "invoice_no"
        in result.columns
    )

    assert (
        "unit_price"
        in result.columns
    )

    # Test normalization
def test_normalize_invoice():

    df = pd.DataFrame({
        "invoice_no": [
            " c123 ",
        ]
    })

    result = normalize_invoice_no(
        df
    )

    assert (
        result.loc[
            0,
            "invoice_no",
        ]
        == "C123"
    )
    
# Test revenue
def test_line_revenue():

    df = pd.DataFrame({
        "quantity": [3],
        "unit_price": [10.0],
    })

    result = calculate_line_revenue(
        df
    )

    assert (
        result.loc[
            0,
            "line_revenue",
        ]
        == 30.0
    )


def test_save_processed_data_writes_invoice_numbers_as_strings(
    tmp_path,
    monkeypatch,
):
    output_files = {
        "CLEAN_RETAIL_FILE": tmp_path / "clean.csv",
        "VALID_SALES_FILE": tmp_path / "valid_sales.csv",
        "RETURNS_FILE": tmp_path / "returns.csv",
        "INVALID_TRANSACTIONS_FILE": tmp_path / "invalid.csv",
        "DAILY_REVENUE_FILE": tmp_path / "daily_revenue.csv",
    }
    for name, path in output_files.items():
        monkeypatch.setattr(preprocessing, name, path)

    valid_sales = pd.DataFrame({
        "invoice_no": ["489434", "A563185"],
        "invoice_date": ["2026-01-01", "2026-01-02"],
    })
    result = CleaningResult(
        clean_data=pd.DataFrame(),
        valid_sales=valid_sales,
        returns=pd.DataFrame(),
        invalid_transactions=pd.DataFrame(),
        daily_revenue=pd.DataFrame(),
    )

    save_processed_data(result)

    saved_csv = output_files["VALID_SALES_FILE"].read_text(encoding="utf-8")
    assert "489434" in saved_csv
    assert "A563185" in saved_csv

    monkeypatch.setattr(
        data_loader,
        "VALID_SALES_FILE",
        output_files["VALID_SALES_FILE"],
    )
    loaded = data_loader.load_valid_sales()
    assert str(loaded["invoice_no"].dtype) == "string"
    assert loaded["invoice_no"].tolist() == ["489434", "A563185"]
    assert pd.api.types.is_datetime64_any_dtype(loaded["invoice_date"])