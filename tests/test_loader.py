import pandas as pd

from ml.data.loader import (
    get_excel_sheets,
    load_retail_data,
)


def test_excel_has_sheets():

    sheets = get_excel_sheets()

    assert len(sheets) > 0


def test_load_retail_data():

    df = load_retail_data()

    assert isinstance(
        df,
        pd.DataFrame,
    )

    assert not df.empty