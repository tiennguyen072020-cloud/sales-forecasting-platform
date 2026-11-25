from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...], *, required: bool = True) -> str | None:
    lookup = {column.casefold(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    if required:
        raise ValueError(f"Missing required column. Expected one of: {', '.join(candidates)}")
    return None

class AnalyticsService:
    """Read processed sales data and prepare dashboard-ready aggregates."""
    def __init__(self, project_root: str | Path):
        self.root = Path(project_root).resolve()
        processed = self.root / "data" / "processed"
        self.daily_path = processed / "daily_revenue.csv"
        self.sales_path = processed / "valid_sales.csv"

        if not self.daily_path.exists():
            raise FileNotFoundError(f"Daily revenue file not found: {self.daily_path}")
        if not self.sales_path.exists():
            raise FileNotFoundError(f"Valid sales file not found: {self.sales_path}")

        self.daily = pd.read_csv(self.daily_path)
        self.sales = pd.read_csv(
            self.sales_path,
            dtype={"invoice_no": "string"},
        )
        self._prepare_columns()

    def _prepare_columns(self) -> None:
        self.daily_date = _find_column(self.daily, ("date", "invoice_date", "invoicedate"))
        self.daily_revenue = _find_column(
            self.daily,
            ("net_revenue", "revenue", "daily_revenue", "sales"),
        )
        self.daily[self.daily_date] = pd.to_datetime(self.daily[self.daily_date], errors="raise")
        self.daily[self.daily_revenue] = pd.to_numeric(self.daily[self.daily_revenue], errors="coerce").fillna(0.0)
        self.daily = self.daily.sort_values(self.daily_date).reset_index(drop=True)

        self.product_code = _find_column(self.sales, ("stock_code", "stockcode", "product_code", "sku"))
        self.product_name = _find_column(self.sales, ("description", "product_name", "product"))
        self.quantity = _find_column(self.sales, ("quantity", "qty"))
        self.sales_revenue = _find_column(
            self.sales,
            ("net_revenue", "revenue", "line_revenue", "total_price", "sales"),
        )
        self.category = _find_column(
            self.sales,
            ("product_category", "category", "category_name"),
            required=False,
        )
        self.sales[self.quantity] = pd.to_numeric(self.sales[self.quantity], errors="coerce").fillna(0.0)
        self.sales[self.sales_revenue] = pd.to_numeric(self.sales[self.sales_revenue], errors="coerce").fillna(0.0)

    def revenue(self) -> dict:
        grouped = (
            self.daily.groupby(self.daily_date, as_index=False)[self.daily_revenue]
            .sum()
            .sort_values(self.daily_date)
        )
        return {
            "total_revenue": float(grouped[self.daily_revenue].sum()),
            "average_daily_revenue": float(grouped[self.daily_revenue].mean()),
            "first_date": grouped[self.daily_date].min().date(),
            "last_date": grouped[self.daily_date].max().date(),
            "values": [
                {"date": row[self.daily_date].date(), "revenue": float(row[self.daily_revenue])}
                for _, row in grouped.iterrows()
            ],
        }

    def growth(self, frequency: str = "month") -> dict:
        rule = {"month": "ME", "quarter": "QE", "year": "YE"}[frequency]
        indexed = self.daily.set_index(self.daily_date)[self.daily_revenue]
        totals = indexed.resample(rule).sum()
        growth = totals.pct_change().replace([np.inf, -np.inf], np.nan) * 100
        values = []
        for period, revenue in totals.items():
            growth_value = growth.loc[period]
            values.append({
                "period": period.strftime("%Y-%m-%d"),
                "revenue": float(revenue),
                "growth_percent": None if pd.isna(growth_value) else float(growth_value),
            })
        return {"frequency": frequency, "values": values}

    def top_products(self, limit: int = 10) -> dict:
        grouped = (
            self.sales.groupby([self.product_code, self.product_name], dropna=False, as_index=False)
            .agg({self.quantity: "sum", self.sales_revenue: "sum"})
            .sort_values(self.sales_revenue, ascending=False)
            .head(limit)
            .reset_index(drop=True)
        )
        values = []
        for index, row in grouped.iterrows():
            values.append({
                "rank": index + 1,
                "product_code": str(row[self.product_code]),
                "product_name": str(row[self.product_name]),
                "quantity": float(row[self.quantity]),
                "revenue": float(row[self.sales_revenue]),
            })
        return {"limit": limit, "values": values}

    def categories(self) -> dict:
        category_source = self.category or "Uncategorized"
        frame = self.sales.copy()
        if self.category is None:
            frame[category_source] = "Uncategorized"

        grouped = (
            frame.groupby(category_source, dropna=False, as_index=False)
            .agg({self.quantity: "sum", self.sales_revenue: "sum"})
            .sort_values(self.sales_revenue, ascending=False)
        )
        total = float(grouped[self.sales_revenue].sum())
        values = []
        for _, row in grouped.iterrows():
            revenue = float(row[self.sales_revenue])
            values.append({
                "category": str(row[category_source]),
                "quantity": float(row[self.quantity]),
                "revenue": revenue,
                "revenue_share_percent": 0.0 if total == 0 else revenue / total * 100,
            })
        return {
            "category_source": category_source,
            "warning": None if self.category else "No category column found; add product_category to valid_sales.csv.",
            "values": values,
        }