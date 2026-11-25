from ml.data.loader import (
    get_excel_sheets,
)


sheets = get_excel_sheets()

print("Available sheets:")

for sheet in sheets:
    print(f"- {sheet}")