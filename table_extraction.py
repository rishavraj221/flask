import pdfplumber
import pprint

with pdfplumber.open("/Users/rishavraj/Downloads/eon_annual_report_2023.pdf") as pdf:
    pages = pdf.pages

    count = 1
    for page in pages:
        print(f"page : {count}")
        tables = page.extract_tables()

        for table in tables:
            pprint.pp(table)

        count += 1
