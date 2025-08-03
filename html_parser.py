# html_parser.py
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import re

def extract_transactions_from_html(html_path, output_csv_path):
    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    records = []
    # Updated class selector to match actual HTML structure
    activity_items = soup.find_all("div", class_="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1")

    for item in activity_items:
        text = item.get_text(separator="\n").strip()
        
        # Skip empty items
        if not text:
            continue
            
        lines = text.split("\n")

        amount = None
        merchant = ""
        description = ""
        timestamp = ""
        tx_type = "DEBIT"

        for line in lines:
            if "₹" in line:
                match = re.search(r"₹([\d,.]+)", line)
                if match:
                    amount = float(match.group(1).replace(",", ""))
                    description = line.strip()
                    
                    # Determine transaction type and merchant from description
                    if line.startswith("Paid"):
                        tx_type = "DEBIT"
                        # Extract merchant name
                        if " to " in line:
                            merchant_part = line.split(" to ")[1]
                            if " using " in merchant_part:
                                merchant = merchant_part.split(" using ")[0].strip()
                            else:
                                merchant = merchant_part.strip()
                    elif line.startswith("Sent"):
                        tx_type = "DEBIT"
                        merchant = "Transfer"
                    elif line.startswith("Received"):
                        tx_type = "CREDIT"
                        merchant = "Received"
                        
            # Updated date parsing for GMT+05:30 format
            elif re.search(r"\d+ \w+ \d{4}, \d{2}:\d{2}:\d{2} GMT\+05:30", line):
                try:
                    # Parse format like "2 Aug 2025, 17:05:50 GMT+05:30"
                    date_str = line.replace("GMT+05:30", "").strip()
                    timestamp = datetime.strptime(date_str, "%d %b %Y, %H:%M:%S")
                except Exception as e:
                    print(f"Date parsing error: {e} for line: {line}")
                    continue

        if amount and timestamp:
            records.append({
                "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Merchant": merchant,
                "Description": description,
                "Amount": amount,
                "Transaction Type": tx_type
            })

    # Write to CSV
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Timestamp", "Merchant", "Description", "Amount", "Transaction Type"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"✅ Extracted {len(records)} transactions into {output_csv_path}")
    return records

if __name__ == "__main__":
    extract_transactions_from_html("data/MyActivity.html", "data/transactions.csv")
