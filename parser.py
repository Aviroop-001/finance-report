# parser.py
import csv
from datetime import datetime

def parse_transactions(csv_path):
    transactions = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                # Updated to handle timestamp format without timezone
                timestamp_str = row['Timestamp'].strip()
                try:
                    # Try parsing with timezone first (legacy format)
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S %Z')
                except ValueError:
                    # Fall back to format without timezone (our HTML parser format)
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                
                transaction = {
                    'timestamp': timestamp,
                    'description': row.get('Description', '').strip(),
                    'merchant': row.get('Merchant', '').strip(),
                    'amount': float(str(row['Amount']).replace('₹', '').replace(',', '').strip()),
                    'type': row.get('Transaction Type', 'DEBIT').strip().upper()
                }
                transactions.append(transaction)
            except Exception as e:
                print(f"Skipping row due to error: {e}\nRow: {row}")
    return transactions
