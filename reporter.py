# reporter.py

from datetime import datetime, timedelta
from collections import defaultdict

def filter_by_timeframe(transactions, mode):
    now = datetime.now()
    if mode == 'daily':
        # Filter for today's transactions only
        today = now.date()
        return [tx for tx in transactions if tx['timestamp'].date() == today]
    elif mode == 'weekly':
        threshold = now - timedelta(weeks=1)
        return [tx for tx in transactions if tx['timestamp'] >= threshold]
    elif mode == 'monthly':
        threshold = now - timedelta(days=30)
        return [tx for tx in transactions if tx['timestamp'] >= threshold]
    elif mode == 'quarterly':
        threshold = now - timedelta(days=90)  # 3 months
        return [tx for tx in transactions if tx['timestamp'] >= threshold]
    else:
        # Default to monthly if unknown mode
        threshold = now - timedelta(days=30)
        return [tx for tx in transactions if tx['timestamp'] >= threshold]

def generate_report(transactions, mode='weekly'):
    recent_txs = filter_by_timeframe(transactions, mode)
    if not recent_txs:
        return f"No transactions found for {mode}."

    total = 0.0
    by_category = defaultdict(float)

    for tx in recent_txs:
        if tx['type'] == 'DEBIT':
            amount = tx['amount']
            total += amount
            by_category[tx['category']] += amount

    lines = [
        f"📊 {mode.capitalize()} Spending Report",
        f"Total Spent: ₹{total:.2f}",
        "-" * 30
    ]

    for category, amt in sorted(by_category.items(), key=lambda x: -x[1]):
        lines.append(f"{category:15}: ₹{amt:.2f}")

    return "\n".join(lines)

def generate_category_breakdown(transactions, mode='weekly', target_category=None):
    """Generate detailed breakdown of transactions by category"""
    recent_txs = filter_by_timeframe(transactions, mode)
    if not recent_txs:
        return f"No transactions found for {mode}."

    # Group transactions by category
    by_category = defaultdict(list)
    total = 0.0

    for tx in recent_txs:
        if tx['type'] == 'DEBIT':
            amount = tx['amount']
            total += amount
            by_category[tx['category']].append(tx)

    if target_category:
        # Show detailed breakdown for specific category
        if target_category not in by_category:
            return f"❌ No '{target_category}' transactions found in {mode} period."
        
        category_txs = by_category[target_category]
        category_total = sum(tx['amount'] for tx in category_txs)
        
        lines = [
            f"📋 {target_category} Transactions - {mode.capitalize()} Period",
            f"Total: ₹{category_total:.2f} ({len(category_txs)} transactions)",
            "=" * 60
        ]
        
        # Sort by amount (highest first)
        sorted_txs = sorted(category_txs, key=lambda x: x['amount'], reverse=True)
        
        for i, tx in enumerate(sorted_txs, 1):
            date_str = tx['timestamp'].strftime('%Y-%m-%d')
            merchant = tx.get('merchant', 'Unknown')
            confidence = tx.get('confidence', 0) * 100 if tx.get('confidence') else 0
            
            # Truncate long descriptions
            desc = tx['description']
            if len(desc) > 50:
                desc = desc[:47] + "..."
            
            lines.append(f"{i:2d}. {date_str} | ₹{tx['amount']:8.2f} | {merchant:20} | {confidence:4.1f}%")
            lines.append(f"    {desc}")
            lines.append("")
        
        return "\n".join(lines)
    
    else:
        # Show summary of all categories with transaction counts
        lines = [
            f"📊 {mode.capitalize()} Spending Breakdown",
            f"Total Spent: ₹{total:.2f}",
            "=" * 50
        ]

        for category, txs in sorted(by_category.items(), key=lambda x: sum(tx['amount'] for tx in x[1]), reverse=True):
            category_total = sum(tx['amount'] for tx in txs)
            percentage = (category_total / total * 100) if total > 0 else 0
            lines.append(f"{category:15}: ₹{category_total:8.2f} ({len(txs):2d} txns) [{percentage:5.1f}%]")

        lines.extend([
            "",
            "💡 Use --category <name> to see detailed breakdown for a specific category",
            "   Example: python main.py breakdown --mode monthly --category Outing"
        ])

        return "\n".join(lines)
