#!/usr/bin/env python3
"""
Automated report generator for weekly and monthly financial insights
"""

import os
from datetime import datetime, timedelta
from main import create_ppt
from dotenv import load_dotenv

load_dotenv()

def generate_weekly_report():
    """Generate and email report for the past week"""
    # Calculate date range (Monday to Sunday)
    today = datetime.now()
    end_date = today - timedelta(days=today.weekday())  # Last Sunday
    start_date = end_date - timedelta(days=7)  # Previous Monday
    
    # Format dates
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Generate report
    output_file = f"reports/weekly_report_{end_str}.pptx"
    os.makedirs("reports", exist_ok=True)
    
    print(f"Generating weekly report for {start_str} to {end_str}")
    create_ppt(
        start_date=start_str,
        end_date=end_str,
        output=output_file,
        email=os.getenv('REPORT_EMAIL'),
        include_insights=True
    )

def generate_monthly_report():
    """Generate and email report for the past month"""
    # Calculate date range (1st to last day of previous month)
    today = datetime.now()
    end_date = today.replace(day=1) - timedelta(days=1)  # Last day of previous month
    start_date = end_date.replace(day=1)  # First day of previous month
    
    # Format dates
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Generate report
    output_file = f"reports/monthly_report_{end_str}.pptx"
    os.makedirs("reports", exist_ok=True)
    
    print(f"Generating monthly report for {start_str} to {end_str}")
    create_ppt(
        start_date=start_str,
        end_date=end_str,
        output=output_file,
        email=os.getenv('REPORT_EMAIL'),
        include_insights=True
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2 or sys.argv[1] not in ['weekly', 'monthly']:
        print("Usage: python automated_reports.py [weekly|monthly]")
        sys.exit(1)
    
    if sys.argv[1] == 'weekly':
        generate_weekly_report()
    else:
        generate_monthly_report() 