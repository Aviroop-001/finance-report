import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import google.generativeai as genai
from categorizer import categorize_transactions
from dotenv import load_dotenv

load_dotenv()

class FinancialProfile:
    def __init__(
        self,
        monthly_income: float,
        monthly_rent: float,
        weekly_travel: float,
        weekly_groceries: float,
        city: str,
        lifestyle: str,  # 'minimal', 'moderate', 'luxurious'
        relationship_status: str,  # 'single', 'in_relationship'
        additional_info: Optional[Dict] = None
    ):
        self.monthly_income = monthly_income
        self.monthly_rent = monthly_rent
        self.weekly_travel = weekly_travel
        self.weekly_groceries = weekly_groceries
        self.city = city
        self.lifestyle = lifestyle
        self.relationship_status = relationship_status
        self.additional_info = additional_info or {}

class InsightsGenerator:
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        # Create model instance
        generation_config = {
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",  # Updated to latest Gemini model
            generation_config=generation_config,
            safety_settings=safety_settings
        )

    def _call_gemini(self, prompt: str) -> str:
        """Generate response using Gemini"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return "Error: Could not generate insights. Please check your API key and try again."

    def generate_insights(
        self,
        profile: FinancialProfile,
        transactions: List[Dict],
        start_date: str,
        end_date: str
    ) -> Dict:
        """Generate financial insights using transaction data and user profile"""
        
        # First categorize the transactions
        try:
            categorized_transactions = categorize_transactions(transactions)
        except Exception as e:
            print(f"Error categorizing transactions: {e}")
            return {
                "error": "Could not categorize transactions",
                "details": str(e)
            }
        
        # Calculate spending metrics
        total_spent = sum(float(tx['amount']) for tx in categorized_transactions)
        spending_by_category = {}
        for tx in categorized_transactions:
            category = tx['category']
            if category not in spending_by_category:
                spending_by_category[category] = 0
            spending_by_category[category] += float(tx['amount'])

        # Calculate monthly averages
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        months = (end.year - start.year) * 12 + end.month - start.month + 1
        monthly_spend = total_spent / months if months > 0 else total_spent

        # Prepare data for AI analysis
        analysis_data = {
            "profile": {
                "monthly_income": profile.monthly_income,
                "monthly_rent": profile.monthly_rent,
                "weekly_travel": profile.weekly_travel,
                "weekly_groceries": profile.weekly_groceries,
                "city": profile.city,
                "lifestyle": profile.lifestyle,
                "relationship_status": profile.relationship_status
            },
            "spending": {
                "total_spent": total_spent,
                "monthly_average": monthly_spend,
                "by_category": spending_by_category
            },
            "period": {
                "start_date": start_date,
                "end_date": end_date,
                "months_analyzed": months
            }
        }

        # Generate prompts for different aspects
        prompts = {
            "overview": self._generate_overview_prompt(analysis_data),
            "improvements": self._generate_improvements_prompt(analysis_data),
            "lifestyle": self._generate_lifestyle_prompt(analysis_data),
            "recommendations": self._generate_recommendations_prompt(analysis_data)
        }

        # Get insights from Gemini
        insights = {}
        for aspect, prompt in prompts.items():
            insights[aspect] = self._call_gemini(prompt)

        return {
            "data": analysis_data,
            "insights": insights
        }

    def _generate_overview_prompt(self, data: Dict) -> str:
        return f"""
        You are a financial advisor. Give a BRIEF 2-3 bullet point overview of this spending data. NO explanations, just facts:

        Monthly Income: ₹{data['profile']['monthly_income']:,.2f}
        Monthly Rent: ₹{data['profile']['monthly_rent']:,.2f} ({(data['profile']['monthly_rent']/data['profile']['monthly_income']*100):.1f}% of income)
        Weekly Travel: ₹{data['profile']['weekly_travel']:,.2f}
        Weekly Groceries: ₹{data['profile']['weekly_groceries']:,.2f}
        City: {data['profile']['city']}
        Lifestyle: {data['profile']['lifestyle']}

        Monthly Average Spend: ₹{data['spending']['monthly_average']:,.2f}

        Categories:
        {self._format_category_spending(data['spending']['by_category'])}

        Format response as:
        • Key fact 1
        • Key fact 2
        • Key fact 3 (if needed)
        """

    def _generate_improvements_prompt(self, data: Dict) -> str:
        return f"""
        You are a financial advisor. List ONLY the top 3 categories that need immediate attention, with specific target reduction amounts. NO explanations:

        Monthly Income: ₹{data['profile']['monthly_income']:,.2f}
        Categories:
        {self._format_category_spending(data['spending']['by_category'])}

        Format response EXACTLY as:
        • Category 1: Reduce by ₹X/month
        • Category 2: Reduce by ₹X/month
        • Category 3: Reduce by ₹X/month
        """

    def _generate_lifestyle_prompt(self, data: Dict) -> str:
        return f"""
        You are a financial advisor. Give ONLY 2-3 specific lifestyle adjustments needed based on this data. NO explanations:

        Profile:
        - Lives in: {data['profile']['city']}
        - Lifestyle: {data['profile']['lifestyle']}
        - Monthly Income: ₹{data['profile']['monthly_income']:,.2f}
        - Monthly Spend: ₹{data['spending']['monthly_average']:,.2f}

        Categories:
        {self._format_category_spending(data['spending']['by_category'])}

        Format response EXACTLY as:
        • Action 1: Specific change
        • Action 2: Specific change
        • Action 3: Specific change (if needed)
        """

    def _generate_recommendations_prompt(self, data: Dict) -> str:
        return f"""
        You are a financial advisor. Give ONLY specific, actionable steps. NO explanations or context:

        Current Situation:
        - Monthly Income: ₹{data['profile']['monthly_income']:,.2f}
        - Monthly Spend: ₹{data['spending']['monthly_average']:,.2f}
        - Location: {data['profile']['city']}
        - Major Categories:
        {self._format_category_spending(data['spending']['by_category'])}

        Format response EXACTLY as:
        IMMEDIATE ACTIONS (next 30 days):
        • Action 1: Specific step
        • Action 2: Specific step
        • Action 3: Specific step

        SAVINGS TARGETS:
        • Target 1: Amount/timeframe
        • Target 2: Amount/timeframe
        • Target 3: Amount/timeframe
        """

    def _format_category_spending(self, category_data: Dict) -> str:
        total = sum(category_data.values())
        formatted = []
        for category, amount in sorted(category_data.items(), key=lambda x: x[1], reverse=True):
            percentage = (amount / total * 100) if total > 0 else 0
            formatted.append(f"- {category}: ₹{amount:,.2f} ({percentage:.1f}%)")
        return "\n".join(formatted) 