
# Pattern: Manual planner → executor (CoT-driven)
#
# LLM first creates a step-by-step plan (free-text CoT).
#
# Script then iterates over the steps and prompts the model to execute each one.
#
# This squarely demonstrates “Planning & Task Decomposition” without a framework.
# Shows the idea of LLM-generated plan → sequential execution.

import os
import json
from datetime import datetime, timedelta
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Step 1: Load API Key from .env
# ----------------------------
# .env file should contain:
# OPENAI_API_KEY=your_api_key_here
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file.")

client = OpenAI(api_key=api_key)

# ----------------------------
# Step 2: Read User History from CSV
# ----------------------------
# Example file: user_history.csv
# timestamp,amount,merchant,merchant_category,location
# 2025-04-18T10:30:00,42.15,Starbucks,Food,New York
# 2025-04-17T18:20:00,125.30,Whole Foods,Grocery,New York
# 2025-04-15T12:10:00,85.00,Amazon,Retail,Online
# 2025-04-12T09:15:00,35.50,Starbucks,Food,New York
# 2025-04-10T20:20:00,200.00,Nike,Retail,New York

df = pd.read_csv("../data/user_history.csv")

user_history = df.to_dict(orient="records")

# ----------------------------
# Step 3: Suspicious Transaction (hardcoded here but in real world it will come when a user os swiping their card)
# ----------------------------
suspicious_transaction = {
    "timestamp": "2025-04-19T03:45:00",
    "amount": 9999.99,
    "merchant": "Electronics Store",
    "merchant_category": "Electronics",
    "location": "New Delhi"
}

# ----------------------------
# Step 4: Extract Features
# ----------------------------
amounts = [tx["amount"] for tx in user_history]
avg_amount = sum(amounts) / len(amounts) if amounts else 0

locations = [tx["location"] for tx in user_history]
common_locations = set([loc for loc in locations if locations.count(loc) > 1])

recent_count = 0
if user_history:
    current_time = datetime.fromisoformat(suspicious_transaction["timestamp"])
    for tx in user_history:
        tx_time = datetime.fromisoformat(tx["timestamp"])
        if current_time - tx_time <= timedelta(hours=24):
            recent_count += 1

features = {
    "avg_transaction_amount": avg_amount,
    "transaction_velocity_24h": recent_count,
    "common_locations": list(common_locations),
    "usual_merchant_categories": list(set([tx["merchant_category"] for tx in user_history])),
    "transaction_count_30d": len(user_history),
    "highest_single_amount": max(amounts) if amounts else 0,
}

# ----------------------------
# Step 5: Generate a Plan (CoT)
# ----------------------------
plan_prompt = f"""You are a financial fraud analyst.
Your task is to analyze this transaction step-by-step.

CURRENT TRANSACTION:
{json.dumps(suspicious_transaction, indent=2)}

USER HISTORY SUMMARY:
{json.dumps(features, indent=2)}

Write a clear step-by-step plan to assess fraud risk (numbered steps).
"""

response = client.responses.create(
    model="gpt-4o-mini",
    input=plan_prompt,
    temperature=0
)

plan = response.output_text
print("\n--- CoT Plan ---")
print(plan)

# ----------------------------
# Step 6: Execute Each Step
# ----------------------------
steps = plan.strip().split("\n")
reasoning_log = ""

print("\n--- Step-by-Step Analysis ---")
for step in steps:
    if not step.strip():
        continue
    reasoning_log += f"\n {step}\n"

    step_reasoning_prompt = f"""Given the following transaction and user data, perform this step:
    Step: {step}

    TRANSACTION:
    {json.dumps(suspicious_transaction)}

    USER FEATURES:
    {json.dumps(features)}
    """

    response = client.responses.create(
        model="gpt-4o-mini",
        input=step_reasoning_prompt,
        temperature=0
    )
    step_reasoning = response.output_text
    reasoning_log += f" Thought: {step_reasoning.strip()}\n"

    print(f"\nStep: {step}")
    print("Thought:", step_reasoning.strip())

# ----------------------------
# Step 7: Generate Final Fraud Risk Report
# ----------------------------
final_prompt = f"""Based on the analysis steps above, summarize the fraud risk as a JSON object with this format:

  "fraud_risk_score": <0-100>,
  "risk_level": "<Low|Medium|High>",
  "explanation": "..."

Analysis Steps: {reasoning_log}
"""

response = client.responses.create(
    model="gpt-4o-mini",
    input=final_prompt,
    temperature=0
)

fraud_report = response.output_text

print("\n--- Fraud Risk Report ---")
print(fraud_report)