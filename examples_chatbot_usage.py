"""
Example script demonstrating AIOps Chatbot endpoints.

Usage:
    python examples_chatbot_usage.py

This script shows how to call the chatbot API for various operational scenarios.
"""

import json
import urllib.request
import urllib.error


BASE_URL = "http://127.0.0.1:8001"


def call_endpoint(path: str, payload: dict) -> dict:
    """Call a chatbot endpoint and return the JSON response."""
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "detail": e.read().decode("utf-8")}
    except Exception as e:
        return {"error": str(e)}


def example_1_capacity_planning():
    """Example 1: Capacity Planning - Which servers need scaling?"""
    print("\n" + "="*80)
    print("Example 1: Capacity Planning Summary")
    print("="*80)
    
    payload = {
        "question": "Which servers are forecast to exceed 90% CPU next week and what should we do?",
        "analysis_mode": "capacity_planning",
        "top_k": 5,
        "ml_output": {
            "server_id": "App-101",
            "horizon": 7,
            "prediction": 94
        }
    }
    
    print("\nRequest:")
    print(json.dumps(payload, indent=2))
    
    response = call_endpoint("/llm/capacity-summary", payload)
    
    print("\nResponse Summary:")
    if "error" not in response:
        print(f"  Risk Level: {response.get('risk', {}).get('level', 'unknown')}")
        print(f"  Recommendation: {response.get('recommendation', 'N/A')}")
        print(f"  Action Plan:")
        for action in response.get('action_plan', [])[:3]:
            print(f"    - {action}")
        print(f"  LLM Answer (first 200 chars): {response.get('answer', 'N/A')[:200]}...")
    else:
        print(f"  Error: {response.get('detail', response.get('error'))}")


def example_2_draft_jira_ticket():
    """Example 2: Draft Jira Ticket - Create an issue for high CPU."""
    print("\n" + "="*80)
    print("Example 2: Draft Jira Ticket")
    print("="*80)
    
    payload = {
        "top_k": 5,
        "ml_output": {
            "server_id": "DB-205",
            "horizon": 7,
            "prediction": 92
        }
    }
    
    print("\nRequest:")
    print(json.dumps(payload, indent=2))
    
    response = call_endpoint("/llm/draft-ticket", payload)
    
    print("\nResponse - Jira Ticket Draft:")
    if "error" not in response and "ticket_draft" in response:
        ticket = response["ticket_draft"]
        print(f"  Summary: {ticket.get('summary', 'N/A')}")
        print(f"  Description:\n    {ticket.get('description', 'N/A').replace(chr(10), chr(10) + '    ')}")
        print(f"  Labels: {', '.join(ticket.get('labels', []))}")
        print(f"  Risk Level: {response.get('risk_level', 'unknown')}")
    else:
        print(f"  Error: {response.get('detail', response.get('error'))}")


def example_3_executive_report():
    """Example 3: Executive Report - High-level summary for leadership."""
    print("\n" + "="*80)
    print("Example 3: Executive Report")
    print("="*80)
    
    payload = {
        "question": "Summarize infrastructure capacity risks for the next 30 days.",
        "top_k": 5,
        "ml_output": {
            "server_id": "App-101",
            "horizon": 30,
            "prediction": 88
        }
    }
    
    print("\nRequest:")
    print(json.dumps(payload, indent=2))
    
    response = call_endpoint("/llm/executive-report", payload)
    
    print("\nResponse - Executive Summary:")
    if "error" not in response:
        print(f"  Risk Level: {response.get('risk', {}).get('level', 'unknown')}")
        print(f"  Executive Summary: {response.get('executive_summary', 'N/A')}")
        print(f"  LLM Analysis (first 300 chars): {response.get('answer', 'N/A')[:300]}...")
    else:
        print(f"  Error: {response.get('detail', response.get('error'))}")


def example_4_root_cause():
    """Example 4: Root Cause Analysis - Investigate why CPU is spiking."""
    print("\n" + "="*80)
    print("Example 4: Root Cause Analysis")
    print("="*80)
    
    payload = {
        "question": "What is the likely root cause of this CPU spike? Has this happened before?",
        "top_k": 5,
        "ml_output": {
            "server_id": "API-312",
            "horizon": 1,
            "prediction": 97
        }
    }
    
    print("\nRequest:")
    print(json.dumps(payload, indent=2))
    
    response = call_endpoint("/llm/root-cause", payload)
    
    print("\nResponse - Root Cause Analysis:")
    if "error" not in response:
        print(f"  Risk Level: {response.get('risk', {}).get('level', 'unknown')}")
        print(f"  Recommendation: {response.get('recommendation', 'N/A')}")
        print(f"  Related Incidents Found: {len(response.get('related_incidents', []))}")
        print(f"  LLM Analysis (first 400 chars): {response.get('answer', 'N/A')[:400]}...")
    else:
        print(f"  Error: {response.get('detail', response.get('error'))}")


def example_5_general_chat():
    """Example 5: General Q&A - Ask custom questions."""
    print("\n" + "="*80)
    print("Example 5: General Q&A (Custom Question)")
    print("="*80)
    
    payload = {
        "question": "Which payment platform servers are at highest risk next week?",
        "analysis_mode": "capacity_planning",
        "top_k": 5,
        "ml_output": {
            "server_id": "payment-api-01",
            "horizon": 7,
            "prediction": 89
        }
    }
    
    print("\nRequest:")
    print(json.dumps(payload, indent=2))
    
    response = call_endpoint("/chat", payload)
    
    print("\nResponse Summary:")
    if "error" not in response:
        print(f"  Risk Level: {response.get('risk', {}).get('level', 'unknown')}")
        print(f"  Retrieved Sources: {len(response.get('sources', []))}")
        print(f"  LLM Answer (first 300 chars): {response.get('answer', 'N/A')[:300]}...")
    else:
        print(f"  Error: {response.get('detail', response.get('error'))}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("AIOps Chatbot Examples")
    print("="*80)
    print(f"\nCalling chatbot at {BASE_URL}")
    print("\nNote: Ensure the chatbot server is running:")
    print("  python start_chatbot.py")
    
    try:
        # Test health endpoint first
        health = call_endpoint("/health", {})
        if "error" not in health:
            print("\n✓ Chatbot is healthy!")
        else:
            print(f"\n✗ Chatbot health check failed: {health.get('error')}")
            return
    except Exception as e:
        print(f"\n✗ Cannot connect to chatbot: {e}")
        print("   Make sure the server is running: python start_chatbot.py")
        return
    
    # Run all examples
    example_1_capacity_planning()
    example_2_draft_jira_ticket()
    example_3_executive_report()
    example_4_root_cause()
    example_5_general_chat()
    
    print("\n" + "="*80)
    print("Examples Complete!")
    print("="*80)
    print("\nFor more endpoints and documentation, see CHATBOT_README.md")
    print("For API specification, see src/forecasting/api.py")


if __name__ == "__main__":
    main()
