import json
from src.forecasting.chatbot import AIOpsChatbot


def main():
    chatbot = AIOpsChatbot()
    question = "Which servers are likely to exceed 90% CPU next week?"
    ml_output = {"server_id": "App-101", "horizon": 7, "prediction": 94}
    result = chatbot.answer(question, ml_output=ml_output, top_k=5, analysis_mode="capacity_planning")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
