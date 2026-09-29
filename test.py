from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

client = ChatNVIDIA()
print("Available active chat models:")
for m in client.available_models:
  if getattr(m, "model_type", None) == "chat":
    print(f" - {m.id}")