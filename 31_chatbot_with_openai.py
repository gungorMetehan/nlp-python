# Get API Keys: https://platform.openai.com/api-keys

from openai import OpenAI  # OpenAI client import

BOT_NAME = "MyChatBot"  # chatbot display name

client = OpenAI()  # initialize OpenAI client

# send messages to model
def chat_with_gpt(messages):
    response = client.responses.create(
        model = "gpt-4.1-mini",
        input = messages
    )
    return response.output_text


if __name__ == "__main__":
    messages = [
        {
            "role": "system",
            "content": f"You are a helpful and concise assistant named {BOT_NAME}."
        }
    ]

    while True:
        user_input = input("Kullanıcı: ")

        if user_input.lower() in ["exit", "q"]:
            print("Konuşma sonlandırıldı.")
            break

        # store user message
        messages.append({
            "role": "user",
            "content": user_input
        })

        try:
            assistant_reply = chat_with_gpt(messages)
        except Exception as e:
            print("Hata oluştu:", e)
            break

        print(f"{BOT_NAME}:", assistant_reply)

        # store assistant reply
        messages.append({
            "role": "assistant",
            "content": assistant_reply
        })