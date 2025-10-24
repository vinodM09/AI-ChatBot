import requests

CONVERSATION_ID = "terminal_user"
URL = "http://127.0.0.1:8000/chat/"

while True:
    msg = input("You: ")
    if msg.lower() in ["exit", "quit"]:
        break

    response = requests.post(URL, json={
        "conversation_id": CONVERSATION_ID,
        "message": msg
    })

    if response.status_code == 200:
        print("Bot:", response.json()["reply"])
    else:
        print("Error:", response.text)
