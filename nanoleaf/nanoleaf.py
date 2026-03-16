import requests

NANOLEAF_IP = "192.168.0.188"
TOKEN_FILE = "token.txt"


def get_token():
    with open(TOKEN_FILE, "r") as f:
        return f.read().strip()


def send_power_state(state: bool):
    token = get_token()
    url = f"http://{NANOLEAF_IP}:16021/api/v1/{token}/state"

    payload = {
        "on": {"value": state}
    }

    response = requests.put(url, json=payload, timeout=5)

    if response.status_code == 204:
        print("Lights ON" if state else "Lights OFF")
    else:
        print("Unexpected response:", response.status_code, response.text)


def turn_on():
    send_power_state(True)


def turn_off():
    send_power_state(False)


if __name__ == "__main__":
    # simple test when run directly
    print("Turning lights ON...")
    turn_on()

    input("Press Enter to turn lights OFF...")

    turn_off()