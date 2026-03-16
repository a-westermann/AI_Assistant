import requests
import sys

# Replace with your Nanoleaf controller IP
NANOLEAF_IP = "192.168.0.188"

url = f"http://{NANOLEAF_IP}:16021/api/v1/new"

try:
    response = requests.post(url, timeout=5)

    if response.status_code == 200:
        data = response.json()
        token = data.get("auth_token")

        if token:
            print("Auth token received:")
            print(token)
        else:
            print("Response received but no token found:")
            print(data)

    else:
        print(f"Unexpected status code: {response.status_code}")
        print(response.text)

except requests.exceptions.RequestException as e:
    print("Request failed:", e)
    