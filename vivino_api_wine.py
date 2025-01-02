import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
url = "https://www.vivino.com/wines/2160185"  # Replace with winery ID or endpoint
response = requests.get(url, headers=headers)
data = response.json()

print(data)
# Extract wine details
# for wine in data['wines']:
#     print(f"Wine: {wine['name']}, Rating: {wine['statistics']['average_rating']}")
