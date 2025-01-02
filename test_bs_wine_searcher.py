import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

# URL of the Wine-Searcher page to scrape
url = "https://www.wine-searcher.com/find?Xwinename="+quote_plus("Chateau Cormeil-Figeac")+"/2020#t2"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
response = requests.get(url, headers=headers)
#print(response.status_code)

# Check if the request was successful
if response.status_code >= 200:
    # Parse the page content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    #print(soup.prettify())
    
    # Find all <div> elements with class 'info-card'
    info_cards = soup.find_all('div', class_='card-text')
    
    # Iterate over each info card and extract desired information
    for card in info_cards:
        print(card)
        # Example: Extract wine name, price, and rating from each card
        # wine_name = card.find('h2', class_='wine-name').text.strip() if card.find('h2', class_='wine-name') else 'N/A'
        # price = card.find('span', class_='price').text.strip() if card.find('span', class_='price') else 'N/A'
        # rating = card.find('span', class_='rating').text.strip() if card.find('span', class_='rating') else 'N/A'
        
        # Print the extracted information
        # print(f"Wine Name: {wine_name}, Price: {price}, Rating: {rating}")
else:
    print("Failed to retrieve the webpage.")