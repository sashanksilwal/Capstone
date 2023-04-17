import csv
import requests
import time
from bs4 import BeautifulSoup

url = "https://gist.githubusercontent.com/bejaneps/ba8d8eed85b0c289a05c750b3d825f61/raw/6827168570520ded27c102730e442f35fb4b6a6d/websites.csv"

# Make request to the URL and get the response with retry logic
retry_attempts = 3
for i in range(retry_attempts):
    try:
        response = requests.get(url)
        break  # Break the loop if request is successful
    except requests.exceptions.ConnectionError:
        if i < retry_attempts - 1:
            print(f"Attempt {i + 1} failed. Retrying after 5 seconds...")
            time.sleep(5)
        else:
            print("Max retry attempts reached. Exiting...")

# Decode the response content as text
content = response.text

# Create a CSV reader object
csv_reader = csv.reader(content.splitlines(), delimiter=',')

# Skip the header row
next(csv_reader)

# Store the second column data in a list
data_list = []
for row in csv_reader:
    data_list.append(row[1])

# Get the first 100 lines of data
data_list = data_list[:100]

for url in data_list:
    # Add "https://" scheme to the URL if it doesn't have one
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    # Make request to the URL and get the response with retry logic
    for i in range(retry_attempts):
        try:
            response = requests.get(url)
            break  # Break the loop if request is successful
        except requests.exceptions.ConnectionError:
            if i < retry_attempts - 1:
                print(f"Attempt {i + 1} failed. Retrying after 5 seconds...")
                time.sleep(5)
            else:
                print("Max retry attempts reached. Skipping URL...", url)
                continue

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all script tags with src attribute containing .js
    js_urls = [tag['src'] for tag in soup.find_all('script', src=True) if '.js' in tag['src']]
    # if the url is relative, make it absolute and add https:// scheme to it if it doesn't have one already 
    js_urls = [url + tag['src'] if not tag['src'].startswith("http://") and not tag['src'].startswith("https://") else tag['src'] for tag in soup.find_all('script', src=True) if '.js' in tag['src']]

    # Add the URLs to a CSV file
    with open('js_urls.csv', 'a') as f:
        writer = csv.writer(f)
        for js_url in js_urls:
            writer.writerow([js_url])
