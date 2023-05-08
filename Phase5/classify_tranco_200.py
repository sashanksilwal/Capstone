import csv
import requests
import time
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

retry_attempts = 3

# Function for retry logic
def make_request_with_retry(url):
    for i in range(retry_attempts):
        try:
            response = session.get(url)
            response.raise_for_status()
            return response
        except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError):
            if i < retry_attempts - 1:
                print(f"Attempt {i + 1} failed. Retrying after 5 seconds...")
                time.sleep(5)
            else:                
                print(f"Max retry attempts reached for {url}. Exiting...")
                # log url to a file 
                with open('failed_urls.txt', 'a') as f:
                    f.write(url)
                    
                return None

# Make request to the URL and get the response with retry logic
session = requests.Session()
# open csv file top-350.csv which contains the top 350 websites from tranco 
with open('top-400.csv', 'r') as f:
    # read the csv file and store the second column data in a list
    reader = csv.reader(f)
    data_list = [row[0] for row in reader]

print(data_list[:5])
# Get the first 100 lines of data
# data_list = data_list[:200]

# Function to make URLs absolute
def make_url_absolute(url):
    if not url.startswith("http://") and not url.startswith("https://"):
        # if the url contains .js in it, then the part till .js
        if '.js' in url:
            return "https://" + url.split('.js')[0] + '.js'
        return "https://" + url
    return url

def make_url_absolute_js(url, base_url):
    if not url.startswith("http://") and not url.startswith("https://"):
        # if the url contains .js in it, then the part till .js
        if '.js' in url:
            return "https://" +base_url+ url.split('.js')[0] + '.js'
        
    return url

# Find JS URLs and make them absolute
def get_js_urls(url):
    response = make_request_with_retry(make_url_absolute(url))
    if response:
        soup = BeautifulSoup(response.content, 'html.parser')
        js_urls = [make_url_absolute_js(tag['src'], url).split(".js")[0]+".js" for tag in soup.find_all('script', src=True) if '.js' in tag['src']]
        return js_urls
    else:
        return []

# Use ThreadPoolExecutor to execute get_js_urls function in parallel
with ThreadPoolExecutor(max_workers=20) as executor:

    # Write URLs to CSV file using a context manager
    with open('js_urls.csv', 'a') as f:
        writer = csv.writer(f)

        for url in data_list:
            # Get the JS URLs for the current website
            js_urls = executor.submit(get_js_urls, url).result()

            # Write the JS URLs to the CSV file
            for js_url in js_urls:
                writer.writerow([make_url_absolute(url), js_url])
                f.flush()

