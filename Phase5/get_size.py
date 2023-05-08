import csv
import requests

# add the header to the file size.csv
with open("/Volumes/SAVE HERE/Capstone-1/Phase5/size_cat.csv", "w") as f:
    f.write("url,js_url,size(bytes)\n")
    f.close()

with open("/Volumes/SAVE HERE/Capstone-1/Phase5/predictions.csv") as f:
    rows = csv.reader(f)

    for row in rows:
        val = float(row[3])

        if   row[2] in ['ads', 'socials', 'analytics', 'marketing']:
            url = row[1]

            try:
                r = requests.get(url)

                if r.status_code != 200:
                    print(f"Error: Request failed with status code {r.status_code}. Skipping to the next line.")
                    continue

                content_length = r.headers.get('Content-Length')

                if content_length is not None:
                    size = int(content_length)
                else:
                    size = len(r.content)

                with open("/Volumes/SAVE HERE/Capstone-1/Phase5/size_cat.csv", "a") as f:
                    f.write(row[0] + "," + row[1] + "," + str(size) + "\n")
                    f.close()

            except requests.exceptions.RequestException as e:
                print(f"Error: {e}. Skipping to the next line.")
                continue

        else:
            continue
