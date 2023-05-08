# read the file tranco_600_urls.txt only take the second column and write it to a csv file top-400.csv

import csv

with open('tranco_600_urls.txt', 'r') as in_file:

    # print each line but only the second value after the comma
    with open('top-400.csv', 'w') as out_file:
        for line in in_file:
            out_file.write(line.split(",")[1])
    # for line in in_file:
    #     print(line.split(",")[1])

    # stripped = (line.strip() for line in in_file)
    # lines = (line.split(",") for line in stripped if line)
    # with open('top-400.csv', 'w') as out_file:
    #     writer = csv.writer(out_file)
    #     writer.writerows(lines[1])

        