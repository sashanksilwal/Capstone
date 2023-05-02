### File Description


`tranco_325_urls.txt`

This file contains the top 325 websites from the Tranco list. (top 350 with 25 url removed where authentication is required) The websites are in the form of a url per line.


`predictions.csv`

This file contains the predictions when the model is run our test webpages. The predictions are in the form of a csv file with the following format:

`url,js_url,prediction,probability`

`url` is the url of the webpage, `js_url` is the url of the javascript file that was requested by the webpage (for the url in the same row), `prediction` is the prediction of the model (12 classes), and `probability` is the probability of the prediction.

`size.csv`

This file contains the size of the javascript files that were requested by the test webpages. The size is in the form of a csv file with the following format:

`url,js_url,size(bytes)`