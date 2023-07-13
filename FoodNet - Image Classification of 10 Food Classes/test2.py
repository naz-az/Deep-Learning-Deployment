import requests
# resp = requests.post("http://127.0.0.1:5000/", files={'file': open('Test/wings-download.jpg','rb')})
resp = requests.post("http://127.0.0.1:5000/", files={'file': open('Test/fried-rice-download.jpg','rb')})

print(resp.json())