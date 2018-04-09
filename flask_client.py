
# A small script to try out web app
import requests
import time

#url = "http://localhost:5000/"
url = 'https://image-captioning-196706.appspot.com/'
f_data = open('[image_path].jpg', 'rb')
file = {'file': f_data}
start_time = time.time()
response = requests.post(url, files=file)
caption  = response.json()
print('Done. Request took {:.2f}s'.format(time.time() - start_time))

print(caption)