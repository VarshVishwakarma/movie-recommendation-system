import gdown

url = "https://drive.google.com/uc?id=1TaJ5E3y7I96dUURtkUEf4TTHoPOPSQ0Z"
gdown.download(url, "similarity.pkl", quiet=False)
