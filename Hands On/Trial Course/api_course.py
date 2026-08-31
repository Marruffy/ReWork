import requests

url = 'https://v2.jokeapi.dev/joke/Programming'

params = {
    'type': 'single',
}

try:
    respo = requests.get(url, params=params)
    #respo.raise_for_status()  # Raise an exception for HTTP errors
    status = respo.status_code
    if status == 400:
        print("Bad Request: The request was invalid or cannot be served.")

#print(respo.status_code)