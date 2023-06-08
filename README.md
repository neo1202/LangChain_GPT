# LangChain with GPT demo website

## Start, init
1. Fork the repo
2. Go into the repo

https://github.com/neo1202/LangChain_GPT.git

## Frontend --- _React, tailwind_


Go to the client folder
```
cd client
```
Install the packages using npm
```
npm install
```

run this code, so you can open website
```
npm run dev
```
[http://localhost:8080](http://localhost:8080) view it in browser.


## Backend --- _Flask_
Go to the server folder
 ```
 cd flask-server
 ```
 
####Create a config file and place your api keys in it
Create a config file
```
touch config.py
```
Put your Api keys in it
```python
API_KEY = "your_api_key_here"
SERP_API_KEY = "your_serpapi_key"
PINECONE_KEY = "your_pinecone_key"
```
 

Create virtual environment for depencies
 ```
 python -m venv .venv
 ```
Activate the virtual environment in your terminal
```
. .venv/bin/activate
```

Install the depencies for this .venv
```
pip install -r requirements.txt
```

*run this code to active the backend server*
```
python -m flask run
```

the server runs at [http://127.0.0.1:5000/](http://127.0.0.1:5000/) <br>
you can go to [http://127.0.0.1:5000/data](http://127.0.0.1:5000/data) to see some msg to ensure you successfully open the server.
