# LangChain with GPT demo website

## Start, init
1. Fork the repo
2. Go into the repo

https://github.com/neo1202/LangChain_GPT.git

## Frontend --- _React, tailwind_

#### `cd client`
switch to the client folder

#### `npm install`
and run npm install to install the packages

#### `npm run dev`
run this code, so you can
open [http://localhost:8080](http://localhost:8080) view it in browser.


## Backend --- _Flask_

 ```
 cd flask-server
 ```

 ```
 python -m venv .venv
 ```

#### `. .venv/bin/activate`

#### `pip install -r requirements.txt`

#### `python -m flask run`
run this code to active the backend server
the server runs at [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
you can go to [http://127.0.0.1:5000/data](http://127.0.0.1:5000/data) to see some msg to ensure you successfully open the server.
