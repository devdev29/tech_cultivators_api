from flask import Flask
import flask
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

bazaar_page=requests.get('https://msamb.com/').content
souped_page=BeautifulSoup(bazaar_page, 'html.parser')
final_bhaav={'commodity':[], 'qty':[], 'price':[]}
bhaav_html=souped_page.find_all(name='tr')[1:]

for bhaav in bhaav_html:
        final_bhaav['commodity'].append(bhaav.find_all('td')[0].text)
        final_bhaav['qty'].append(bhaav.find_all('td')[1].text)
        final_bhaav['price'].append(bhaav.find_all('td')[3].text)

print(final_bhaav)
@app.route("/bazaar_bhav",methods=['GET'])
def get_bhaav():
        bazaar_page=requests.get('https://msamb.com/').content
        souped_page=BeautifulSoup(bazaar_page, 'html.parser')
        final_bhaav={'commodity':[], 'qty':[], 'price':[]}
        bhaav_html=souped_page.find_all(name='tr')[1:]

        for bhaav in bhaav_html:
                final_bhaav['commodity'].append(bhaav.find_all('td')[0].text)
                final_bhaav['qty'].append(bhaav.find_all('td')[1].text)
                final_bhaav['price'].append(bhaav.find_all('td')[3].text)
        
        return flask.jsonify(final_bhaav)