import requests
from bs4 import BeautifulSoup

def get_dob(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    info_table = soup.find('table', {'class': 'infobox biography vcard'})
    trs = info_table.find_all('tr')

    for tr in trs:
        ths = tr.find_all('th')
        for th in ths:
            if 'Born' in th.text:
                dob = tr.find('span', {'class': 'bday'}).text
                return dob

url = 'https://en.wikipedia.org/wiki/Aaron_Swartz'
print(get_dob(url))
