import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def get_stocks_details_of_pattern(url, page_num):
    logging.info(f'Getting stocks from {url} page {page_num}')

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Try to find the table with different class names
    table = soup.find('table', class_='table-light')
    if table is None:
        table = soup.find('table', class_='screener_table')
    if table is None:
        table = soup.find('table', {'id': 'screener-content'})

    if table is None:
        logging.info("Could not find the table. The website structure might have changed.")
        return pd.DataFrame()

    rows = table.find_all('tr')[1:]  # Skip the header row

    data = []
    for row in rows:
        cols = row.find_all('td')
        if len(cols) > 10:
            ticker = cols[1].text.strip()
            company = cols[2].text.strip()
            sector = cols[3].text.strip()
            industry = cols[4].text.strip()
            country = cols[5].text.strip()
            market_cap = cols[6].text.strip()
            price = cols[8].text.strip()
            change = cols[9].text.strip()
            volume = cols[10].text.strip()

            data.append({
                'Ticker': ticker,
                'Company': company,
                'Sector': sector,
                'Industry': industry,
                'Country': country,
                'Market Cap': market_cap,
                'Price': price,
                'Change': change,
                'Volume': volume
            })

    df = pd.DataFrame(data)
    return df


def save_to_csv(df, filename):
    logging.info("Saving data to CSV")
    df.to_csv(filename, index=False)
    logging.info(f"Data saved to {filename}")


def main(screener_url, pattern, total_pages, page_size, objects, filename):

    pattern_df = pd.DataFrame()
    tmp_pattern_df = pd.DataFrame()

    for i in range(total_pages):
        full_url = f'{screener_url}{pattern}&r={objects}'

        tmp_pattern_df = get_stocks_details_of_pattern(full_url, i)
        pattern_df = pd.concat([pattern_df, tmp_pattern_df], ignore_index=True)

        if i == 0:
            objects = page_size
        else:
            objects += page_size

    save_to_csv(pattern_df, filename)
    return pattern_df