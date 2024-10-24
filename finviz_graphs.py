import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
from PIL import Image
import io
from datetime import datetime
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def get_double_bottom_stocks():
    url = "https://finviz.com/screener.ashx?v=111&f=ta_pattern_doublebottom"
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

def save_to_csv(df, filename='double_bottom_stocks.csv'):
    df.to_csv(filename, index=False)
    logging.info(f"Data saved to {filename}")

def process_chart_image(driver, chart, ticker):
    logging.info(f"Processing chart image for {ticker}")
    try:
        # Find the canvas element within the chart
        canvas = chart.find_element(By.CSS_SELECTOR, "canvas.second")

        screenshot = driver.get_screenshot_as_png()
        screenshot = Image.open(io.BytesIO(screenshot))

        canvas_location = canvas.location
        canvas_size = canvas.size

        left = canvas_location['x']
        top = canvas_location['y']
        right = left + canvas_size['width']
        bottom = top + canvas_size['height']

        chart_image = screenshot.crop((left, top, right, bottom))

        chart_image.save(f"{ticker}_chart.png")

        return True
    except Exception as e:
        logging.error(f"Error processing chart image: {str(e)}")
        return False

def close_popup_ad(driver):
    logging.info("Attempting to close popup ad")
    try:
        # Wait for the close button to be clickable
        close_button = WebDriverWait(driver, 1).until(
            EC.element_to_be_clickable((By.ID, "aymStickyFooterClose"))
        )
        close_button.click()
        logging.info("Popup ad closed successfully")
    except Exception as e:
        logging.warning(f"No popup ad found or unable to close: {str(e)}")

def scan_chart_image(ticker='ADEA'):
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--ignore-ssl-errors")

    # Initialize the Chrome driver with the options
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(f"https://finviz.com/quote.ashx?t={ticker}&ty=c&p=d&b=1")

    chart = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "chart")))

    # Close popup ad
    close_popup_ad(driver)

    if process_chart_image(driver, chart, ticker):
        logging.info(f"Successfully captured chart for {ticker}")
    else:
        logging.error(f"Failed to capture chart for {ticker}")

    driver.quit()

def main():
    max_retries = 2
    double_bottom_stocks = pd.DataFrame()

    # Retry loop for get_double_bottom_stocks
    for attempt in range(max_retries):
        try:
            logging.info("Attempting to get double bottom stocks")
            double_bottom_stocks = get_double_bottom_stocks()
            if not double_bottom_stocks.empty:
                logging.info("Successfully retrieved double bottom stocks")
                break
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed to get double bottom stocks. Error: {str(e)}")
            if attempt < max_retries - 1:
                logging.info("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                logging.error("Max retries reached. Unable to fetch double bottom stocks data.")

    if not double_bottom_stocks.empty:
        logging.info("Saving data to CSV")
        save_to_csv(double_bottom_stocks)
        
        first_ticker = double_bottom_stocks['Ticker'].iloc[0]
        
        # Retry loop for scan_chart_image
        for attempt in range(max_retries):
            try:
                logging.info(f"Attempting to scan chart image for {first_ticker}")
                scan_chart_image(first_ticker)
                logging.info(f"Successfully scanned chart image for {first_ticker}")
                break
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed to scan chart image. Error: {str(e)}")
                if attempt < max_retries - 1:
                    logging.info("Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    logging.error("Max retries reached. Unable to scan chart image.")
    else:
        logging.warning("No double bottom stocks found. Skipping CSV save and chart scan.")

if __name__ == "__main__":
    main()
