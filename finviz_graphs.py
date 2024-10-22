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
from datetime import datetime

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
        print("Could not find the table. The website structure might have changed.")
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
    print(f"Data saved to {filename}")

def scan_chart_image(ticker='ADEA'):
    driver = webdriver.Chrome()  # Make sure you have chromedriver installed and in PATH
    driver.get(f"https://finviz.com/quote.ashx?t={ticker}&ty=c&p=d&b=1")

    # Wait for the chart to load and take a screenshot
    chart = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "chart0")))
    chart.screenshot(f"{ticker}_chart.png")
    driver.quit()

    # Process the image
    image = cv2.imread(f"{ticker}_chart.png")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for purple color in HSV
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([170, 255, 255])

    # Create a mask for purple color
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Edge detection
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)

    # Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    # Draw lines on the original image
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(f"{ticker}_chart_with_lines.png", image)
    print(f"Processed chart saved as {ticker}_chart_with_lines.png")

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
                print(double_bottom_stocks)
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
