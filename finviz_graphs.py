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
import pytesseract

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

def save_chart_img(driver, chart, ticker):
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

        img_path = f"{ticker}_chart.png"
        chart_image.save(img_path)

        logging.info(f"Successfully saved chart image for {ticker}")
        return img_path
    
    except Exception as e:
        logging.error(f"Failed to process chart image for {ticker}. Error: {str(e)}")
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

def apply_white_theme(driver):
    logging.info("Attempting to apply white theme")
    try:
        theme_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[data-testid="chart-layout-theme"]'))
        )
        theme_button.click()
        time.sleep(2)
        logging.info("White theme applied successfully")
    except Exception as e:
        logging.error(f"Unable to apply white theme: {str(e)}")

def get_chart_lines(img_path: str, color_rgb: tuple[int, int, int]):
    # Read the image
    img = cv2.imread(img_path)
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Convert RGB color to HSV
    rgb_color = np.uint8([[color_rgb]])  # RGB color
    hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
    
    # Get the HSV values
    hue = hsv_color[0][0][0]
    
    # Define range of the color in HSV
    lower_bound = np.array([max(0, hue - 10), 100, 100])
    upper_bound = np.array([min(180, hue + 10), 255, 255])
    
    # Threshold the HSV image to get only the specified color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Bitwise-AND mask and original image
    color_only = cv2.bitwise_and(img, img, mask=mask)
    
    # Convert to grayscale
    gray = cv2.cvtColor(color_only, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    
    # Create a blank image to draw the lines on
    line_image = np.zeros_like(img)
    
    # Draw the lines on the blank image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (37, 111, 149), 2)
    
    # Save the image with only the specified color lines
    cv2.imwrite('ADEA_specific_color_lines.png', line_image)
    
    return line_image

def scan_chart_image(ticker: str, color_rgb: tuple[int, int, int], radius: int):

    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--ignore-ssl-errors")

    # Initialize the Chrome driver with the options
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(f"https://finviz.com/quote.ashx?t={ticker}&ty=c&p=d&b=1")

    apply_white_theme(driver)

    chart = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "chart")))

    close_popup_ad(driver)

    chart_img_path = save_chart_img(driver, chart, ticker)

    driver.quit()

    focus_on_lines_img_path = save_img_using_shape_from_color_lines(ticker=ticker, img_path=chart_img_path, color_rgb=color_rgb, line_radius=radius)

def save_img_using_shape_from_color_lines(ticker: str, img_path: str, color_rgb: tuple[int, int, int], line_radius: int):
    logging.info(f"Starting save image based on lines")
    try:
        # Read the image
        img = cv2.imread(img_path)
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Convert RGB color to HSV
        rgb_color = np.uint8([[color_rgb]])  # RGB color
        hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
        
        # Get the HSV values
        hue = hsv_color[0][0][0]
        
        # Define range of the blue color in HSV
        lower_bound = np.array([max(0, hue - 10), 100, 100])
        upper_bound = np.array([min(180, hue + 10), 255, 255])
        
        # Threshold the HSV image to get only the blue color
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Bitwise-AND mask and original image
        blue_only = cv2.bitwise_and(img, img, mask=mask)
        
        # Convert to grayscale
        gray = cv2.cvtColor(blue_only, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        # Create a blank mask to draw the shape
        shape_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Draw a thick line on the mask
                cv2.line(shape_mask, (x1, y1), (x2, y2), 255, thickness=line_radius*2)
        
        # Dilate the shape to create a more continuous area
        kernel = np.ones((5,5), np.uint8)
        shape_mask = cv2.dilate(shape_mask, kernel, iterations=2)
        
        # Apply the shape mask to the original image
        result_image = cv2.bitwise_and(img, img, mask=shape_mask)
        
        # Draw the blue lines on top of the result image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result_image, (x1, y1), (x2, y2), color_rgb, 2)

        # Save the image with the blue line and surrounding text
        img_path = f"{ticker}_focus_on_lines.png"
        cv2.imwrite(img_path, result_image)

        logging.info("Successfully saved image based on lines")
        return img_path
    
    except Exception as e:
        logging.error(f"Failed to save image based on lines. Error: {str(e)}")

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
    
    scan_chart_image(ticker='MYE', color_rgb=(37, 111, 149), radius=50)
    # save_img_using_shape_from_color_lines(img_path='ADEA_chart.png', color_rgb=(37, 111, 149), text_radius=50)
