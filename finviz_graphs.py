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
import re

import matplotlib.pyplot as plt


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

html_tags = {
    'theme_tag': {
        'tagDataTestID': 'a[data-testid="chart-layout-theme"]',
        'WebDriverWait': 2
    },
    'close_popup_tag': {
        'tagID': 'aymStickyFooterClose',
        'sleep_before': 3,
        'WebDriverWait': 3
    },
    'chart_tag': {
        'tag': 'canvas',
        'sleep_before': 5,
        'WebDriverWait': 5
    }
}

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

def save_chart_img(driver, ticker):
    logging.info(f"Processing chart image for {ticker}")
    try:
        time.sleep(html_tags['chart_tag']['sleep_before'])
        # Find the first canvas element within the chart
        canvas = WebDriverWait(driver, html_tags['chart_tag']['WebDriverWait']).until(
            EC.presence_of_element_located((By.TAG_NAME, html_tags['chart_tag']['tag']))
        )

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
        time.sleep(html_tags['close_popup_tag']['sleep_before'])
        close_button = WebDriverWait(driver, html_tags['close_popup_tag']['WebDriverWait']).until( EC.element_to_be_clickable((By.ID, html_tags['close_popup_tag']['tagID'])) )
        close_button.click()

        logging.info("Popup ad closed successfully")
    except Exception as e:
        logging.warning(f"No popup ad found or unable to close: {str(e)}")

def apply_white_theme(driver):
    logging.info("Attempting to apply white theme")
    try:
        theme_button = WebDriverWait(driver, html_tags['theme_tag']['WebDriverWait']).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, html_tags['theme_tag']['tagDataTestID']))
        )
        theme_button.click()
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

def scan_chart_image(ticker: str):
    logging.info(f"Starting chart scan for {ticker}")
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--ignore-ssl-errors")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(f"https://finviz.com/quote.ashx?t={ticker}&ty=c&p=d&b=1")

        apply_white_theme(driver)
        close_popup_ad(driver)

        chart_img_path = save_chart_img(driver, ticker)
        logging.info(f"Chart image saved for {ticker} at {chart_img_path}")
        return chart_img_path

    except Exception as e:
        logging.error(f"Error processing chart for {ticker}: {str(e)}")
        return None
    finally:
        if driver:
            driver.quit()

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
                
                vertical_line_color = (0, 0, 255)
                # Add a definite sign (red vertical line) on the right side of blue lines
                cv2.line(result_image, (x2, 0), (x2, result_image.shape[0]), vertical_line_color, 2)

        # Save the image with the blue line and surrounding text
        img_path = f"{ticker}_focus_on_lines.png"
        cv2.imwrite(img_path, result_image)

        logging.info("Successfully saved image based on lines")
        return img_path
    
    except Exception as e:
        logging.error(f"Failed to save image based on lines. Error: {str(e)}")
        return None

def paint_red_line_white_space(ticker: str, img_path: str, rad_rgb: tuple):
    try:
        # Read the image
        img = cv2.imread(img_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get image dimensions
        height, width = gray.shape
        
        # Scan the image from top to bottom
        for y in range(height):
            row = gray[y]
            if np.all(row > 200):  # Assuming white space has pixel values > 200
                # Draw a red horizontal line
                cv2.line(img, (0, y), (width, y), (0, 0, 255), 1)
        
        # Save the modified image
        output_path = f"{ticker}_red_lines_on_whitespace.png"
        cv2.imwrite(output_path, img)
        
        logging.info(f"Successfully painted red lines on whitespace for {ticker}")
        return output_path
    
    except Exception as e:
        logging.error(f"Failed to paint red lines on whitespace for {ticker}. Error: {str(e)}")
        return None

def get_numbers_from_image(img_path: str) -> list:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    img = cv2.imread(img_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)

    # Apply adaptive thresholding
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

    # Apply sharpening filter
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(thresholded, -1, kernel)

    # Add white margin space around the sharpened image
    margin = 5  # Adjust this value to increase or decrease the margin size
    height, width = sharpened.shape
    white_background = np.ones((height + 2*margin, width + 2*margin), dtype=np.uint8) * 255
    white_background[margin:margin+height, margin:margin+width] = sharpened
    
    # Update sharpened with the new image that includes white margins
    sharpened = white_background

    threshold_img_path = img_path.replace('.png', '_threshold.png')
    cv2.imwrite(threshold_img_path, sharpened)

    numbers = pytesseract.image_to_string(sharpened)
    
    return numbers

def save_imgs_using_croping(ticker: str, img_path: str, vertical_line_color: tuple):
    logging.info(f"Starting save_imgs_using_croping for {ticker}")
    
    # Load the image
    img = cv2.imread(img_path)
    
    # Find the x-coordinate of the vertical line
    height, width = img.shape[:2]
    for x in range(width):
        if np.all(img[:, x] == vertical_line_color):
            break
    
    # Crop the image from just after the vertical line to the right
    cropped_img = img[:, x+5:]
    
    # Save the cropped image
    cropped_img_path = img_path.replace('.png', '_cropped.png')
    cv2.imwrite(cropped_img_path, cropped_img)
    
    logging.info(f"Finished save_imgs_using_croping for {ticker}")
    return cropped_img_path

def crop_img_using_red_line(ticker: str, img_path: str, rad_rgb: tuple, is_vertical_scan: bool):
    logging.info(f"Starting crop_img_using_red_line for {ticker}")
    
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        logging.error(f"Failed to load image: {img_path}")
        return 0

    height, width = img.shape[:2]
    
    # Function to check if a pixel is red (within a tolerance)
    def is_red(pixel):
        return np.all(np.abs(pixel - rad_rgb) < 10)
    
    # Initialize variables
    crop_start = 0
    crop_count = 0
    red_lines_found = False
    
    if is_vertical_scan:
        # Scan from top to bottom
        for y in range(height):
            if any(is_red(img[y, x]) for x in range(width)):
                red_lines_found = True
                if crop_start < y:
                    # Crop and save the image
                    cropped = img[crop_start:y, :]
                    if not cropped.size == 0:
                        cv2.imwrite(f"{ticker}_crop_{crop_count}.png", cropped)
                        crop_count += 1
                    else:
                        logging.warning(f"Skipped empty crop at y={y}")
                crop_start = y + 1
    else:
        # Scan from left to right
        for x in range(width):
            if any(is_red(img[y, x]) for y in range(height)):
                red_lines_found = True
                if crop_start < x:
                    # Crop and save the image
                    cropped = img[:, crop_start:x]
                    if not cropped.size == 0:
                        cv2.imwrite(f"{ticker}_crop_{crop_count}.png", cropped)
                        crop_count += 1
                    else:
                        logging.warning(f"Skipped empty crop at x={x}")
                crop_start = x + 1
    
    # Save the last crop
    if is_vertical_scan:
        last_crop = img[crop_start:, :]
    else:
        last_crop = img[:, crop_start:]
    
    if not last_crop.size == 0:
        cv2.imwrite(f"{ticker}_crop_{crop_count}.png", last_crop)
        crop_count += 1
    else:
        logging.warning("Skipped empty last crop")
    
    if not red_lines_found:
        logging.warning("No red lines found in the image")
    
    logging.info(f"Finished crop_img_using_red_line for {ticker}. Created {crop_count} crops.")
    return crop_count  # Return the number of crops created

def cover_white_on_black(img_path: str, black_rgb: tuple, white_rgb: tuple):
    logging.info(f"Starting cover_white_on_black for image: {img_path}")
    
    # Load the image
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    
    # Function to check if a pixel is black (within a tolerance)
    def is_black(pixel):
        return np.all(np.abs(pixel - black_rgb) < 70)
    
    # Scan the image for black pixels
    for y in range(height):
        if any(is_black(img[y, x]) for x in range(width)):
            # Draw a horizontal white line
            cv2.line(img, (0, y), (width, y), white_rgb, 1)
    
    # Save the modified image
    output_path = img_path.replace('.png', '_covered.png')
    cv2.imwrite(output_path, img)
    
    logging.info(f"Finished cover_white_on_black. Output saved to: {output_path}")
    return output_path

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
                focus_on_lines_img_path = scan_chart_image(first_ticker, color_rgb=(37, 111, 149), radius=50)
                if focus_on_lines_img_path:
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
    
    ticker = 'GO'

    chart_img_path = scan_chart_image(ticker=ticker)
    # chart_img_path = f'{ticker}_chart.png'

    focus_on_lines_img_path = save_img_using_shape_from_color_lines(ticker=ticker, img_path=chart_img_path, color_rgb=(37, 111, 149), line_radius=60)

    vertical_cropped_img_path = save_imgs_using_croping(ticker=ticker, img_path=focus_on_lines_img_path, vertical_line_color=(0, 0, 255))

    covered_img_path = cover_white_on_black(img_path=vertical_cropped_img_path, black_rgb=(0, 0, 0), white_rgb=(255, 255, 255))

    painted_img_path = paint_red_line_white_space(ticker=ticker, img_path=covered_img_path, rad_rgb=(0, 0, 255))
    
    number_of_croppd = crop_img_using_red_line(ticker=ticker, img_path=painted_img_path, rad_rgb=(0, 0, 255), is_vertical_scan=True)

    count = number_of_croppd
    print(f"Count: {count}")
    for i in range(count):

        print(f"Processing crop {i}")
        img_path = f'{ticker}_crop_{i}.png'
        numbers = get_numbers_from_image(img_path)
        print(f"Numbers found in the image: {numbers}")

