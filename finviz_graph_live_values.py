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
import os

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


def save_chart_img(driver, ticker, image_folder: str):
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

        img_path = f"{image_folder}/{ticker}_chart.png"
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


def scan_chart_image(ticker: str, image_folder: str):
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

        chart_img_path = save_chart_img(driver, ticker, image_folder)
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
        focused_lines_path = img_path.replace('.png', '_focused_lines.png')
        cv2.imwrite(focused_lines_path, result_image)

        logging.info("Successfully saved image based on lines")
        return focused_lines_path
    
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
        red_lines_on_whitespace_path = img_path.replace('.png', '_red_lines_on_whitespace.png')
        cv2.imwrite(red_lines_on_whitespace_path, img)
        
        logging.info(f"Successfully painted red lines on whitespace for {ticker}")
        return red_lines_on_whitespace_path
    
    except Exception as e:
        logging.error(f"Failed to paint red lines on whitespace for {ticker}. Error: {str(e)}")
        return None


def get_numbers_from_image(img_path: str) -> list:
    logging.info(f"Reading number for img path: {img_path}")
    
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
    
    logging.info(f"Finished get_numbers_from_image for {img_path}")
    return numbers


def save_imgs_using_croping(ticker: str, img_path: str, vertical_line_color: tuple):
    logging.info(f"Starting save_imgs_using_croping for {ticker}")
    
    # Load the image
    img = cv2.imread(img_path)
    
    # Find the x-coordinate of the vertical line from right
    height, width = img.shape[:2]
    for x in range(width - 1, -1, -1):
        if np.all(img[:, x] == vertical_line_color):
            break
    
    # Crop the image from the left edge to just before the vertical line
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
        return []

    height, width = img.shape[:2]
    
    # Function to check if a pixel is red (within a tolerance)
    def is_red(pixel):
        return np.all(np.abs(pixel - rad_rgb) < 10)
    
    # Initialize variables
    crop_start = 0
    extracted_paths = []
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
                        extracted_img_path = img_path.replace('.png', f'_extracted_num_{len(extracted_paths)}.png')
                        cv2.imwrite(extracted_img_path, cropped)
                        extracted_paths.append(extracted_img_path)
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
                        extracted_img_path = img_path.replace('.png', f'_extracted_num_{len(extracted_paths)}.png')
                        cv2.imwrite(extracted_img_path, cropped)
                        extracted_paths.append(extracted_img_path)
                    else:
                        logging.warning(f"Skipped empty crop at x={x}")
                crop_start = x + 1
    
    # Save the last crop
    if is_vertical_scan:
        last_crop = img[crop_start:, :]
    else:
        last_crop = img[:, crop_start:]
    
    if not last_crop.size == 0:
        extracted_img_path = img_path.replace('.png', f'_extracted_num_{len(extracted_paths)}.png')
        cv2.imwrite(extracted_img_path, last_crop)
        extracted_paths.append(extracted_img_path)
    else:
        logging.warning("Skipped empty last crop")
    
    if not red_lines_found:
        logging.warning("No red lines found in the image")
    
    logging.info(f"Finished crop_img_using_red_line for {ticker}. Created {len(extracted_paths)} crops.")
    return extracted_paths


def cover_vertical_rgb_lines_when_rgb_pixel_found(img_path: str, found_pixel_rgb: tuple, covered_line_rgb: tuple):
    logging.info(f"Starting cover_vertical_rgb_lines_when_rgb_pixel_found for image: {img_path}")
    
    # Load the image
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    
    # Function to check if a pixel is black (within a tolerance)
    def is_black(pixel):
        return np.all(np.abs(pixel - found_pixel_rgb) < 70)
    
    # Scan the image for black pixels
    for y in range(height):
        if any(is_black(img[y, x]) for x in range(width)):
            # Draw a horizontal white line
            cv2.line(img, (0, y), (width, y), covered_line_rgb, 1)
    
    # Save the modified image
    output_path = img_path.replace('.png', '_covered.png')
    cv2.imwrite(output_path, img)
    
    logging.info(f"Finished cover_vertical_rgb_lines_when_rgb_pixel_found. Output saved to: {output_path}")
    return output_path


def read_stocks_from_csv(filename: str) -> pd.DataFrame:
    try:
        stocks_df = pd.read_csv(filename)
        logging.info(f"Successfully read data from {filename}")
        return stocks_df
    except Exception as e:
        logging.error(f"Error reading CSV file {filename}: {str(e)}")
        return pd.DataFrame()


def main(pattern: str, base_folder: str, filename: str, found_rgb_lines: tuple):

    stocks_df = read_stocks_from_csv(filename)
    image_folder = f'{base_folder}/{pattern}_images'

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    first_ticker = stocks_df['Ticker'].iloc[0]

    if first_ticker:

        chart_img_path = scan_chart_image(ticker=first_ticker, image_folder=image_folder)

        focus_on_lines_img_path = save_img_using_shape_from_color_lines(ticker=first_ticker, img_path=chart_img_path, color_rgb=found_rgb_lines, line_radius=60)

        vertical_cropped_img_path = save_imgs_using_croping(ticker=first_ticker, img_path=focus_on_lines_img_path, vertical_line_color=(0, 0, 255))

        covered_on_black_img_path = cover_vertical_rgb_lines_when_rgb_pixel_found(img_path=vertical_cropped_img_path, found_pixel_rgb=(0, 0, 0), covered_line_rgb=(255, 255, 255))

        painted_img_path = paint_red_line_white_space(ticker=first_ticker, img_path=covered_on_black_img_path, rad_rgb=(0, 0, 255))
        
        extracted_num_paths = crop_img_using_red_line(ticker=first_ticker, img_path=painted_img_path, rad_rgb=(0, 0, 255), is_vertical_scan=True)

        for img_path in extracted_num_paths:
            numbers = get_numbers_from_image(img_path)
            print(f"Numbers found in the image: {numbers}")


if __name__ == "__main__":

    pattern = 'ta_p_channel'
    base_folder = 'assets'
    filename = f'{base_folder}/stocks_pattern_{pattern}.csv'
    found_blue_lines = (37, 111, 149)
    # found_parpel_lines = (142, 73, 156)

    main(pattern, base_folder, filename, found_blue_lines)

