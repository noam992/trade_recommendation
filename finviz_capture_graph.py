import os
import io
import time
import logging
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options


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


def close_popup_privacy(driver):
    logging.info("Attempting to close privacy popup")
    try:
        time.sleep(html_tags['close_popup_tag']['sleep_before'])
        close_button = WebDriverWait(driver, html_tags['close_popup_tag']['WebDriverWait']).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[mode="primary"]'))
        )
        close_button.click()
        logging.info("Privacy popup closed successfully")
    except Exception as e:
        logging.warning(f"No privacy popup found or unable to close: {str(e)}")


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


def close_popup_ad(driver):
    logging.info("Attempting to close popup ad")
    try:
        time.sleep(html_tags['close_popup_tag']['sleep_before'])
        close_button = WebDriverWait(driver, html_tags['close_popup_tag']['WebDriverWait']).until( EC.element_to_be_clickable((By.ID, html_tags['close_popup_tag']['tagID'])) )
        close_button.click()

        logging.info("Popup ad closed successfully")
    except Exception as e:
        logging.warning(f"No popup ad found or unable to close: {str(e)}")


def scroll_to_bottom(driver, scroll_amount: int):
    driver.execute_script(f"window.scrollBy(0, {scroll_amount});")


def save_chart_img(driver, ticker, image_folder: str, scroll_amount: int):
    logging.info(f"Processing chart image for {ticker}")
    try:
        time.sleep(html_tags['chart_tag']['sleep_before'])
        # Find the first canvas element within the chart
        canvas = WebDriverWait(driver, html_tags['chart_tag']['WebDriverWait']).until(
            EC.presence_of_element_located((By.TAG_NAME, html_tags['chart_tag']['tag']))
        )

        screenshot = driver.get_screenshot_as_png()
        screenshot = Image.open(io.BytesIO(screenshot))

        # Get the device pixel ratio to account for zoom
        device_pixel_ratio = driver.execute_script('return window.devicePixelRatio;')

        canvas_location = canvas.location
        canvas_size = canvas.size

        # Adjust coordinates for zoom level and scroll position
        left = int(canvas_location['x'] * device_pixel_ratio)
        top = int((canvas_location['y'] - scroll_amount) * device_pixel_ratio)  # Adjust for scroll
        right = int((canvas_location['x'] + canvas_size['width']) * device_pixel_ratio)
        bottom = int((canvas_location['y'] - scroll_amount + canvas_size['height']) * device_pixel_ratio)

        chart_image = screenshot.crop((left, top, right, bottom))

        img_path = f"{image_folder}/{ticker}_chart.png"
        chart_image.save(img_path)

        logging.info(f"Successfully saved chart image for {ticker}")
        return img_path
    
    except Exception as e:
        logging.error(f"Failed to process chart image for {ticker}. Error: {str(e)}")
        return False


def scan_chart_image(ticker: str, image_folder: str):
    scroll_amount = 280
    chrome_zoom = 1.50

    logging.info(f"Starting chart scan for {ticker}")
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--ignore-ssl-errors")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--enable-unsafe-swiftshader")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument(f"--force-device-scale-factor={chrome_zoom}")

    driver = None
    try:

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(f"https://finviz.com/quote.ashx?t={ticker}&ty=c&p=d&b=1")

        # close_popup_privacy(driver)

        apply_white_theme(driver)
        close_popup_ad(driver)
        scroll_to_bottom(driver, scroll_amount)

        chart_img_path = save_chart_img(driver, ticker, image_folder, scroll_amount)
        logging.info(f"Chart image saved for {ticker} at {chart_img_path}")
        return chart_img_path

    except Exception as e:
        logging.error(f"Error processing chart for {ticker}: {str(e)}")
        return None
    finally:
        if driver:
            driver.quit()


def main(pattern: str, base_folder: str, ticker_name: str):

    image_folder = f'{base_folder}/{pattern}_images'

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    graph_img_path = scan_chart_image(ticker=ticker_name, image_folder=image_folder)
    return graph_img_path


# if __name__ == "__main__":
#     result = main(pattern = 'ta_p_channel', base_folder = 'assets', ticker_name = 'ACCD')
#     print(result)