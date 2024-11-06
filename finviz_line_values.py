import os
import logging
import numpy as np
import cv2
import pytesseract


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


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


def detect_straight_line_by_color(ticker: str, img_path: str, color_rgb: tuple[int, int, int], line_frame: int, covered_line_rgb: tuple[int, int, int], detect_minLineLength: int):
    """
    Process image to detect horizontal lines (180 degrees) using HoughLinesP with parameters:
    threshold=40 (min votes needed), minLineLength=400 (min pixel length),
    maxLineGap=20 (max gap between line segments to treat as single line)
    """
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
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=detect_minLineLength, maxLineGap=20)
        
        # Create a blank mask to draw the shape
        shape_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        height = img.shape[0]
        width = img.shape[1]
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate angle of the line
                angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                # Only process horizontal lines (180 degrees with some tolerance)
                if abs(angle - 180) < 5 or angle < 5:
                    # Draw horizontal lines across image at y position of detected line
                    # Add 50 pixels up and down from line position
                    y_start = max(0, y1 - line_frame)
                    y_end = min(height, y1 + line_frame)
                    for y in range(y_start, y_end):
                        cv2.line(shape_mask, (0, y), (width, y), 255, 1)
        
        # Apply the shape mask to the original image
        img = cv2.bitwise_and(img, img, mask=shape_mask)
        
        # Draw the orange lines on top of the result image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if abs(angle - 180) < 5 or angle < 5:
                    cv2.line(img, (x1, y1), (x2, y2), covered_line_rgb, 2)
                    
                    vertical_line_color = (0, 0, 255)
                    # Add a definite sign (red vertical line) on the right side of orange lines
                    cv2.line(img, (x2, 0), (x2, img.shape[0]), vertical_line_color, 2)

        # Save the image with the orange line and surrounding text
        img_path = img_path.replace('.png', '_focused_lines.png')
        cv2.imwrite(img_path, img)

        logging.info(f"Successfully saved image based on horizontal lines.")
    
        lines_counter = count_printed_lines(
            img_path=img_path,
            image=img,
            vertical_line_color=(0, 0, 255),
            covered_line_rgb=covered_line_rgb
        )
        
        if lines_counter != 1:
            text = f"Found {lines_counter}. expected to find 1"
            logging.info(text)
            return [], []
        
        return img_path, img
    
    except Exception as e:
        logging.error(f"Failed to save image based on lines. Error: {str(e)}")
        return None


def paint_red_line_white_space(ticker: str, img_path: str, image, rad_rgb: tuple):
    try:
        # Read the image
        img = image
        
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
        img_path = img_path.replace('.png', '_red_lines_on_whitespace.png')
        cv2.imwrite(img_path, img)
        
        logging.info(f"Successfully painted red lines on whitespace for {ticker}")
        return img_path, img
    
    except Exception as e:
        logging.error(f"Failed to paint red lines on whitespace for {ticker}. Error: {str(e)}")
        return None


def get_numbers_from_image(img_path: str, image) -> list:
    logging.info(f"Reading number for img path image")
    
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    img = image
    
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

    img_path = img_path.replace('.png', '_threshold.png')
    cv2.imwrite(img_path, sharpened)

    numbers = pytesseract.image_to_string(sharpened)
    
    logging.info(f"Finished get_numbers_from_image for {img_path}")
    return numbers


def crop_graph_area_by_red_line(ticker: str, img_path: str, image, vertical_line_color: tuple):
    logging.info(f"Starting crop_graph_area_by_red_line for {ticker}")
    
    # Load the image
    img = image # cv2.imread(img_path)
    
    # Find the x-coordinate of the vertical line from right
    height, width = img.shape[:2]
    for x in range(width - 1, -1, -1):
        if np.all(img[:, x] == vertical_line_color):
            break
    
    # Crop the image from the left edge to just before the vertical line
    cropped_img = img[:, x+5:]
    
    # Save the cropped image
    img_path = img_path.replace('.png', '_cropped.png')
    cv2.imwrite(img_path, cropped_img)
    
    logging.info(f"Finished crop_graph_area_by_red_line for {ticker}")
    return img_path, cropped_img


def crop_graph_area(ticker: str, img_path: str, image):
    logging.info(f"Starting crop_graph_area for {ticker}")
    
    # Load the image
    img = image
    
    # Keep only the rightmost 100 pixels
    height, width = img.shape[:2]
    cropped_img = img[:, max(0, width-80):]
    
    # Save the cropped image
    img_path = img_path.replace('.png', '_cropped.png')
    cv2.imwrite(img_path, cropped_img)
    
    logging.info(f"Finished crop_graph_area for {ticker}")
    return img_path, cropped_img


def crop_img_using_red_line(ticker: str, img_path: str, image, rad_rgb: tuple, is_vertical_scan: bool):
    logging.info(f"Starting crop_img_using_red_line for {ticker}")
    
    # Load the image
    img = image
    if img is None:
        logging.error(f"Failed to load image: {img_path}")
        return []

    height, width = img.shape[:2]
    
    # Function to check if a pixel is red (within a tolerance)
    def is_red(pixel):
        return np.all(np.abs(pixel - rad_rgb) < 10)
    
    # Initialize variables
    crop_start = 0
    imgs = []
    img_paths = []

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
                        img_path = img_path.replace('.png', f'_extracted_num_{len(imgs)}.png')
                        cv2.imwrite(img_path, cropped)
                        imgs.append(cropped)
                        img_paths.append(img_path)
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
                        img_path = img_path.replace('.png', f'_extracted_num_{len(imgs)}.png')
                        cv2.imwrite(img_path, cropped)
                        imgs.append(cropped)
                        img_paths.append(img_path)
                    else:
                        logging.warning(f"Skipped empty crop at x={x}")
                crop_start = x + 1
    
    # Save the last crop
    if is_vertical_scan:
        last_crop = img[crop_start:, :]
    else:
        last_crop = img[:, crop_start:]
    
    if not last_crop.size == 0:
        img_path = img_path.replace('.png', f'_extracted_num_{len(imgs)}.png')
        cv2.imwrite(img_path, last_crop)
        imgs.append(last_crop)
        img_paths.append(img_path)
    else:
        logging.warning("Skipped empty last crop")
    
    if not red_lines_found:
        logging.warning("No red lines found in the image")
    
    logging.info(f"Finished crop_img_using_red_line for {ticker}. Created {len(imgs)} crops.")
    return img_paths, imgs


def cover_vertical_rgb_lines_when_rgb_pixel_found(img_path: str, image, found_pixel_rgb: tuple, covered_line_rgb: tuple):
    logging.info(f"Starting cover_vertical_rgb_lines_when_rgb_pixel_found for image: {img_path}")
    
    # Load the image
    img = image
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
    return output_path, img


def replace_yellow_with_white(ticker: str, img_path: str, image, yellow_rgb: tuple, white_rgb: tuple):
    logging.info(f"Starting replace_yellow_with_white for {ticker}")
    try:
        # Load the image
        img = image.copy()  # Create a copy to avoid modifying original
        
        # Define yellow color range based on input yellow_rgb
        lower_yellow = np.array([max(0, c - 50) for c in yellow_rgb])
        upper_yellow = np.array([min(255, c + 50) for c in yellow_rgb])
        
        # Create mask for yellow pixels using RGB ranges
        yellow_mask = cv2.inRange(img, lower_yellow, upper_yellow)
        
        # Get dimensions of detected mask area
        mask_coords = cv2.findNonZero(yellow_mask)
        if mask_coords is not None:
            x, y, w, h = cv2.boundingRect(mask_coords)
            # Add small white border around detected yellow area
            border_size = 1
            cv2.rectangle(img, 
                         (max(0, x-border_size), max(0, y-border_size)), 
                         (min(img.shape[1], x+(w-1)+border_size), min(img.shape[0], y+(h-1)+border_size)), 
                         white_rgb, 
                         border_size)
            
            # Fill white to the left and right of the mask
            img[y:y+h, 0:x] = white_rgb  # Fill left side
            img[y:y+h, x+w:img.shape[1]] = white_rgb  # Fill right side
            
            logging.info(f"Detected yellow area - Width: {w}, Height: {h}")

        # Replace yellow pixels with white
        img[yellow_mask > 0] = white_rgb
        
        # Save the modified image
        output_path = img_path.replace('.png', '_yellow_replaced.png')
        cv2.imwrite(output_path, img)
        
        logging.info(f"Finished replace_yellow_with_white for {ticker}")
        return output_path, img
        
    except Exception as e:
        logging.error(f"Error in replace_yellow_with_white for {ticker}: {str(e)}")
        return None, None


def crop_black_area(ticker: str, img_path: str, image):
    logging.info(f"Starting crop_black_area for {ticker}")
    try:
        # Create a copy of the image to avoid modifying original
        img = image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find non-black pixels
        non_black = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
        
        # Find bounding box of non-black pixels
        coords = cv2.findNonZero(non_black)
        x, y, w, h = cv2.boundingRect(coords)
        
        # Crop the image
        cropped = img[y:y+h, x:x+w]
        
        # Save cropped image
        output_path = img_path.replace('.png', '_cropped_vertical.png')
        cv2.imwrite(output_path, cropped)
        
        logging.info(f"Finished crop_black_area for {ticker}")
        return output_path, cropped
        
    except Exception as e:
        logging.error(f"Error in crop_black_area for {ticker}: {str(e)}")
        return None, None


def count_printed_lines(img_path: str, image, vertical_line_color: tuple, covered_line_rgb: tuple):
    
    # Load the image
    img = image
    
    # Find the x-coordinate of the vertical line from right
    height, width = img.shape[:2]
    for x in range(width - 1, -1, -1):
        if np.all(img[:, x] == vertical_line_color):
            break
    
    # Crop the image from the left edge to just before the vertical line
    cropped_img = img[:, x-10:x-5]
    
    # Save the cropped image
    cropped_img_path = img_path.replace('.png', '_for_counting_detected_lines.png')
    cv2.imwrite(cropped_img_path, cropped_img)
    
    # Count groups of covered_line_rgb pixels
    line_groups = 0
    in_line = False
    
    # Scan each row
    for y in range(height):
        # Check if any pixel in this row matches covered_line_rgb
        if any(np.all(cropped_img[y, x] == covered_line_rgb) for x in range(cropped_img.shape[1])):
            if not in_line:
                line_groups += 1
                in_line = True
        else:
            in_line = False
            
    logging.info(f"Found {line_groups} groups of covered lines")
    return line_groups


def delete_all_the_images(image_folder: str, img_endings: list):
    for file in os.listdir(image_folder):
        if not any(file.endswith(ending) for ending in img_endings):
            os.remove(os.path.join(image_folder, file))


def main(img_path: str, img, ticker_name: str, ticker_price: float):

    base_graph_img_path = os.path.dirname(img_path)
    
    # img_path_2, img_2 = crop_graph_area_by_red_line(
    #     ticker=ticker_name,
    #     img_path=img_path_1,
    #     image=img_1,
    #     vertical_line_color=(0, 0, 255)
    # )

    img_path_2, img_2 = crop_graph_area(
        ticker=ticker_name,
        img_path=img_path,
        image=img
    )

    img_path_3, img_3 = crop_black_area(
        ticker=ticker_name,
        img_path=img_path_2,
        image=img_2
    )
        
    img_path_4, img_4 = replace_yellow_with_white(
        ticker=ticker_name,
        img_path=img_path_3,
        image=img_3,
        yellow_rgb=(0, 255, 255),
        white_rgb=(255, 255, 255)
    )
        
    img_path_5, img_5 = paint_red_line_white_space(
        ticker=ticker_name,
        img_path=img_path_4,
        image=img_4,
        rad_rgb=(0, 0, 255)
    )
        
    img_paths_6, imgs_6 = crop_img_using_red_line(
        ticker=ticker_name,
        img_path=img_path_5,
        image=img_5,
        rad_rgb=(0, 0, 255),
        is_vertical_scan=True
    )
        
    color_numbers = []
    for img_path_6, img_6 in zip(img_paths_6, imgs_6):
        number = get_numbers_from_image(img_path_6, img_6)
        # Split the string into separate numbers if there are multiple
        number_list = str(number).strip().split('\n')
            
        for num in number_list:
            if not num.strip():  # Skip empty strings
                continue
            try:
                cleaned_number = float(num.strip().replace(' ', '').replace(',', '.'))
                price_up_bound = ticker_price * 7
                price_down_bound = ticker_price / 7
                if price_down_bound <= cleaned_number <= price_up_bound:
                    color_numbers.append(cleaned_number)
                    logging.info(f"Valid number found in image: {cleaned_number}")
                else:
                    logging.info(f"Number {cleaned_number} is outside valid price range [{price_down_bound}, {price_up_bound}]")
            except ValueError:
                logging.info(f"Invalid number found in image: {num}")
                continue
                
    color_numbers
    
    delete_all_the_images(base_graph_img_path, img_endings=['_chart.png'])
    return color_numbers


# if __name__ == "__main__":

#     ticker_name = 'TC'
#     ticker_price = 0.9099

#     graph_img_path = f'assets/ta_p_channel_images/{ticker_name}_chart.png'
#     covered_line_rgb = (0, 165, 255)
#     found_blue_lines = (37, 111, 149)
#     found_purple_lines = (142, 73, 156)

#     purple, blue = main(graph_img_path, ticker_name, found_blue_lines, found_purple_lines, covered_line_rgb, ticker_price)
#     logging.info(f"Purple: {purple}, Blue: {blue}")
