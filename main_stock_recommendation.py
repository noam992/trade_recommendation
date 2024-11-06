import logging
import pandas as pd
from finviz_pattern_list import main as get_finviz_pattern_list
from finviz_capture_graph import main as get_finviz_capture_graph
from finviz_line_values import main as get_finviz_line_values, detect_straight_line_by_color
from measure_calculations import channel_range, ratio_of_current_price_to_channel_range


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Parameters
screener_url = 'https://finviz.com/screener.ashx?v=110&s='
pattern = 'ta_p_channel'
page_size = 20
objects = 0
folder = 'assets'
filename = f'{folder}/stocks_pattern_{pattern}.csv'
covered_line_rgb = (0, 165, 255)
line_colors = [
    ((37, 111, 149), 'support_rgb'), # blue
    ((142, 73, 156), 'resistance_rgb') # purple
]


def read_stocks_from_csv(filename: str) -> pd.DataFrame:
    try:
        stocks_df = pd.read_csv(filename)
        logging.info(f"Successfully read data from {filename}")
        return stocks_df
    except Exception as e:
        logging.error(f"Error reading CSV file {filename}: {str(e)}")
        return pd.DataFrame()


def save_to_csv(df, path_to_save):
    logging.info("Saving data to CSV")
    df.to_csv(path_to_save, index=False)
    logging.info(f"Data saved to {filename}")


def main():

    stocks_df = get_finviz_pattern_list(screener_url, pattern, page_size, objects, filename)
    # stocks_df = read_stocks_from_csv(filename)
    
    # Initialize new columns with NaN values
    stocks_df['average_resistance'] = float('nan')
    stocks_df['average_support'] = float('nan')
    stocks_df['channel_range'] = float('nan')
    stocks_df['current_price_ratio_channel'] = float('nan')

    for index, row in stocks_df.iterrows():
        logger.info(f"# Processing stock: {row['Ticker']}")

        ticker_name  = row['Ticker']
        ticker_price = float(row['Price'])

        graph_img_path = get_finviz_capture_graph(pattern, folder, ticker_name)
        # graph_img_path = f'assets/ta_p_channel_images/{ticker_name}_chart.png'
    
        results = {'resistance_rgb': [], 'support_rgb': []}
        for color_rgb, color_name in line_colors:

            img_path, img = detect_straight_line_by_color(
                ticker=ticker_name,
                img_path=graph_img_path,
                color_rgb=color_rgb,
                line_frame=60,
                covered_line_rgb=covered_line_rgb,
                detect_minLineLength=50
            )

            if img_path == [] and img == []:
                logging.info(f"No valid image found for {ticker_name} with {color_name} color - continuing to next iteration")
                continue

            color_result = get_finviz_line_values(img_path, img, ticker_name, ticker_price)

            results[color_name] = color_result

        if len(results['resistance_rgb']) > 0 and len(results['support_rgb']) > 0:
            logger.info(f"Resistance: {results['resistance_rgb']}, Dupport: {results['support_rgb']}")
            average_resistance = sum(results['resistance_rgb']) / len(results['resistance_rgb'])
            average_support = sum(results['support_rgb']) / len(results['support_rgb'])

            if average_resistance >= average_support:
                logger.info(f"Average Resistance: {average_resistance}, Average Support: {average_support}")
                stocks_df.at[index, 'average_resistance'] = average_resistance
                stocks_df.at[index, 'average_support'] = average_support
                
                # Calculate and store channel range and price ratio
                try:
                    channel_range_value = channel_range(average_support, average_resistance)
                    price_ratio = ratio_of_current_price_to_channel_range(ticker_price, average_support, average_resistance)
                    
                    stocks_df.at[index, 'channel_range'] = channel_range_value
                    stocks_df.at[index, 'current_price_ratio_channel'] = price_ratio
                    logger.info(f"Channel Range: {channel_range_value}, Price Ratio: {price_ratio}")
                except ValueError as e:
                    logger.error(f"Error calculating metrics for {ticker_name}: {str(e)}")
    
            plus_avg_line_path = filename.replace('.csv', '_with_avg.csv')
            save_to_csv(stocks_df, plus_avg_line_path)


if __name__ == "__main__":
    main()