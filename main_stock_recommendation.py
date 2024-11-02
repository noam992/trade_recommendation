import logging
import pandas as pd
from finviz_pattern_list import main as get_finviz_pattern_list
from finviz_capture_graph import main as get_finviz_capture_graph
from finviz_line_values import main as get_finviz_line_values


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Parameters
screener_url = 'https://finviz.com/screener.ashx?v=110&s='
pattern = 'ta_p_channel'
total_pages = 3
page_size = 20
objects = 0
folder = 'assets'
filename = f'{folder}/stocks_pattern_{pattern}.csv'
covered_line_rgb = (0, 165, 255)
found_blue_lines = (37, 111, 149)
found_purple_lines = (142, 73, 156)


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

    # stocks_df = get_finviz_pattern_list(screener_url, pattern, total_pages, page_size, objects, filename)
    stocks_df = read_stocks_from_csv(filename)
    
    # Initialize new columns with NaN values
    stocks_df['average_purple'] = float('nan')
    stocks_df['average_blue'] = float('nan')

    for index, row in stocks_df.iterrows():
        logger.info(f"# Processing stock: {row['Ticker']}")

        ticker_name  = row['Ticker']
        ticker_price = float(row['Price'])

        graph_img_path = get_finviz_capture_graph(pattern, folder, ticker_name)

        purple_result, blue_result = get_finviz_line_values(graph_img_path, ticker_name, found_blue_lines, found_purple_lines, covered_line_rgb, ticker_price)
                                                        
        if len(purple_result) > 0 and len(blue_result) > 0:
            logger.info(f"Purple: {purple_result}, Blue: {blue_result}")
            average_purple = sum(purple_result) / len(purple_result)
            average_blue = sum(blue_result) / len(blue_result)

            logger.info(f"Average Purple: {average_purple}, Average Blue: {average_blue}")
            stocks_df.at[index, 'average_purple'] = average_purple
            stocks_df.at[index, 'average_blue'] = average_blue
    
            plus_avg_line_path = filename.replace('.csv', '_with_avg.csv')
            save_to_csv(stocks_df, plus_avg_line_path)


if __name__ == "__main__":
    main()