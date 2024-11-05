import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def channel_range(lower_bound: float, upper_bound: float):
    if lower_bound > upper_bound:
        raise ValueError("Lower bound is greater than upper bound")
    return upper_bound - lower_bound


def ratio_of_current_price_to_channel_range(current_price: float, lower_bound: float, upper_bound: float):
    try:
        channel = channel_range(lower_bound, upper_bound)
        if channel == 0:
            logging.warning(f"Channel range is 0 for price {current_price}, bounds {lower_bound}-{upper_bound}")
            return 0
        return (current_price - lower_bound) / channel
    except Exception as e:
        logging.error(f"Error calculating price ratio: {str(e)}")
        return 0
