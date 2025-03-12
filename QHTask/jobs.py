import schedule
from time import sleep
import logging
from update_fut_pos import update_positions
from QUANTAXIS.QASU.save_tdx import QA_SU_save_future_day

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # 每天下午4点更新期货交易数据
    schedule.every().day.at('16:30').do(QA_SU_save_future_day)
    schedule.every().day.at('17:50').do(update_positions)

    # 其他任务...

    while True:
        schedule.run_pending()
        sleep(3)

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")

finally:
    logging.info("Exiting the script.")
