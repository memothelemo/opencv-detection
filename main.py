import cv2
import traceback

from src.main import main


if __name__ == "__main__":
    try: main()
    except Exception as ex:
        traceback.print_exception(ex)
        exit_code = 1
    finally:
        # Close down all windows if there's something wrong with the
        # program or the developer forget to close all windows.
        cv2.destroyAllWindows()
