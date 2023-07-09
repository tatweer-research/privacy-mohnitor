from datetime import datetime


def get_current_date():
    current_date = datetime.now().date()
    return current_date.strftime('%Y-%m-%d')


if __name__ == '__main__':
    current_date_string = get_current_date()
    print(current_date_string)
