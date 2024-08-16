from preprocess import get_preprocessor
def main():
    preprocessor = get_preprocessor()
    overwrite=   True
    preprocessor.store_data(overwrite)
if __name__ == '__main__':
    main()