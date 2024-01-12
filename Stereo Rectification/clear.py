import os

def clear_files():
    png_files = [file for file in os.listdir() if file.endswith('.png')]
    mat_files = [file for file in os.listdir() if file.endswith('.mat')]

    for file in png_files:
        print("Deleted : " + file)
        os.remove(file)

    for file in mat_files:
        print("Deleted : " + file)
        os.remove(file)

clear_files()