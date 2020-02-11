# this code with fix the test image directory, placing each image in a directory with it's class
# this is in line with the format expected by ImageDataGenerator
import os
import sys
import shutil

def main():
    directory_str = r"C:\Users\dsingal\Downloads\asl-alphabet\asl_alphabet_test\asl_alphabet_test"
    directory = os.fsencode(directory_str)
    os.chdir(directory)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            # create directory name
            raw_name = filename.split("_")[0]
            try:
                os.mkdir(raw_name)
            except Exception as e:
                pass
            shutil.move(filename, (raw_name + "/" + filename))
        else:
            continue


if __name__ == "__main__":
    main()
