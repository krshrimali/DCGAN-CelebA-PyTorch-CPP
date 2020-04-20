import sys, os, shutil, time

def isImage(filename):
    """
    Returns True if filename is an image extension, else False
    Supported extensions: JPG, JPEG, PNG
    """
    return True in [True if extension in filename else False for extension in [".jpg", ".jpeg", ".png"]]

def get_files(folder_path = None):
    """
    This function returns list of files in the given folder path

    Arguments
    =========
    1. folder_path, defualt = None, this is path of folder, type: str

    Returns
    =========
    List of files in the given folder (but with absolute path), type: list of str
    """

    if folder_path is None:
        print("Given path is None")
        print("Exiting")
        sys.exit(0)

    list_files = [folder_path + "/" + filename if folder_path[-1] == "/" and isImage(filename) else folder_path + filename if isImage(filename) else -1 for filename in os.listdir(folder_path)]
    list_files = [x for x in list_files if x != -1]
    return list_files

def split_dataset(list_files, train = None, test = None):
    """
    Splits dataset into train % and test %. No default values. User should pass them as arguments.

    Arguments
    =========
    1. list_files: list of files in the folder (prefer to be absolue paths), type: list of strings
    2. train: train percent for splitting, example: 90 if you want to split 90% of files into train/ folder, type: int
    3. test: test percent for splitting, example: 10 if you want to split 10% of files into test/ folder, type: int

    Note: pass either train or test, and the other will be calculated automatically.

    Returns
    ========
    None
    """

    if train is None and test is not None:
        train_files = list_files[:int(len(list_files) * (100 - test)/100)]
        test_files  = list_files[int(len(list_files) * (100 - test)/100):]
    elif test is None and train is not None:
        train_files = list_files[:int(len(list_files) * 90/100)]
        test_files  = list_files[int(len(list_files) * 90/100):]
    else:
        print("Pass train or test argument to this function")
        print("Exiting...")
        sys.exit(0) 
    
    train_folder = "data/train/"
    test_folder = "data/test/"

    if not os.path.isdir(train_folder):
        if not os.path.isdir("data/"):
            os.mkdir("data/")
        os.mkdir("data/train/")
    if not os.path.isdir(test_folder):
        if not os.path.isdir("data/"):
            os.mkdir("data/")
        os.mkdir("data/test/")

    print("Copying {} files to train/ folder".format(len(train_files)))

    start = time.time()
    for index, file_source_path in enumerate(train_files):
        file_destination_path = train_folder + file_source_path.split("/")[-1]
        shutil.move(file_source_path, file_destination_path)
        if index % int(len(train_files)/10) == 0:
            print("Copied {} files so far...".format(index + 1))
    end = time.time()

    print("Time taken to move files into {} folder: {:.4f} seconds".format(train_folder, end-start))
    print("Copying {} files to test/ folder".format(len(test_files)))

    start = time.time()
    for index, file_source_path in enumerate(test_files):
        file_destination_path = "data/test/" + file_source_path.split("/")[-1]
        shutil.move(file_source_path, file_destination_path)
        if index % int(len(test_files)/10) == 0:
            print("Copied {} files so far...".format(index + 1))
    end = time.time()

    print("Time taken to move files into {} folder: {:.4f} seconds".format(test_folder, end-start))

    print("Copying done")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = "dcgan/"

    list_files = get_files(folder_path)
    print("Total files: ", len(list_files))
    split_dataset(list_files, train = 90)
