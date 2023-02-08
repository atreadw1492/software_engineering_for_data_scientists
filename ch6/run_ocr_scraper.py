

import cv2
import pytesseract
import pathlib
from tqdm import tqdm
from multiprocessing import Pool
import time

#desktop = pathlib.Path("/Users/Amily/Downloads/English/Img")

desktop = pathlib.Path("/Users/Amily/Downloads/large-receipt-image-dataset-SRD")


files = list(desktop.rglob("*"))

files = [str(file) for file in files if ".jpg" in file.name]


"""all_text = []
failures = []
for file in tqdm(files):

    try:
        image = cv2.imread(file)        
        all_text[file] = pytesseract.image_to_string(image)

    except Exception:
        failures.append(file)
    #print(text)"""
    
    

def scrape_text(file):

    image = cv2.imread(file)        

    return pytesseract.image_to_string(image)


start = time.time()
all_text = [scrape_text(file) for file in files]
end = time.time()
print(end - start)
    #print(text)


if __name__ == "__main__":

    start = time.time()
    cores_pool = Pool(3)
    cores_pool.map(scrape_text, files)
    end = time.time()
    print(end - start)






    
