# import time
# from selenium import webdriver
#
# driver = webdriver.Chrome('/path/to/chromedriver')  # Optional argument, if not specified will search path.
# driver.get('http://www.google.com/');
# time.sleep(5) # Let the user actually see something!
# search_box = driver.find_element_by_name('q')
# search_box.send_keys('ChromeDriver')
# search_box.submit()
# time.sleep(5) # Let the user actually see something!
# driver.quit()
import pandas as pd

def NNStatFormatter():
    data = pd.read_csv("data/NNModelKeyStats.csv")

    data = data.T
    data["Loss"] = data[0]
    data["Theoretical Accuracy"] = data[1]
    data = data.drop(0, axis=1)
    data = data.drop(1, axis=1)
    data = data.drop("Unnamed: 0", axis=0)
    data.to_csv("Data/NNmodelKeyStats.csv")






