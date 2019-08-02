# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:04:46 2019

@author: Raman
"""

from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

driver = webdriver.Chrome()
driver.get("https://www.youtube.com/results?search_query=Travel&sp=EgIQAQ%253D%253D")

"""
for i in range(5):
    time.sleep(1)
    driver.find_element_by_tag_name('body').send_keys(Keys.END)
"""    
user_data = driver.find_elements_by_xpath('//*[@id="video-title"]')
links = []
for i in user_data:
            links.append(i.get_attribute('href'))

print(len(links))

df = pd.DataFrame(columns = ['video-id', 'title', 'description', 'category','location'])
wait = WebDriverWait(driver, len(links))
v_category = "CATEGORY_NAME"
for x in links:
            driver.get(x)
            v_id = x.strip('https://www.youtube.com/watch?v=')
            v_title = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"div#container h1 yt-formatted-string"))).text
            v_location = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "yt-formatted-string a"))).text
            v_description =  wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"div#description yt-formatted-string"))).text
            df.loc[len(df)] = [v_id, v_title, v_description, v_category, v_location]