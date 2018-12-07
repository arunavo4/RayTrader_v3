from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

path_to_chromedriver = "G:\\AI Trading\\Code\\RayTrader_v3\\automation\\chromedriver.exe"

chrome_options = Options()
chrome_options.add_argument("--disable-infobars")
browser = webdriver.Chrome(executable_path = path_to_chromedriver,chrome_options=chrome_options)

browser.get("https://kite.zerodha.com/marketwatch")

# Login to kite
e_userId = browser.find_element_by_xpath('//*[@id="container"]/div/div/div/form/div[2]/input')
e_pass = browser.find_element_by_xpath('//*[@id="container"]/div/div/div/form/div[3]/input')

e_userId.click()
e_userId.send_keys("XD9631")

e_pass.click()
e_pass.send_keys("#3rdapr1972")

e_pass.send_keys(Keys.ENTER)


#exit browser
#browser.quit()