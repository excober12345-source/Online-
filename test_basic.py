# test_basic.py
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# Test ChromeDriver installation
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Test if browser opens
driver.get("https://www.google.com")
print("Browser opened successfully!")
input("Press Enter to close browser...")
driver.quit()
