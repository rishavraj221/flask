# import requests
from bs4 import BeautifulSoup

# # URL of the page to scrape
# url = "https://www.linkedin.com/company/origamiagents/about/"

# # Send an HTTP GET request to fetch the page source
# response = requests.get(url)

# print(f"response : {response}")

# # Check if the request was successful
# if response.status_code == 200:
#     # Parse the HTML content using BeautifulSoup
#     soup = BeautifulSoup(response.text, 'html.parser')

#     # Example: Extracting specific data (e.g., all links)
#     links = soup.find_all('a')  # Find all <a> tags
#     for link in links:
#         print(link.get('href'))  # Print the href attribute of each link
# else:
#     print(f"Failed to fetch the page. Status code: {response.status_code}")

# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# # from webdriver_manager.chrome import ChromeDriverManager

# # Set up the browser driver (ensure you have the Chrome driver installed)
# options = Options()
# # options.add_argument("--headless")  # Run in headless mode
# driver = webdriver.Chrome(service=Service(
#     '/opt/homebrew/bin/chromedriver'), options=options)

# # Open the webpage
# driver.get("https://www.linkedin.com/company/origamiagents/about/")

# # Get the page source after JavaScript loads
# html_source = driver.page_source

# # Parse it with BeautifulSoup
# soup = BeautifulSoup(html_source, 'html.parser')
# print(soup.title.text)

# # Quit the browser
# driver.quit()

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Set up your credentials
linkedin_email = "picalive.help@gmail.com"
linkedin_password = "p1CALIVE.HELP@GMAIL.COM"

# Initialize WebDriver (ensure chromedriver is in PATH or provide its location)
driver = webdriver.Chrome()

try:
    # Open LinkedIn
    driver.get(
        "https://www.linkedin.com/search/results/companies/?keywords=rugs&page=1")
    time.sleep(2)  # Allow the page to load

    # Enter email
    email_input = driver.find_element(By.ID, "username")
    email_input.send_keys(linkedin_email)

    # Enter password
    password_input = driver.find_element(By.ID, "password")
    password_input.send_keys(linkedin_password)

    # Click the login button
    login_button = driver.find_element(By.XPATH, '//button[@type="submit"]')
    login_button.click()
    time.sleep(20)  # Wait for login to complete

    company_links = driver.find_elements(
        By.XPATH, '//a[contains(@href, "/company/")]')

    # Extract the href attributes
    hrefs = [link.get_attribute('href') for link in company_links]

    # Print all hrefs
    for href in hrefs:
        print(href)

    # Iterate through hrefs and open each one in a new tab
    for index, href in enumerate(hrefs, start=1):
        print(f"Opening {index}/{len(hrefs)}: {href}")

        # Open a new tab
        # driver.execute_script("window.open('');")
        # Switch to the new tab
        # driver.switch_to.window(driver.window_handles[-1])

        # Load the company's page
        driver.get(f"{href}about")
        # Wait for the page to load (use explicit waits if necessary)
        time.sleep(3)

        # Extract data from the company page
        try:
            org_module_card = driver.find_element(
                By.XPATH, '//section[contains(@class, "org-top-card")]')
            overview_section = driver.find_element(
                By.XPATH, '//section[contains(@class, "org-about-module__margin-bottom")]')

            # Extract details from the section
            company_name = org_module_card.find_element(
                By.XPATH, './/h1').text
            company_description = overview_section.find_element(
                By.XPATH, './/p[contains(@class, "break-words")]').text

            company_details = overview_section.find_element(
                By.XPATH, './/dl[contains(@class, "overflow-hidden")]')

            company_details_headers = company_details.find_elements(
                By.XPATH, './/h3')
            company_details_value = company_details.find_elements(
                By.XPATH, './/dd')

            print(f"Company Name: {company_name}")
            print(f"Description: {company_description}")
            for header in company_details_headers:
                print(f"Company header : {header.text}")
            for val in company_details_value:
                print(f"Company value : {val.text}")

        except Exception as e:
            print(f"Failed to extract data for {href}: {e}")

    time.sleep(20)  # Wait to load search results

finally:
    # Close the browser
    driver.quit()
