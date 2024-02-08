import requests
from bs4 import BeautifulSoup
import json
import csv


# Extract and print doctor details
def extract_data_from_page(soup):
    doctor_elements = soup.find_all('div', class_='doctor-list')
    for doctor in doctor_elements:
        doctor_name = doctor.find('div', class_='doctor-dea1').find('h6').text.strip()
        specialty = doctor.find('div', class_='doctor-dea1').find('p', class_='speci').text.strip()
        department = doctor.find('div', class_='doctor-dea1').find('p', class_='department').text.strip()
        location = doctor.find('div', class_='doctor-dea1').find('p', class_='location-h').text.strip()

        print(f"Doctor Name: {doctor_name}, Specialty: {specialty}, Department: {department}, Location: {location}")
        doctor_data.append({
            'Doctor Name': doctor_name,
            'Specialty': specialty,
            'Department': department,
            'Location': location
        })

# hospital website
url = 'https://www.kimshealth.org/trivandrum/doctors/'
current_page = 1
doctor_data = []
while True:
    # Construct the URL for the current page
    current_url = f'{url}?page={current_page}'
    response = requests.get(current_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        extract_data_from_page(soup)
        load_more_link = soup.find('a', class_='endless_more')
        if load_more_link:
            current_page += 1
        else:
            break
    else:
        print(f"Failed to retrieve the webpage. Status Code: {response.status_code}")
        break

# JSON file
with open('doctors_data.json', 'w', encoding='utf-8') as json_file:
    json.dump(doctor_data, json_file, ensure_ascii=False, indent=4)

# CSV file
csv_header = ['Doctor Name', 'Specialty', 'Department', 'Location']
with open('doctors_data.csv', 'w', encoding='utf-8', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_header)
    for doctor in doctor_data:
        csv_writer.writerow([doctor['Doctor Name'], doctor['Specialty'], doctor['Department'], doctor['Location']])
