import requests

url = 'https://3deb-2409-40c0-78-62da-15b0-7a85-a1a6-b299.ngrok-free.app/grade'
files = {
    'zip_path': open('/Users/jastegsingh/PycharmProjects/Graded/Archive 2.zip', 'rb'),
    'correct_answer_pdf': open('/Users/jastegsingh/PycharmProjects/Graded/Correct Answer (1).pdf', 'rb')
}
response = requests.post(url, files=files)

if response.status_code == 200:
    with open('grading_results.csv', 'wb') as f:
        f.write(response.content)
    print('File downloaded successfully.')
else:
    print(f'Error: {response.status_code}')
    print(response.text)
import sys
print(sys.version)