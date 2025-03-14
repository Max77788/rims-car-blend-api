import base64

# Read the JSON file
with open('car-wheels-project-key.json', 'rb') as file:
    json_data = file.read()

# Encode the JSON data to Base64
base64_encoded = base64.b64encode(json_data).decode('utf-8')

print(base64_encoded)
