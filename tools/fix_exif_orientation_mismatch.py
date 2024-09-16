from PIL import Image
import json
import os

"""
EXIF Orientation Tag (274)

The orientation tag indicates how the image should be displayed. It helps
software understand how to properly orient the image if it's been rotated or
flipped. The value for the orientation tag can be one of several integers, each
representing a different orientation:

    * 1: Normal (0 degrees)
    * 2: Flipped horizontally
    * 3: Upside down (180 degrees)
    * 4: Flipped vertically
    * 5: Rotated 90 degrees clockwise and flipped horizontally
    * 6: Rotated 90 degrees clockwise
    * 7: Rotated 90 degrees clockwise and flipped vertically
    * 8: Rotated 90 degrees counterclockwise
"""

# Change these paths accordingly
directory_path = '../images/'
json_file_path = '../test.json'

filenames = set()

for root, _, files in os.walk(directory_path):
    for img in files:
        img_path = os.path.join(root, img)
        with Image.open(img_path) as im:
            try:
                exif = im._getexif()
                if exif is not None:
                    orientation = exif.get(274)  # 274 is the EXIF tag for orientation
                    if orientation is not None:
                        if orientation in range(5, 9):
                            filenames.add(img)
            except Exception as e:
                print(f"{img_path}: Error {e}")


# Load the JSON data
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Update JSON data
for item in data['images']:
    current_file = item['file_name'].split('/')[-1]
    if current_file in filenames:
        # Swap width and height values
        print(f"SWAP FOR {'-'*10} {current_file}")
        item['width'], item['height'] = item['height'], item['width']

# Save the updated JSON data back to the file
with open(json_file_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print()
print(f"{len(filenames)} annotations updated successfully.")

choice = input("Do you want to save the filenames? [N/y] ")

if choice in 'yY':
    with open('filenames.txt', 'w') as f:
        for file in filenames:
            f.write(file + '\n')

