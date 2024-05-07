import streamlit as st
import numpy as np
from PIL import Image
import requests
import cv2
import numpy as np
from pydantic import BaseModel
import json

class Item(BaseModel):
    text: str

base_url = "https://f290-34-16-157-66.ngrok-free.app/"

img_file_buffer = st.camera_input('take a picture')
prompt = st.text_input('Enter your prompt')

item = Item(text=prompt)
button = st.button('Upload', on_click=st.write('Uploaded'))


if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    image.save("yuha1.png")
    image = cv2.imread("yuha1.png")
    img_array = np.array(image)
    img_array_fin = img_array.astype(np.uint8)

    cv2.imwrite('yuha2.png', img_array_fin)

    # Send request to the API endpoint
    url = base_url + "upload/"
    files = {"file": open("yuha2.png", "rb")}

    if button:
        response = requests.post(url, files=files)

        if response.status_code == 200:

            # Send request to the process endpoint with text data
            process_url = base_url + "process/"
            response = requests.post(process_url, json=item.dict())

        # Check if the request was successful
        if response.status_code == 200:
            # Decode the imaage bytes from the response content
            image_bytes = response.content
            list_img = json.loads(image_bytes)

        # Convert the list back to a NumPy array
            image_array = np.array(list_img, dtype=np.uint8)

            # Reshape the array to its original dimensions
            original_height, original_width, channels = 1024, 1024, 3
            image_array = image_array.reshape(original_height, original_width, channels)
            st.image(image_array)

        else:
            print("Failed to process image. Status code:", response.status_code)

