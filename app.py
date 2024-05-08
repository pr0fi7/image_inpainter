import streamlit as st
import numpy as np
from PIL import Image
import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
import mediapipe
import pandas as pd

pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,

)
pipeline = pipeline.to("cuda")



def transform_image(init_image, mask_image, text):
    prompt = text + ", 8k resolution, detailed, beatiful eyes"
    negative_prompt = 'ugly, distorted, disformed, change face, problem with fingers'
    init_image = Image.open(init_image).convert("RGB")
    mask_image = Image.open(mask_image).convert("RGB")
    image = pipeline(prompt=prompt, negative_prompt = negative_prompt,image=init_image, mask_image=mask_image, height=1024, width=1024).images[0]
    image_np = np.array(image)
    image_np = image_np.astype(np.uint8)

    return image_np



def resize_image(image):
    # Load the input image
    input_image = cv2.imread(image)

    # Get the dimensions of the input image
    original_height, original_width = input_image.shape[:2]

    # Determine the maximum dimension for resizing
    max_dim = max(original_height, original_width)

    # Calculate the scaling factor to resize the image while preserving the aspect ratio
    scale_factor = 1024 / max_dim

    # Resize the image using the scaling factor
    resized_image = cv2.resize(input_image, (0, 0), fx=scale_factor, fy=scale_factor)

    # Get the new dimensions of the resized image
    resized_height, resized_width = resized_image.shape[:2]

    # Calculate the padding needed to make the image 1024x1024
    top_pad = (1024 - resized_height) // 2
    bottom_pad = 1024 - resized_height - top_pad
    left_pad = (1024 - resized_width) // 2
    right_pad = 1024 - resized_width - left_pad

    # Pad the image with black spaces
    resized_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return resized_image

def create_mask(resized_image):


    mp_face_mesh = mediapipe.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    results = face_mesh.process(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    landmarks = results.multi_face_landmarks[0]


    face_oval = mp_face_mesh.FACEMESH_FACE_OVAL


    df = pd.DataFrame(list(face_oval), columns = ["p1", "p2"])
    routes_idx = []

    p1 = df.iloc[0]["p1"]
    p2 = df.iloc[0]["p2"]

    for i in range(0, df.shape[0]):

        #print(p1, p2)

        obj = df[df["p1"] == p2]
        p1 = obj["p1"].values[0]
        p2 = obj["p2"].values[0]

        route_idx = []
        route_idx.append(p1)
        route_idx.append(p2)
        routes_idx.append(route_idx)

    # -------------------------------

    # for route_idx in routes_idx:
    #     print(f"Draw a line between {route_idx[0]}th landmark point to {route_idx[1]}th landmark point")
    routes = []

    for source_idx, target_idx in routes_idx:

        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]

        relative_source = (int(resized_image.shape[1] * source.x), int(resized_image.shape[0] * source.y))
        relative_target = (int(resized_image.shape[1] * target.x), int(resized_image.shape[0] * target.y))

        #cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness = 2)

        routes.append(relative_source)
        routes.append(relative_target)
    mask = np.zeros((resized_image.shape[0], resized_image.shape[1]))
    mask = cv2.fillConvexPoly(mask, np.array(routes), 255)

    inverted_mask = 255 - mask

    return inverted_mask


img_file_buffer = st.camera_input('take a picture')
text = st.text_input('Enter your prompt')

button = st.button('Upload', on_click=st.write('Uploaded'))


if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    image.save("yuha1.png")
    image = cv2.imread("yuha1.png")
    img_array = np.array(image)
    img_array_fin = img_array.astype(np.uint8)

    cv2.imwrite('yuha2.png', img_array_fin)

    # # Send request to the API endpoint
    # url = base_url + "upload/"
    # files = {"file": open("yuha2.png", "rb")}

    if button:
        resized_image = resize_image("yuha2.png")
        mask = create_mask(resized_image)
        cv2.imwrite("resized_image.png", resized_image)
        cv2.imwrite("mask_image.png", mask)

        transformed_image = transform_image('resized_image.png', 'mask_image.png', text)

            # Reshape the array to its original dimensions
        original_height, original_width, channels = 1024, 1024, 3
        image_array = transformed_image.reshape(original_height, original_width, channels)
        st.image(image_array)
