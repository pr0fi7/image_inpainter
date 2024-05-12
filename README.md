# Background Inpainting App

Welcome to the Background Inpainting App repository!

This repository contains an app that changes the background of an image according to your prompt while keeping the face unchanged. The app is built using Streamlit and utilizes stable_diffusion for processing. You can use different servers for deployment depending on your preference and available resources. You can try the app on your own [here](https://huggingface.co/spaces/pr0fi7/face_inpainter). It takes some time to build it, so be patient;). Don't forget that it is as of now strictly inpainter so it can produce an image only after taking a picture. 

## Files

- `app.py`: Streamlit app file for use with Hugging Face Spaces.
- `streamlit_app.py`: Streamlit app file for use with Google Colab.
- `image_inpainting_api.ipynb`: Notebook containing the process to deploy the app via ngrok for Google Colab usage.
- `requirements.txt`: File specifying the Python dependencies required to run the app.
- `.gitignore`: File specifying which files and directories to ignore in Git.
- `media`: Directory with very first image results, to compare after some improvements.

## Challenges Faced

Initially, the plan was to use the DALL-E 2 model for image processing. However, its inpainting capabilities are limited. Therefore, the model was switched to Stable Diffusion. Due to the lack of GPU resources, I've migrated to Google Colab for computation initially, but later, the app was adapted to use Hugging Face Spaces.

## Usage

### Using Hugging Face Spaces

1. Create an account on Hugging Face.
2. Switch to GPU runtime.
3. Copy `app.py` and `requirements.txt` to your Hugging Face Spaces.
4. Hugging Face will build the app for you.

### Using Google Colab

1. Create a Pro version Google Colab account.
2. Create an authorication key for ngrok(for explanation go to the dedicated cell in `image_inpainting_api.ipynb`)
3. Deploy the app via ngrok.
4. Access the app through the generated ngrok URL.
5. Don't forget to change the url to which you send request as it is always different after each deployment. You can change it in `streamlit_app.py`.

## Ongoing Project

This project is ongoing, and future improvements include:
- Connecting Hugging Face with GitHub for seamless integration.
- Adding the possibility to choose the size of the desired image during inpainting.
- Improving prompts for better performance.

## Contact

If you have any questions or suggestions, feel free to reach out to me on [LinkedIn](https://www.linkedin.com/in/mark-shevchenko-218149259/). 

Thank you for visiting this repository! 🚀

![gif_BwAv2b3e_1704783363157_raw (1)](https://github.com/pr0fi7/image_inpainter/assets/53155116/27147161-f536-43ff-a843-ae7e3843ad21)


