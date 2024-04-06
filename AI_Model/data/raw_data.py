import warnings
import requests

warnings.filterwarnings("ignore")


# Function for downloading different files from gdrive
# Downloading datasets from gdrive
def get_direct_download_link(gdrive_link):
    file_id = gdrive_link.split("/")[-2]
    direct_download_link = f"https://drive.google.com/uc?export=download&id={file_id}"
    return direct_download_link

def download_file_from_link(download_link, save_to_path):
    response = requests.get(download_link)
    if response.status_code == 200:
        with open(save_to_path, 'wb') as f:
            f.write(response.content)
    else:
        print("Failed to download file. Status code:", response.status_code)

