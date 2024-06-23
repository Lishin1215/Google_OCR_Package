
# Imports the Google Cloud client library
from google.cloud import vision
import io


def run_quickstart() -> vision.EntityAnnotation:
    """Provides a quick start example for Cloud Vision."""

    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # The path of the image file to annotate
    file_path = "assests/Referral_letter_example.jpg"

    # The URI of the image file to annotate
    # file_uri = "gs://cloud-samples-data/vision/label/wakeupcat.jpg"

    # Loads the image into memory
    with io.open(file_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    # image.source.image_uri = file_uri

    # Performs label detection on the image file
    response = client.text_detection(image=image)
    texts = response.text_annotations

    print("Texts:")
    # for text in texts:
        # print(text.description)
        # print("=============================")
    print(texts[0].description)

    return texts[0].description

if __name__ == "__main__":
    run_quickstart()