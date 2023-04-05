from google.cloud import storage
import os

def upload_csv_to_gcs(dataframe, destination_folder):
    """Uploads a CSV file to a specified Google Cloud Storage bucket in a specified folder.

    Args:
        csv_file_path (str): The local path to the CSV file to upload.
        folder_name (str): The name of the folder in the bucket to upload the file to.
    """
    # Instantiate a client for interacting with the Google Cloud Storage API

    from google.oauth2 import service_account

    # Replace [PATH_TO_PRIVATE_KEY_FILE] with the path to your private key file.
    credentials = service_account.Credentials.from_service_account_file('pymaggie-8cf90f8de8ce.json')

    client = storage.Client(credentials=credentials)

    # Get a reference to the specified bucket
    bucket = client.get_bucket("pymaggie_csv_storage")

    # Get a reference to the specified folder within the bucket
    folder = bucket.blob(str(session['session_id']))

    #changing into csv
    csv_file_path = str(session['session_id'])+"_uploaded"+'.csv'
    dataframe.to_csv(csv_file_path, index=False)

    # Get the name of the CSV file
    csv_file_name = os.path.basename(csv_file_path)

    # Uploads the CSV file to the destination folder
    blob = bucket.blob(str(session['session_id']) + "/" + csv_file_name)
    blob.upload_from_filename(csv_file_path)

    print(f"File {csv_file_name} uploaded to gs://pymaggie_csv_storage/{session['session_id']}/{csv_file_name}")


upload_csv_to_gcs("credit_record.csv",'pymaggie_csv_storage',"test")