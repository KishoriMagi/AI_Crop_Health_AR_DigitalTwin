import dropbox
import os
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

DROPBOX_ACCESS_TOKEN = os.getenv('DROPBOX_ACCESS_TOKEN')

def get_dropbox_client():
    if not DROPBOX_ACCESS_TOKEN or DROPBOX_ACCESS_TOKEN == 'your_dropbox_access_token_here':
        print("Warning: Dropbox access token is not set properly.")
        return None
    return dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

def upload_test_request(test_id, data_dict):
    """
    Creates a CSV from the data_dict and uploads it to Dropbox /Requests/
    """
    dbx = get_dropbox_client()
    if not dbx:
        return False
        
    try:
        # Convert dict to DataFrame then to CSV in memory
        df = pd.DataFrame([data_dict])
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        
        file_path = f"/Requests/test_request_{test_id}.csv"
        
        dbx.files_upload(csv_bytes, file_path, mode=dropbox.files.WriteMode.overwrite)
        print(f"Successfully uploaded {file_path}")
        return file_path
    except Exception as e:
        print(f"Error uploading to Dropbox: {e}")
        return None

def fetch_and_process_results():
    """
    Lists files in /Results/, downloads them, reads as CSV, and returns a list of dictionaries.
    Also moves processed files to an Archive folder or deletes them.
    """
    dbx = get_dropbox_client()
    if not dbx:
        return []

    results = []
    try:
        # Check if Results folder exists, if not, create it or just list
        try:
            res = dbx.files_list_folder("/Results")
        except dropbox.exceptions.ApiError as e:
            if e.error.is_path() and e.error.get_path().is_not_found():
                return [] # Folder doesn't exist yet
            raise

        for entry in res.entries:
            if isinstance(entry, dropbox.files.FileMetadata) and entry.name.endswith('.csv'):
                # Download the file
                metadata, response = dbx.files_download(entry.path_display)
                
                # Parse CSV
                csv_content = BytesIO(response.content)
                df = pd.read_csv(csv_content)
                
                # Convert first row to dict (assuming 1 result per file)
                if not df.empty:
                    result_dict = df.iloc[0].to_dict()
                    results.append({
                        'file_path': entry.path_display,
                        'data': result_dict
                    })
                
                # Optionally delete or move the file after processing so we don't process it again
                dbx.files_delete_v2(entry.path_display)
                print(f"Processed and deleted {entry.path_display}")
                
    except Exception as e:
        print(f"Error fetching results from Dropbox: {e}")

    return results

