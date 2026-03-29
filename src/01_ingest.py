# src/01_ingest.py

import os
import json
import logging
from datetime import datetime

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
RAW_DATA_DIR = "data/raw"
SUPPORTED_EXTENSIONS = (".txt", ".csv", ".zip")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Metadata output file
LOCAL_METADATA_FILE = "data_metadata.json"
S3_METADATA_KEY = "metadata/data_metadata.json"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def validate_environment() -> None:
    """Validate required environment variables."""
    required_vars = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "S3_BUCKET_NAME"
    ]

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


def validate_raw_data_dir() -> None:
    """Validate the local raw data directory exists."""
    if not os.path.exists(RAW_DATA_DIR):
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

    if not os.path.isdir(RAW_DATA_DIR):
        raise NotADirectoryError(f"Path is not a directory: {RAW_DATA_DIR}")


def get_s3_client():
    """Create and return an S3 client."""
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=AWS_REGION
    )


def ensure_bucket_exists(s3_client, bucket_name: str, region: str) -> None:
    """
    Check whether the S3 bucket exists.
    If not, create it automatically.
    """
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logging.info("S3 bucket '%s' already exists and is accessible.", bucket_name)

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")

        logging.warning(
            "Bucket '%s' not accessible or does not exist. Attempting to create it.",
            bucket_name
        )

        try:
            if region == "us-east-1":
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": region}
                )

            logging.info("Created S3 bucket '%s' in region '%s'.", bucket_name, region)

        except ClientError as create_error:
            raise RuntimeError(
                f"Failed to create bucket '{bucket_name}'. Original error: {create_error}"
            ) from create_error


def list_supported_files(directory: str):
    """Return a list of supported files in the raw data directory."""
    files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            files.append(filename)
    return files


def upload_file_to_s3(s3_client, local_path: str, bucket_name: str, s3_key: str) -> None:
    """Upload a single file to S3."""
    s3_client.upload_file(local_path, bucket_name, s3_key)
    logging.info("Uploaded '%s' to 's3://%s/%s'", local_path, bucket_name, s3_key)


def generate_metadata(uploaded_files: list) -> dict:
    """Generate dataset ingestion metadata."""
    metadata = {
        "dataset_name": "NASA CMAPSS FD003",
        "source": "NASA Prognostics Data Repository",
        "collection_methodology": {
            "source_download": "Dataset manually downloaded from NASA public repository",
            "local_storage": f"Stored locally in '{RAW_DATA_DIR}'",
            "cloud_upload": "Uploaded to AWS S3 raw data zone using boto3",
            "future_processing": (
                "Processed structured outputs can later be written to AWS RDS PostgreSQL "
                "for downstream analytics, modeling, and dashboard access"
            )
        },
        "cloud_architecture": {
            "raw_storage": f"s3://{BUCKET_NAME}/raw/",
            "metadata_storage": f"s3://{BUCKET_NAME}/{S3_METADATA_KEY}",
            "compute": "AWS EC2 for ETL and preprocessing",
            "database": "AWS RDS PostgreSQL for processed data"
        },
        "schema": {
            "columns": [
                "unit_number",
                "cycle",
                "operational_setting_1",
                "operational_setting_2",
                "operational_setting_3",
                "sensor_1",
                "sensor_2",
                "sensor_3",
                "sensor_4",
                "sensor_5",
                "sensor_6",
                "sensor_7",
                "sensor_8",
                "sensor_9",
                "sensor_10",
                "sensor_11",
                "sensor_12",
                "sensor_13",
                "sensor_14",
                "sensor_15",
                "sensor_16",
                "sensor_17",
                "sensor_18",
                "sensor_19",
                "sensor_20",
                "sensor_21"
            ],
            "target_variable_note": (
                "Remaining Useful Life (RUL) is derived later during preprocessing, "
                "not stored directly in the raw FD003 training file"
            )
        },
        "ingestion_summary": {
            "upload_timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "file_count": len(uploaded_files),
            "uploaded_files": uploaded_files
        }
    }
    return metadata


def save_metadata_locally(metadata: dict, filepath: str) -> None:
    """Save metadata dictionary as a local JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logging.info("Saved metadata locally to '%s'.", filepath)


def upload_metadata_to_s3(s3_client, bucket_name: str, local_metadata_path: str, s3_key: str) -> None:
    """Upload local metadata JSON to S3."""
    upload_file_to_s3(s3_client, local_metadata_path, bucket_name, s3_key)


def upload_raw_files_to_s3():
    """
    Upload all supported files from the local raw data directory to AWS S3
    and generate/upload dataset metadata.
    """
    validate_environment()
    validate_raw_data_dir()

    logging.info("Connecting to AWS S3...")
    s3_client = get_s3_client()

    ensure_bucket_exists(s3_client, BUCKET_NAME, AWS_REGION)

    files_to_upload = list_supported_files(RAW_DATA_DIR)

    if not files_to_upload:
        raise FileNotFoundError(
            f"No supported files found in '{RAW_DATA_DIR}'. "
            f"Expected extensions: {SUPPORTED_EXTENSIONS}"
        )

    uploaded_files = []

    for filename in files_to_upload:
        local_file_path = os.path.join(RAW_DATA_DIR, filename)
        s3_key = f"raw/{filename}"

        try:
            upload_file_to_s3(s3_client, local_file_path, BUCKET_NAME, s3_key)
            uploaded_files.append(filename)
        except ClientError as e:
            logging.error("Failed to upload '%s': %s", filename, e)
        except Exception as e:
            logging.error("Unexpected error uploading '%s': %s", filename, e)

    if not uploaded_files:
        raise RuntimeError("No files were uploaded successfully.")

    # Generate and upload metadata
    metadata = generate_metadata(uploaded_files)
    save_metadata_locally(metadata, LOCAL_METADATA_FILE)
    upload_metadata_to_s3(s3_client, BUCKET_NAME, LOCAL_METADATA_FILE, S3_METADATA_KEY)

    logging.info("All raw data successfully uploaded to AWS S3.")
    logging.info("Uploaded files: %s", uploaded_files)
    logging.info("Metadata uploaded to: s3://%s/%s", BUCKET_NAME, S3_METADATA_KEY)


if __name__ == "__main__":
    try:
        upload_raw_files_to_s3()
    except NoCredentialsError:
        logging.error("AWS credentials not found. Check your .env file.")
    except Exception as e:
        logging.error("Ingestion pipeline failed: %s", e)