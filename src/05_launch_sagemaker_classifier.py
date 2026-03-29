import os
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from dotenv import load_dotenv

load_dotenv()

sagemaker_session = sagemaker.Session()
role = os.getenv("SAGEMAKER_ROLE_ARN")
bucket = os.getenv("S3_BUCKET_NAME")


def launch_classifier_training_job():
    print("Launching SageMaker failure classification training job for FD003...")

    train_data_uri = f"s3://{bucket}/processed/"
    output_uri = f"s3://{bucket}/models/sagemaker_output/classifier/"

    estimator = SKLearn(
        entry_point="sagemaker_xgboost_classifier.py",
        source_dir="src",
        dependencies=["src/requirements.txt"],
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        framework_version="1.0-1",
        py_version="py3",
        sagemaker_session=sagemaker_session,
        output_path=output_uri,
    )

    estimator.fit({
        "training": train_data_uri
    })


if __name__ == "__main__":
    launch_classifier_training_job()