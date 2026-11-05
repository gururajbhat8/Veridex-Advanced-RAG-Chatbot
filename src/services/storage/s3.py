from functools import lru_cache
from typing import Optional, BinaryIO
import os
import logging
import boto3
from botocore.exceptions import ClientError

from src.core.config import settings

logger = logging.getLogger(__name__)

class S3StorageService:
    """
    Manages all interactions with AWS Simple Storage Service (S3):
    - Ensuring the private storage bucket exists
    - Uploading raw documents and extracted images
    - Generating temporary, secure Presigned URLs for frontend viewing
    """

    def __init__(self):
        self.bucket_name = settings.S3_BUCKET_NAME
        self.region = settings.AWS_REGION

        self.s3_client = boto3.client(
            "s3",
            region_name=self.region,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )

        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> None:
        """
        Private helper method: Verifies if the S3 bucket exists.
        If it does not exist, it creates the bucket with private ACL.
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"[S3StorageService] Bucket '{self.bucket_name}' verified and accessible.")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            
            if error_code in ["404", "NoSuchBucket"]:
                logger.info(f"[S3StorageService] Bucket '{self.bucket_name}' not found. Creating...")
                try:
                    if self.region == "us-east-1":
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={"LocationConstraint": self.region}
                        )
                    logger.info(f"[S3StorageService] Successfully created private bucket: {self.bucket_name}")
                except ClientError as create_err:
                    logger.error(f"[S3StorageService] Failed to create bucket: {create_err}")
                    raise create_err
            else:
                logger.error(f"[S3StorageService] Error accessing bucket '{self.bucket_name}': {e}")
                raise e

    def upload_file(
        self,
        file_obj: BinaryIO,
        s3_key: str,
        content_type: str = "application/pdf"
    ) -> str:
        """
        Uploads a file binary stream directly to S3.
        
        Parameters:
        - file_obj: Open file stream or BytesIO stream
        - s3_key: Object key/path in S3 (e.g. 'documents/doc123.pdf' or 'images/img1.png')
        - content_type: MIME type of the file
        
        Returns:
        - The S3 URI string: s3://bucket_name/s3_key
        """
        try:
            self.s3_client.upload_fileobj(
                Fileobj=file_obj,
                Bucket=self.bucket_name,
                Key=s3_key,
                ExtraArgs={"ContentType": content_type}
            )
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            logger.info(f"[S3StorageService] File successfully uploaded to: {s3_uri}")
            return s3_uri
        except ClientError as e:
            logger.error(f"[S3StorageService] Upload failed for key '{s3_key}': {e}")
            raise e

    def generate_presigned_url(self, s3_key: str, expiration_seconds: int = 900) -> str:
        """
        Generates a secure, temporary pre-signed URL allowing frontend clients
        to view or download private S3 objects without public bucket access.
        
        Parameters:
        - s3_key: Object key inside the bucket
        - expiration_seconds: URL validity period in seconds (default: 900s = 15 minutes)
        """
        try:
            url = self.s3_client.generate_presigned_url(
                ClientMethod="get_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": s3_key
                },
                ExpiresIn=expiration_seconds
            )
            return url
        except ClientError as e:
            logger.error(f"[S3StorageService] Failed to generate presigned URL for '{s3_key}': {e}")
            raise e


@lru_cache()
def get_s3_service() -> S3StorageService:
    """
    Singleton factory: Returns a single authenticated S3StorageService instance.
    """
    return S3StorageService()