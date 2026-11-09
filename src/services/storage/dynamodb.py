from functools import lru_cache
import logging

import boto3
from botocore.exceptions import ClientError

from langchain_community.chat_message_histories import DynamoDBChatMessageHistory

from src.core.config import settings 
 
logger = logging.getLogger(__name__)


class DynamoDBService:
    """
    Manages the lifecycle of AWS DynamoDB for conversational memory:
    - Verifies or automatically creates the table 'veridex-chat-history'
    - Provisions a primary Partition Key 'SessionId' required by LangChain
    - Provides factory instances of DynamoDBChatMessageHistory for specific session IDs
    """

    def __init__(self):
        self.table_name = settings.DYNAMODB_TABLE_NAME
        
        self.region = settings.AWS_REGION

        self.dynamodb_client = boto3.client(
            "dynamodb",
            region_name=self.region,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )

        self._ensure_table_exists()

    def _ensure_table_exists(self) -> None:
        """
        Private helper method:
        Checks if the table exists in DynamoDB.
        If it does not exist, it creates the table with 'SessionId' as the primary key.
        """
        try:
            self.dynamodb_client.describe_table(TableName=self.table_name)
            logger.info(f"[DynamoDBService] Table '{self.table_name}' verified and ready.")
            print(f"[DynamoDBService] Table '{self.table_name}' verified and ready.")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            
            # If the table does not exist, DynamoDB raises ResourceNotFoundException
            if error_code == "ResourceNotFoundException":
                logger.info(f"[DynamoDBService] Table '{self.table_name}' not found. Creating...")
                print(f"[DynamoDBService] Table '{self.table_name}' not found. Creating table...")
                try:
                    
                    self.dynamodb_client.create_table(
                        TableName=self.table_name,
                        KeySchema=[
                            {"AttributeName": "SessionId", "KeyType": "HASH"}
                        ],
                        AttributeDefinitions=[
                            {"AttributeName": "SessionId", "AttributeType": "S"}
                        ],
                        BillingMode="PAY_PER_REQUEST"
                    )

                    waiter = self.dynamodb_client.get_waiter("table_exists")
                    waiter.wait(TableName=self.table_name)
                    logger.info(f"[DynamoDBService] Table '{self.table_name}' created successfully.")
                    print(f"[DynamoDBService] Table '{self.table_name}' created successfully.")
                except ClientError as create_err:
                    logger.error(f"[DynamoDBService] Failed to create DynamoDB table: {create_err}")
                    raise create_err
            else:
                logger.error(f"[DynamoDBService] Error accessing DynamoDB table: {e}")
                raise e

    def get_session_history(self, session_id: str) -> DynamoDBChatMessageHistory:
        """
        Returns a LangChain DynamoDBChatMessageHistory instance tied to a specific session_id.
        
        Parameters:
        - session_id: A unique conversation string (e.g. 'session_user42_conv1')
        """
        return DynamoDBChatMessageHistory(
            table_name=self.table_name,
            session_id=session_id,
            boto3_session=None  # LangChain reads the AWS credentials automatically or uses defaults
        )


@lru_cache()
def get_dynamodb_service() -> DynamoDBService:
    """
    Singleton factory: Returns a single authenticated DynamoDBService instance.
    """
    return DynamoDBService()