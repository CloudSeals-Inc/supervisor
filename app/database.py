import os
from motor.motor_asyncio import AsyncIOMotorClient
import logging

logger = logging.getLogger(__name__)

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017/MibaDB")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "MibaDB")

class Database:
    client: AsyncIOMotorClient = None
    db = None

db = Database()

async def connect_to_mongo():
    db.client = AsyncIOMotorClient(MONGODB_URI)
    db.db = db.client[MONGODB_DB_NAME]
    logger.info("Connected to MongoDB: %s", MONGODB_DB_NAME)

async def close_mongo_connection():
    db.client.close()
    logger.info("Closed MongoDB connection")
