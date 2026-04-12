import os
import asyncpg
import logging
import asyncio

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

class Database:
    pool: asyncpg.Pool = None

    async def connect(self):
        """Establish connection pool with retry logic."""
        attempts = 5
        while attempts > 0:
            try:
                self.pool = await asyncpg.create_pool(
                    DATABASE_URL,
                    min_size=1,
                    max_size=10
                )
                logger.info("Connected to PostgreSQL successfully (Supervisor)")
                return
            except Exception as e:
                attempts -= 1
                logger.error(f"PostgreSQL connection attempt failed (Supervisor): {e}. Retrying in 5s...")
                await asyncio.sleep(5)
        
        raise Exception("Could not connect to PostgreSQL after multiple attempts")

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
            logger.info("Closed PostgreSQL connection (Supervisor)")

db = Database()

async def connect_to_db():
    await db.connect()

async def close_db_connection():
    await db.disconnect()
