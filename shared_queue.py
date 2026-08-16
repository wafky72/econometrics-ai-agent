# shared_queue.py
import asyncio
from typing import Dict

# Using a dictionary to store each user's message queue
message_queues: Dict[str, asyncio.Queue] = {}

async def get_or_create_queue(user_id: str) -> asyncio.Queue:
    """Get or create a user's message queue"""
    if user_id not in message_queues:
        message_queues[user_id] = asyncio.Queue()
    return message_queues[user_id]

async def log_execution(message: str, user_id: str):
    """Write a message to a user's queue"""
    queue = await get_or_create_queue(user_id)
    await queue.put(message)

async def get_message(user_id: str):
    """Get a message from a user's queue"""
    queue = await get_or_create_queue(user_id)
    return await queue.get()

def queue_empty(user_id: str) -> bool:
    """Check if a user's queue is empty"""
    if user_id not in message_queues:
        return True
    return message_queues[user_id].empty()

def cleanup_queue(user_id: str):
    """Clean up a user's message queue"""
    if user_id in message_queues:
        del message_queues[user_id]
