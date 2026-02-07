import logging
from threading import Thread
import time

from src.db.factory import make_database
from src.repositories.request_log import LogsRepository

logger = logging.getLogger(__name__)


def cleanup_task(inactive_minutes: int = 30, check_interval_seconds: int = 300):
    """
    Background task to clean up inactive sessions.
    
    Args:
        inactive_minutes: Sessions inactive for this many minutes will be deleted
        check_interval_seconds: How often to run the cleanup (default: 5 minutes)
    """
    logger.info(
        f"Cleanup task started: will check every {check_interval_seconds}s "
        f"and delete sessions inactive > {inactive_minutes} minutes"
    )
    
    while True:
        try:
            time.sleep(check_interval_seconds)
            
            # Get database instance
            database = make_database()
            
            # Use context manager to get session (matching your dependencies.py pattern)
            with database.get_session() as session:
                # Run cleanup
                deleted_count = LogsRepository.cleanup_inactive_sessions(
                    session, 
                    inactive_minutes=inactive_minutes
                )
                
                if deleted_count > 0:
                    logger.info(
                        f"Cleanup: Deleted {deleted_count} logs from inactive sessions "
                        f"(inactive > {inactive_minutes} minutes)"
                    )
            
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}", exc_info=True)


def start_cleanup_task(inactive_minutes: int = 30, check_interval_seconds: int = 300):
    """Start the cleanup task in a background thread."""
    thread = Thread(
        target=cleanup_task,
        args=(inactive_minutes, check_interval_seconds),
        daemon=True,
        name="session-cleanup"
    )
    thread.start()
    logger.info("Background cleanup thread started")