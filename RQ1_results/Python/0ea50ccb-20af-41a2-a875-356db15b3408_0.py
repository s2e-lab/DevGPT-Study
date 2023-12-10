import logging
import sqlite3

class SessionManager:
    # Other methods and attributes

    def create_sessions_table(self):
        """Create the sessions table if it doesn't already exist."""
        logging.debug("CLASS SessionManager - create_sessions_table: Starting process to check or create sessions table.")
        
        try:
            with sqlite3.connect(self.db_name) as connection:
                logging.debug("CLASS SessionManager - create_sessions_table: Database connection established.")
                cursor = connection.cursor()
                
                logging.debug("CLASS SessionManager - create_sessions_table: Preparing SQL statement for sessions table creation.")
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    username TEXT,
                    role TEXT,
                    expiration TEXT
                )
                """
                
                cursor.execute(create_table_sql)
                logging.debug("CLASS SessionManager - create_sessions_table: SQL statement executed.")
                
                connection.commit()
                logging.debug("CLASS SessionManager - create_sessions_table: Changes committed to the database.")
                
                logging.info("CLASS SessionManager - create_sessions_table: Sessions table check and creation process completed successfully.")
                
        except sqlite3.Error as e:
            logging.error("CLASS SessionManager - create_sessions_table: SQLite error occurred: %s", e)
        except Exception as ex:
            logging.error("CLASS SessionManager - create_sessions_table: An unexpected error occurred: %s", ex)
