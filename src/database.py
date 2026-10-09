import os

import oracledb


# ============================================
# ORACLE DATABASE CONFIGURATION
# ============================================

DB_USER = "SYSTEM"
DB_PASSWORD = os.getenv("ORACLE_PASSWORD")
if not DB_PASSWORD:
    raise RuntimeError("ORACLE_PASSWORD environment variable is not set.")
DB_HOST = "djb"
DB_PORT = 1521
DB_SERVICE = "free"


# ============================================
# DATABASE CONNECTION
# ============================================

def get_connection():

    dsn = oracledb.makedsn(
        DB_HOST,
        DB_PORT,
        service_name=DB_SERVICE
    )

    connection = oracledb.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        dsn=dsn
    )

    return connection


# ============================================
# CONNECTION TEST
# ============================================

if __name__ == "__main__":

    try:

        connection = get_connection()

        print("===================================")
        print("ORACLE DATABASE CONNECTION")
        print("===================================")

        print("Connection successful!")
        cursor = connection.cursor()

        # Check connected user
        cursor.execute("SELECT USER FROM dual")
        result = cursor.fetchone()
        print("Connected User:", result[0])

        # Check tables
        cursor.execute("""
            SELECT table_name
            FROM user_tables
            ORDER BY table_name
        """)

        print("Tables in this schema:")

        for row in cursor:
            print(row[0])

        cursor.close()
        connection.close()

        print("Connection closed.")

    except oracledb.Error as error:

        print("Database connection failed.")
        print(error)