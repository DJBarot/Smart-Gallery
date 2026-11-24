import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import get_connection

connection = get_connection()
cursor = connection.cursor()

cursor.execute("""
SELECT
USER,
SYS_CONTEXT('USERENV', 'CON_NAME'),
SYS_CONTEXT('USERENV', 'SERVICE_NAME')
FROM DUAL
""")

print("Database connection successful!")
print("Connection details:", cursor.fetchone())

cursor.execute("""
SELECT TABLE_NAME
FROM USER_TABLES
WHERE TABLE_NAME IN ('IMAGES', 'IMAGE_FEATURES')
ORDER BY TABLE_NAME
""")

print("Available tables:", cursor.fetchall())

cursor.close()
connection.close()