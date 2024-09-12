import sqlite3
db = sqlite3.connect('ragnews.db')
# no standard file extension for sqlite

cursor = db.cursor()
sql = '''
SELECT count(*) FROM articles;
'''
cursor.execute(sql)
row = cursor.fetchone()
print(f"row={row}")
