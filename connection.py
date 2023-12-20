import mysql.connector

def connect_to_database():
    conn = mysql.connector.connect(host='localhost', user='root', password='', database='swinedb')
    if conn.is_connected():
        print('Connected to the database')
        return conn
    else:
        print('Failed to connect to the database')
        return None

def fetch_data_from_database(conn, sql_query):
    cur = conn.cursor()
    cur.execute(sql_query)
    data = cur.fetchall()
    return data