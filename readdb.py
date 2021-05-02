import sqlite3
import csv
from pdfgenerate import create_pdf
def insert_db(filename1):
    conn=sqlite3.connect('result.db')
    c=conn.cursor()
    c.execute('''DROP TABLE student''')
    c.execute('''CREATE TABLE student 
    (ID INT PRIMARY KEY NOT NULL,
    NAME TEXT NOT NULL,
    MATH REAL,
    SCIENCE REAL,
    OPT_MATH REAL,
    ENGLISH REAL,
    OOP REAL,
    PERCENT REAL,
    STATUS TEXT NOT NULL);''')
    with open(filename1,newline='') as csvfile:
        data=csv.DictReader(csvfile)
        for row in data:
            c.execute('''INSERT INTO student 
                (ID,NAME,MATH,SCIENCE,OPT_MATH,ENGLISH,OOP,PERCENT,STATUS) VALUES 
                    (?,?,?,?,?,?,?,?,?)''',(row['ID'],row['NAME'],
                    row['MATH']
                        ,row['SCIENCE'],row['OPT_MATH'],
                        row['ENGLISH'],row['OOP'],
                        row['PERCENT'],
                        row['STATUS'])
                        )
            conn.commit()
        conn.close()
def fetch(ID1):
    conn=sqlite3.connect('result.db')
    c=conn.cursor()
    c.execute('''SELECT * FROM student WHERE ID=?''',(ID1,))
    x=c.fetchall()
    return x[0]
def update_db(ID,sub):
    sub_name=sub[0]
    sub_mark=sub[1]
    conn=sqlite3.connect('result.db')
    c=conn.cursor()
    comm="UPDATE student SET {}= ? WHERE ID=?".format(sub_name)
    c.execute(comm,(sub_mark,ID,))
    y=fetch(ID)
    res=0
    status='PASS'
    for i in range(2,len(y)-2):
        res=res+y[i]
        if y[i]<=30:
            status='Fail'
    res=res/(len(y)-3)
    c.execute('''UPDATE student
    SET PERCENT= ? WHERE ID=?''',(res,ID,))
    c.execute('''UPDATE student
    SET STATUS= ? WHERE ID=?''',(status,ID,))
    conn.commit()
    conn.close()
update_db(6,('MATH',55))
update_db(6,('SCIENCE',65))
update_db(6,('OPT_MATH',54))
update_db(6,('ENGLISH',76))
update_db(6,('OOP',86))
create_pdf((6,'NITIN',55,65,54,76,86,67.2,'PASS'))