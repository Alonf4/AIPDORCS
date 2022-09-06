import csv

with open('~\\..\\Dynamo\\Project 001\\BeamsData.csv', 'r') as csv_file:
    beamsFile = csv.reader(csv_file, delimiter=',')
    for row in beamsFile:
        match row[0]:
            case 'Beam ID':
                print(row)
            case 'Beam Connections':
                connections = row[1:]
                print(connections)