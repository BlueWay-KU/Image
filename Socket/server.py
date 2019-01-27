from socket import *
import socket
import os
import time
import sys

src = "C:\\Users\\Inyong\\Desktop\\"

def fileName():
    imgFileName = src + 'input.jpg'
    Name = 'input.jpg'
    return Name

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("", 49153))
server_socket.listen(5)

print("TCPServer Waiting for client on port 49153")

while True:

    client_socket, address = server_socket.accept()
    print("I got a connection from ", address)

    data = None


    while True:
        img_data = client_socket.recv(1024)
        data = img_data
        if img_data:
            while img_data:
                print("recving Img...")
                img_data = client_socket.recv(1024)
                data += img_data
            else:
                break

    img_fileName = fileName()
    img_file = open(img_fileName, "wb")
    print(img_fileName)
    print("finish img recv")
    print(sys.getsizeof(data))
    img_file.write(data)
    img_file.close()
    print("Finish ")




client_socket.close()
print("SOCKET closed... END")
