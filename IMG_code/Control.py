import serial

def high_byte(_num):
    return int((int(_num) >> 8) & 0xFF)

def low_byte(_num):
    return int(int(_num) & 0xFF)

class Control():

    def __init__(self,name):
        self.ser = serial.Serial()
        self.COM_PORT = name
        self.port_connected = False
        self.posx = 0
        self.posz = 0
        self.posy = 0
        self.posmidx = 650
        self.posmidz = 600

    def updateposx(self, value):
        self.posx = int(value)
        print("X:", self.posx)

    def updateposz(self, value):
        self.posz = int(value)
        print("Z:", self.posz)

    def updateposy(self, value):
        self.posy = int(value)
        print("Y:", self.posy)
    
    def updatestrcomport(self):
        self.COM_PORT_TXT = str(self.plainTextEdit.toPlainText())
        print("Serial Port:", self.COM_PORT_TXT)

    def uart1_connect(self):
        if not self.port_connected:
            try:
                self.ser.baudrate = 115200
                self.ser.port = self.COM_PORT
                self.ser.timeout = None
                self.ser.rts = 0
                self.ser.open()  # Open serial port

                    # print port open or close
                if self.ser.isOpen():
                    print('Connected:', self.ser.portstr)
                    self.port_connected = True
            except serial.serialutil.SerialException:
                print("Warning: Serial Port", self.COM_PORT, "is not connected.")
        else:
            if 'COM' in self.COM_PORT:
                print("Warning:", self.COM_PORT, "is being opened.")

    def uart1_disconnect(self):
        if self.port_connected:
            try:
                self.ser.close()  # Close serial port
                print('Disconnected:', self.COM_PORT)
                self.port_connected = False
            except serial.serialutil.SerialException:
                print("Warning: Unable to disconnect Serial Port", self.COM_PORT, ".")
        else:
            if 'COM' in self.COM_PORT:
                print("Warning:", self.COM_PORT, "is not opened.")

    def set_home_command(self):
        data = serial.to_bytes([0x46, 0x58, 0x00, 0x00,
                                0x5A, 0x00, 0x00,
                                0x59, 0x00, 0x00,
                                0x50, low_byte(0), 0x52, low_byte(135), 0x53])
        print("SET HOME")
        print("Passcode: ", data)
        if self.port_connected:
            self.ser.write(data)
            print(self.ser.readline().decode())
        else:
            print("Warning: Serial Port", self.COM_PORT, "is not opened.")
    
    def send_data_command(self):
        data = serial.to_bytes(
                [0x46, 0x58, high_byte(self.posx), low_byte(self.posx), 0x5A, high_byte(self.posz),
                 low_byte(self.posz), 0x53])
        print("X:", self.posx, "Z:", self.posz)
        print("Passcode: ", data)
        if self.port_connected:
            self.ser.write(data)
            print(self.ser.readline().decode())
        else:
            print("Warning: Serial Port", self.COM_PORT, "is not opened.")

    