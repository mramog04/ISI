import pickle


class Vector3D:
    x = 0
    y = 0
    z = 0
    
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
    
    def set_x(self,x):
        self.x = x
    
    def set_y(self,y):
        self.y = y
    
    def set_z(self,z):
        self.z = z
    
    def toString(self):
        return "(" + str(self.x) + "," + str(self.y) + "," + str(self.z) + ")"
    
    def print(self):
        print(self.toString())
    
    def add_v(self,v):
        self.x += v.x
        self.y += v.y
        self.z += v.z
    
    def sub_v(self,v):
        self.x -= v.x
        self.y -= v.y
        self.z -= v.z
    
    def mul_scalar(self,scalar):
        self.x *= scalar
        self.y *= scalar
        self.z *= scalar
        
    def mod(self):
        return (self.x**2 + self.y**2 + self.z**2)**0.5
    
    def save_file_txt(self):
        archivo = open("vector.txt","a")
        archivo.write(self.toString())
        archivo.write("\n")
        archivo.close()
        
    def save_file_pkl(self):
        tuplaVector = (self.x,self.y,self.z)
        archivo = open("vector.pkl","ab")
        pickle.dump(tuplaVector,archivo)
        archivo.close()