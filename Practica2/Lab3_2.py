from Vector3D import Vector3D

v1 = Vector3D(0,0,0)
v1.print()

v1.set_x(-6)
v1.set_y(10)
v1.set_z(5)
v1.print()

v2 = Vector3D(5,-1,0)
v1.add_v(v2)
v1.print()

v2 = Vector3D(-1,-1,-9)
v1.sub_v(v2)
v1.print()

v1.mul_scalar(3.5)
v1.print()

print(v1.mod())

v1.save_file_txt()
v1.save_file_pkl()

