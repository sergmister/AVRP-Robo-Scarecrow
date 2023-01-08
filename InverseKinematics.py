import numpy as np
import ikpy.chain
import ikpy.utils.plot as plot_utils

from math import pi


class IK:
  def __init__(self):
    self.chain = ikpy.chain.Chain.from_urdf_file("birdlaser.urdf")

  def translate_from_coords(self, x, y, z):
    return (-z, -x, y)

  def translate_to_coords(self, x, y, z):
    return (-y, z, -x)

  def translate_to_angle(self, rad):
    return rad * 180 / pi + 90

  def calc(self, x, y, z):
    result = self.chain.inverse_kinematics(self.translate_from_coords(x, y, z))
    return {"rotate": self.translate_to_angle(result[3]) + 6, "tilt": self.translate_to_angle(result[5])}

if __name__ == '__main__':
  ik = IK()
  while True:
    print(ik.calc(*list(map(float, input().strip().split()))))
