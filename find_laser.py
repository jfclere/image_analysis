#! /usr/bin/env python
import sys
import argparse
import cv2
import numpy

class FindLaser():
  def __init__(self, height, width, red):
    self.channels = {
            'hue': None,
            'saturation': None,
            'value': None,
            'laser': None,
    }
    self.previous_position = None
    self.height = height
    self.width = width
    # looking for white
    self.hue_min = 0
    self.hue_max = 10
    self.sat_min = 0
    self.sat_max = 10
    self.val_min = 200
    self.val_max = 255
    # looking for red
    if red:
      self.hue_min = 20
      self.hue_max = 160
      self.sat_min = 100
      self.sat_max = 255

    self.trail = numpy.zeros((height, width, 3), numpy.uint8)

  def is_inside(self, row, col, center, radius):
    """ This function returns True if point is inside/on the boundary, False otherwise """
    # eqn = (point[0] - center[0]) ** 2 + (point[1] - center[1])**2 - (radius**2)
    eqn = (col - center[0]) ** 2 + (row - center[1])**2 - (radius**2)
    print(eqn)
    return eqn <= 0.

  def value_in_circle(self, img, center, radius):
    print("value_in_circle")
    print(img.shape[0])
    print(img.shape[1])
    print(center[0])
    print(center[1])
    print("value_in_circle")
    # for row in range(img.shape[0]):
    startrow = int(center[1]-radius)
    endrow = int(center[1]+radius)
    startcol = int(center[0]-radius)
    endcol = int(center[0]+radius)
    numpts = 0
    totalval = 0
   
    for row in range(startrow, endrow):
      for col in range(startcol, endcol):
        if self.is_inside(row, col, center, radius):
            # Means the point is inside/on the face
            # Make it opaque
            #mg[row][col][3] = 0
            print("negatif...")
            numpts = numpts + 3
            totalval = totalval + img[row][col][0]
            totalval = totalval + img[row][col][1]
            totalval = totalval + img[row][col][2]
    if numpts == 0:
      return 0
    print("value: ")
    print(totalval/numpts)
    return totalval/numpts

  def value_red_in_circle(self, img, center, radius):
    print("value__red_in_circle")
    print(img.shape[0])
    print(img.shape[1])
    print(center[0])
    print(center[1])
    print("value__red_in_circle")
    # for row in range(img.shape[0]):
    startrow = int(center[1]-radius)
    endrow = int(center[1]+radius)
    startcol = int(center[0]-radius)
    endcol = int(center[0]+radius)
    numpts = 0
    totalval = 0
   
    for row in range(startrow, endrow):
      for col in range(startcol, endcol):
        if self.is_inside(row, col, center, radius):
            # Means the point is inside/on the face
            # Make it opaque
            #mg[row][col][3] = 0
            print("negatif...")
            numpts = numpts + 1
            totalval = totalval + img[row][col][2]
    if numpts == 0:
      return 0
    print("value red: " + str(totalval/numpts))
    return totalval/numpts

  def threshold_image(self, channel):
        if channel == "hue":
            minimum = self.hue_min
            maximum = self.hue_max
        elif channel == "saturation":
            minimum = self.sat_min
            maximum = self.sat_max
        elif channel == "value":
            minimum = self.val_min
            maximum = self.val_max

        (t, tmp) = cv2.threshold(
            self.channels[channel],  # src
            maximum,  # threshold value
            0,  # we dont care because of the selected type
            cv2.THRESH_TOZERO_INV  # t type
        )

        (t, self.channels[channel]) = cv2.threshold(
            tmp,  # src
            minimum,  # threshold value
            255,  # maxvalue
            cv2.THRESH_BINARY  # type
        )

        if channel == 'hue':
            # only works for filtering red color because the range for the hue
            # is split
            self.channels['hue'] = cv2.bitwise_not(self.channels['hue'])

  def track(self, frame, mask):
        """
        Track the position of the laser pointer.

        Code taken from
        http://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
        """
        center = None

        countours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2]

        # only proceed if at least one contour was found
        if len(countours) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            print(len(countours))
            # the laser is a white circle surrounded with red
            for c in countours:
              center,radius = cv2.minEnclosingCircle(c)
              if (radius < 2):
                continue
              print("center")
              print(center)
              print("radius")
              print(radius)
           
              bright = self.value_in_circle (frame, center, radius)
              if (bright > 250):
                print(str(bright) + " Found at " + str(center) + " radius " + str(radius))
              else:
                continue

              # check for red around the bright contour
              red = self.value_red_in_circle (frame, center, radius+10)
              if (red > 160):
                print(str(red) + " Found red at " + str(center) + " radius " + str(radius))
              else:
                continue

              (x,y) = center
              x = int(x)
              y = int(y)
              radius = int(radius)
              # draw the circle and centroid on the frame,
              cv2.circle(frame, (x, y), radius+10,
                         (0, 255, 255), 2)
              cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.drawContours(frame, countours, -1, (0,255,0), 3)
            self.trail = frame;

        # cv2.add(self.trail, frame, frame)
        # self.trail = frame;
        self.previous_position = center

  def detect(self, frame):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # split the video frame into color channels
        h, s, v = cv2.split(hsv_img)
        self.channels['hue'] = h
        self.channels['saturation'] = s
        self.channels['value'] = v

        # Threshold ranges of HSV components; storing the results in place
        self.threshold_image("hue")
        self.threshold_image("saturation")
        self.threshold_image("value")

        # Perform an AND on HSV components to identify the laser!
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['hue'],
            self.channels['value']
        )
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['saturation'],
            self.channels['laser']
        )

        # Merge the HSV components back together.
        hsv_image = cv2.merge([
            self.channels['hue'],
            self.channels['saturation'],
            self.channels['value'],
        ])

        self.track(frame, self.channels['laser'])

        return hsv_image

if __name__ == '__main__':
    image = cv2.imread("/tmp/now.jpg");
    if image is None:
      exit(1)
    height, width, channels = image.shape
    print (height, width, channels)
    finder = FindLaser(height, width, 0)
    image = finder.detect(image)
    cv2.imwrite("result.jpg", finder.trail);
