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
    # looking for red (#[ 68  32 186] not too bad RGB)
    # H:346 S:70.6% L: 42.7 % (from Internet)
    # was hue_min = 20, hue_max = 160, sat_min = 100, sat_max = 255
    # For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
    if red:
      self.hue_min = 20
      self.hue_max = 160
      self.sat_min = 150
      self.sat_max = 255
      self.val_min = 150
      self.val_max = 210

    self.trail = numpy.zeros((height, width, 3), numpy.uint8)

  def is_inside(self, row, col, center, radius):
    """ This function returns True if point is inside/on the boundary, False otherwise """
    # eqn = (point[0] - center[0]) ** 2 + (point[1] - center[1])**2 - (radius**2)
    eqn = (col - center[0]) ** 2 + (row - center[1])**2 - (radius**2)
    return eqn <= 0.

  def value_in_circle(self, img, center, radius):
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
            numpts = numpts + 3
            totalval = totalval + img[row][col][0]
            totalval = totalval + img[row][col][1]
            totalval = totalval + img[row][col][2]
    if numpts == 0:
      return 0
    print("value_in_circle value: " + str(totalval/numpts))
    return totalval/numpts

  def value_red_in_circle(self, img, center, radius, radiusext):
    startrow = int(center[1]-radius)
    endrow = int(center[1]+radius)
    startcol = int(center[0]-radius)
    endcol = int(center[0]+radius)
    for row in range(startrow, endrow):
      for col in range(startcol, endcol):
        if self.is_inside(row, col, center, radius):
            # Means the point is inside/on the face
            img[row][col] = 0 # make it dark.


    numpts = 0
    totalred = 0
    totalgreen = 0
    totalblue = 0
    startrow = int(center[1]-radiusext)
    endrow = int(center[1]+radiusext)
    startcol = int(center[0]-radiusext)
    endcol = int(center[0]+radiusext)
   
    for row in range(startrow, endrow):
      for col in range(startcol, endcol):
        if self.is_inside(row, col, center, radius):
            # Means the point is inside/on the face
            if img[row][col][0] == 0 and img[row][col][1] == 0 and img[row][col][2] == 0:
               continue # skip center...
            numpts = numpts + 1
            totalred = totalred + img[row][col][2]
            totalgreen = totalgreen + img[row][col][1]
            totalblue = totalblue + img[row][col][0]
    if numpts == 0:
      return 0
    mred = totalred/numpts
    mgreen = totalgreen/numpts
    mblue = totalblue/numpts 

    # try to find the red of the laser
    # ignore bright points
    #if mgreen>240 and mblue>240:
    #   return 0
    print("value_red_in_circle " + str(mblue) + " " + str(mgreen) + " " + str(mred))
    #  return 0 # Not mostly red
    print("value_red_in_circle value red: " + str(totalred/numpts))
    return totalred/numpts

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
            i = 0
            for c in countours:
              center,radius = cv2.minEnclosingCircle(c)
              if (radius < 1):
                continue
              print("center: " + str(center))
              print("radius: " + str(radius))
           
              bright = self.value_in_circle (frame, center, radius)
              if (bright > 250):
                print(str(bright) + " Found at " + str(center) + " radius " + str(radius))
              else:
                continue

              # check for red around the bright contour
              red = self.value_red_in_circle(frame, center, radius, radius*2)
              if (red > 160):
                print(str(red) + " Found red at " + str(center) + " radius " + str(radius))
              else:
                print("MERDE")
                continue

              (x,y) = center
              x = int(x)
              y = int(y)
              radius = int(radius)
              # draw the circle and centroid on the frame,
              # looking to the laser color as seen by the camera
              # 50, 47, 237 near but not...
              # read in the jpg (vertically from the center)
              #[ 83  76 149]
              #[ 84  77 158]
              #[ 86  76 166]
              #[ 87  70 173]
              #[ 72  48 172]
              #[ 68  32 186] not too bad
              #[ 64  14 186]
              #[ 92  27 206]
              #[ 95  37 185]
              #[168 137 212]
              #[168 137 212]
              #[254 241 255]
              #[0 0 0] (0 was written there on purpose)

              cv2.circle(frame, (x, y), radius+30,
                         (68, 32, 186), 2)
              print(frame[y-12, x])
              print(frame[y-11, x])
              print(frame[y-10, x])
              print(frame[y-9, x])
              print(frame[y-8, x])
              print(frame[y-7, x])
              print(frame[y-6, x])
              print(frame[y-5, x])
              print(frame[y-4, x])
              print(frame[y-3, x])
              print(frame[y-2, x])
              print(frame[y-1, x])
              if i == 0:
                frame[y-5, x] = 255
              i = i + 1
              # cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.drawContours(frame, countours, -1, (0,255,0), 3)

        # cv2.add(self.trail, frame, frame)
        self.trail = frame;
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
    finder = FindLaser(height, width, 1)
    image = finder.detect(image)
    #cv2.imwrite("result.jpg", finder.trail);
    cv2.imwrite("result.jpg", image)
