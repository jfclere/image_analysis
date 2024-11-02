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
      print(row)
      for col in range(startcol, endcol):
        print(col)
        if self.is_inside(row, col, center, radius):
            # Means the point is inside/on the face
            # Make it opaque
            #mg[row][col][3] = 0
            print("negatif...")
            numpts = numpts + 3
            totalval = totalval + img[row][col][0]
            totalval = totalval + img[row][col][1]
            totalval = totalval + img[row][col][2]
            img[row][col] = 255
        # else:
            # Means the point is outside the face
            # Make it transparent
            #img[row][col][3] = 255
            #img[row][col] = 0
    print("value: ")
    print(totalval/numpts)
    return img

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
            # for c in countours:
            c = max(countours, key=cv2.contourArea)
            center,radius = cv2.minEnclosingCircle(c)
            print("center")
            print(center)
            print("radius")
            print(radius)
           
            #self.trail = self.value_in_circle (frame, center, radius)
            trail = self.value_in_circle (frame, center, radius)
            cv2.imwrite("result.jpg", trail);
            # JFC (x,y) = center
            # JFC x = int(x)
            # JFC y = int(y)
            # JFC radius = int(radius)
            # JFC circle_mask = numpy.zeros((self.height,self.width), numpy.uint8)
            # JFC circle_img = cv2.circle(circle_mask, (x, y) ,radius, (255,255,255),-1)
            # JFC masked_data = cv2.bitwise_and(frame, frame, mask=circle_img)
              # _,thresh = cv2.threshold(circle_mask,1,255,cv2.THRESH_BINARY)
              # circle_contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
              # print(len(circle_contours))
              # x,y,w,h = cv2.boundingRect(circle_contours[0])
              # crop = masked_data[y:y+h,x:x+w]
            # self.trail = masked_data
            # num = numpy.mean(circle_img) * 100 / 255
            # print("cv2.countNonZero image: " + str(num))
            cv2.drawContours(frame, countours, -1, (0,255,0), 3)
            # self.trail = frame;
            c = max(countours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            moments = cv2.moments(c)
            if moments["m00"] > 0:
                center = int(moments["m10"] / moments["m00"]), \
                         int(moments["m01"] / moments["m00"])
            else:
                center = int(x), int(y)

            # only proceed if the radius meets a minimum size
            if radius > 9:
                print("Found: " + str(radius))
                print("Found at " + str(x) + ":" + str(y));
                # draw the circle and centroid on the frame,
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                # then update the ponter trail
                if self.previous_position:
                    cv2.line(self.trail, self.previous_position, center,
                             (255, 255, 255), 2)
            else:
                print("Too small???" + str(radius))

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
    # cv2.imwrite("result.jpg", finder.trail);
