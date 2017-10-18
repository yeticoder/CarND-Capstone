import rosbag
from cv_bridge import CvBridge
import sys
import os
import cv2

if (len(sys.argv) <2):
	print "invalid number of arguments:   " + str(len(sys.argv))
	print "should be 2: bagfile"
	sys.exit(1)

bag_file = sys.argv[1]

bag = rosbag.Bag(bag_file)
bagContents = bag.read_messages()
bagName = bag.filename
print("Processing bag %s" % bagName)
folder = bagName.replace('.bag', '')

if os.path.isdir(folder):
    print("Folder already exists")
    sys.exit(1)
else:
    os.makedirs(folder)

# listOfTopics = []
# for topic, msg, t in bagContents:
#     if topic not in listOfTopics:
#         print(topic)
#         listOfTopics.append(topic)


"/image_raw"
"/current_pose"

bridge = CvBridge()
sequences = []
for subtopic, msg, t in bag.read_messages("/current_pose"):	# for each instant in time that has data for topicName
    sequences.append(str(msg.header.seq))

print("Found %s messages" % len(sequences))

i = 0
for subtopic, msg, t in bag.read_messages("/image_raw"):	# for each instant in time that has data for topicName
    i += 1
    cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    img_name = os.path.join(folder, "img_%04d.jpg" % i)
    cv2.imwrite(img_name, cv2_img)

print("Exported %s images" % i)
