# AI-driven Interactive Lab Assistant w/ OpenCV & ChatGPT
#
# NVIDIA® Jetson Nano
#
# By Kutluhan Aktar
#
# In remote learning, provide insightful guidance on lab equipment as auto-generated lessons
# via object detection and artificial intelligence. 
# 
#
# For more information:
# https://www.hackster.io/kutluhan-aktar


from _class import lab_assistant_op
from threading import Thread


# Define the lab_assistant object.
lab_assistant = lab_assistant_op("model/ai-driven-interactive-lab-assistant-linux-aarch64.eim")

# Define and initialize threads.
Thread(target=lab_assistant.camera_feed).start()

# Activate the user interface (GUI).
lab_assistant.show_interface()