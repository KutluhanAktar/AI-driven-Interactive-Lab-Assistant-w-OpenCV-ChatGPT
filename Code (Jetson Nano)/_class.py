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

import cv2
import numpy
from guizero import App, Window, Box, Text, TextBox, PushButton, ButtonGroup, Combo, CheckBox, MenuBar, Picture, info, yesno, warn
from gtts import gTTS
from edge_impulse_linux.image import ImageImpulseRunner
import os
import requests
import subprocess
import webbrowser
import datetime
from time import sleep


class lab_assistant_op():
    def __init__(self, model_file):
        # Initialize the USB high-quality camera feed.
        self.camera = cv2.VideoCapture(0)
        sleep(3)
        # Define the required variables to configure camera settings.
        self.con_opt = "normal"
        self.cam_init = True
        self.frame_size = (480,480)
        # Define the required configurations to run the Edge Impulse FOMO object detection model.
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model_file = os.path.join(dir_path, model_file)
        self.detected_class = "Waiting..."
        # Define the required variables for the OpenAI API.
        self.OPENAI_API_KEY = "OPENAI_API_KEY"
        self.OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
        # Assign the defined user interface structure.
        self.create_user_interface()
        sleep(10)

    def run_inference(self, notify=False):
        # Run inference to detect various types of lab equipment.
        with ImageImpulseRunner(self.model_file) as runner:
            try:
                # Print the information of the Edge Impulse model converted to a Linux (AARCH64) application (.eim).
                model_info = runner.init()
                print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
                labels = model_info['model_parameters']['labels']
                # Get the currently captured image with the high-quality USB camera.
                # Then, convert the any given frame format to RGB by temporarily saving the frame as a JPG file and reading it.
                cv2.imwrite("./tmp/temp_frame.jpg", self.latest_frame)
                # After reading the temporary JPG file, resize the converted frame depending on the given model so as to run an inference. 
                test_img = cv2.imread("tmp/temp_frame.jpg")
                features, cropped = runner.get_features_from_image(test_img)
                res = runner.classify(features)
                # Obtain the prediction (detection) results for each label (class).
                if "bounding_boxes" in res["result"].keys():
                    print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                    # If the Edge Impulse model predicts a class successfully:
                    if(len(res["result"]["bounding_boxes"]) == 0):
                        self.detected_class = "[...]"
                    else:
                        for bb in res["result"]["bounding_boxes"]:
                            # Get the latest detected labels:
                            self.detected_class = bb['label']
                            print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                            cropped = cv2.rectangle(cropped, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)
                # Show the model detection image with the passed bounding boxes if any on the screen.
                cv2.imshow("Model Detection Image", cropped)
                # Remove the temporary image file.
                os.remove("./tmp/temp_frame.jpg")
                # Update the interface inquiry options according to the predicted class (label).
                __l = self.detected_class.replace("_", " ")
                self.inquiry_input.clear()
                self.inquiry_input.insert(0, "How to use {} in labs?".format(__l))
                self.inquiry_input.insert(1, "Please list the lab experiments with {}".format(__l))
                self.inquiry_input.insert(2, "Tell me about recent research papers on {}".format(__l))
                self.inquiry_input.insert(3, "How to repair {}?".format(__l))
                # Notify the user of the detection results.
                self.model_notify.value = "Latest Detection: " + self.detected_class
                print("\n\nLatest Detected Label => " + self.detected_class)
                # If requested, also inform the user via a pop-up window on the screen.
                if notify:
                    self.app.warn("Latest Detected Label", self.detected_class)        
            # Stop the running inference.    
            finally:
                if(runner):
                    runner.stop()

    def chatgpt_get_information(self, inquiry):
        # Make a cURL call (request) to the OpenAI API in order to get information regarding the given lab component from ChatGPT.
        # Define the required HTML headers.
        headers = {"Authorization": "Bearer " + self.OPENAI_API_KEY, "Content-Type": "application/json"}
        # Define POST data parameters in the JSON format.
        json_data = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": inquiry}, {"role": "user", "content": "Please add the latest asked question before the answer."}], "temperature": 0.6}
        # Obtain information from ChatGPT by making a cURL call to the OpenAI API.
        curl = requests.post(self.OPENAI_ENDPOINT, json=json_data, headers=headers)
        if(curl.status_code == 200):
            chatgpt_info = curl.json()
            chatgpt_info = chatgpt_info["choices"][0]["message"]["content"]
            return chatgpt_info
        else:
            return "Error: ChatGPT"
    
    def chatgpt_show_information(self):
        selected_inquiry = self.inquiry_input.value
        # Notify the user of the passed inquiry.
        user_notification = "Inquiry: " + selected_inquiry + "\n\nWaiting response from ChatGPT..."
        info("Status", user_notification)
        # Obtain information from ChatGPT.
        chatgpt_info = self.chatgpt_get_information(selected_inquiry)
        print("ChatGPT Response Received Successfully!")
        # Display the received information generated by ChatGPT on the second window.
        self.second_window.show()
        self.second_w_text.value = chatgpt_info
        # If requested, convert the received ChatGPT response to audio lesson and save it.
        audio_lesson_opt = yesno("Audio Lesson", "Save the obtained ChatGPT response as audio lesson?")
        if (audio_lesson_opt == True):
            print("\n\nConverting ChatGPT response...")
            self.text_to_speech_lesson(chatgpt_info)

    def text_to_speech_lesson(self, _text, language="en"):
        # Convert the information about the given lab equipment, generated by ChatGPT, to speech (audio lesson).
        text_to_speech = gTTS(text=_text, lang=language, slow=False)
        # Define the audio file (MP3) path, including the current date.
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = self.detected_class + "_" + date + ".mp3"
        file_path = "audio_lesson/" + file_name
        # Save the generated speech (audio) file.
        text_to_speech.save(file_path)
        sleep(2)
        print("\nAudio Lesson Saved: " + file_name)
        # If requested, play the generated audio lesson immediately.
        if(self.audio_input.value == True):
            print("\nLatest Audio Lesson: Playing...")
            os.system("mpg123 " + file_path)

    def display_camera_feed(self, threshold1=120, threshold2=170, iterations=1):
        if(self.cam_init == True):
            # Display the real-time video stream generated by the USB camera.
            ret, img = self.camera.read()
            # Resize the captured frame depending on the given object detection model.
            mod_img = cv2.resize(img, self.frame_size)
            # Define the structuring element used for canny edge modifications.
            kernel = numpy.ones((4,4), numpy.uint8)
            # Apply the given image conversion option to the resized frame.
            if(self.con_opt == "normal"):
                self.latest_frame = mod_img
            if(self.con_opt == "gray"):
                self.latest_frame = cv2.cvtColor(mod_img, cv2.COLOR_BGR2GRAY)
            if(self.con_opt =="blur"):
                self.latest_frame = cv2.GaussianBlur(mod_img, (11,11), 0)
            if(self.con_opt =="canny"):
                self.latest_frame = cv2.Canny(mod_img, threshold1, threshold2)
            if(self.con_opt =="canny+"):
                canny_img = cv2.Canny(mod_img, threshold1, threshold2)
                self.latest_frame = cv2.dilate(canny_img, kernel, iterations=iterations)
            if(self.con_opt =="canny++"):
                canny_img = cv2.Canny(mod_img, threshold1, threshold2)
                self.latest_frame = cv2.dilate(canny_img, kernel, iterations=iterations+1)
            if(self.con_opt =="canny-"):
                canny_img = cv2.Canny(mod_img, threshold1, threshold2)
                canny_img = cv2.dilate(canny_img, kernel, iterations=iterations)
                self.latest_frame = cv2.erode(canny_img, kernel, iterations=iterations)
            if(self.con_opt =="canny--"):
                canny_img = cv2.Canny(mod_img, threshold1, threshold2)
                canny_img = cv2.dilate(canny_img, kernel, iterations=iterations)
                self.latest_frame = cv2.erode(canny_img, kernel, iterations=iterations+1)
            # Show the modified frame on the screen.
            cv2.imshow("Lab Assistant Camera Feed", self.latest_frame)
        else:
            s_img = cv2.imread("assets/camera_stopped.png")
            s_img = cv2.resize(s_img, self.frame_size)
            cv2.imshow("Lab Assistant Camera Feed", s_img)
        # Stop the camera feed if requested.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.camera.release()
            cv2.destroyAllWindows()
            print("\nCamera Feed Stopped!")
    
    def save_img_sample(self):
        # Create the file name for the image sample.
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        given_class = self.col_buttons_input.value.strip()
        filename = "./samples/IMG_{}_{}.jpg".format(given_class, date)
        # If requested, save the recently captured image (latest frame) with the applied conversion as a sample.
        cv2.imwrite(filename, self.latest_frame)
        # Notify the user after saving the sample.
        self.col_save_notify.value = "Latest Label: " + given_class
        print("\nSample Saved Successfully: " + filename)

    def change_img_conv(self, con_opt):
        # Change the image conversion option.
        self.con_opt = con_opt
        print("\nCamera Image Conversion ==> " + con_opt)
    
    def camera_stop(self, action):
        # Stop or resume the real-time camera feed.
        if(action == True):
            self.cam_init = False
            print("\nCamera Feed Stopped!")
            sleep(1)
        else:
            self.cam_init = True
            print("\nCamera Feed Resume!")
            sleep(1) 
    
    def camera_feed(self):
        # Start the camera feed loop.
        while True:
            self.display_camera_feed()

    def create_user_interface(self, appWidth=900, appHeight=600, b_set=["#FFE681", "#BA0C2E", 12, [6,10], "#BA0C2E"], _cursor="hand1"):
        # Design the user interface (GUI) via the guizero module.
        # And, disable the app window resizing to provide a better user experience.
        self.app = App(title="AI-driven Lab Assistant w/ ChatGPT", bg="#5C5B57", width=appWidth, height=appHeight)
        self.app.font = "Comic Sans MS" 
        self.app.tk.resizable(False, False)
        menubar = MenuBar(self.app, toplevel=["Edge Impulse Model", "Previous Lessons", "Help"], options=[[["Go to public model page", lambda:self.menu_com("inspect_m")], ["Inspect Enterprise Features", lambda:self.menu_com("inspect_e")]], [["Inspect", lambda:self.menu_com("pre_lesson")]], [["Project Tutorial", lambda:self.menu_com("o_tutorial")], ["ChatGPT", lambda:self.menu_com("h_chatgpt")]]])
        menubar.bg="#F9E5C9"
        menubar.tk.config(bg="#F9E5C9", fg="#5C5B57", activebackground="#F5F5F0", activeforeground="#5C5B57", cursor="plus")
        # Layout.
        app_header = Box(self.app, width=appWidth, height=50, align="top")
        app_header_text = Text(app_header, text="  ⬇️ Data Collection", color="white", size=20, align="left")
        app_header_text.tk.config(cursor="sb_down_arrow")
        app_header_text = Text(app_header, text="⬇️ Generate Lesson  ", color="white", size=20, align="right")
        app_header_text.tk.config(cursor="sb_down_arrow")
        app_data_collect = Box(self.app, width=appWidth/2, height=appHeight-50, layout="grid", align="left")
        app_run_model = Box(self.app, width=appWidth/2, height=appHeight-50, layout="grid", align="right")
        app_run_model.bg = "#215E7C"
        # User frame conversion configurations.
        conv_buttons = Text(app_data_collect, text="Conversions: ", color="#FFE681", size=14, grid=[0,0], align="left")
        conv_buttons_con = Box(app_data_collect, grid=[1,0], layout="auto", width="fill", height="fill", align="left")
        conv_buttons_con.tk.config(pady=8, padx=15)
        conv_buttons_con_1 = Box(conv_buttons_con, layout="auto", width="fill", height="fill", align="top")
        conv_buttons_con_1.tk.config(pady=4)
        conv_buttons_con_2 = Box(conv_buttons_con, layout="auto", width="fill", height="fill", align="top")
        conv_buttons_con_2.tk.config(pady=4)
        conv_buttons_con_3 = Box(conv_buttons_con, layout="auto", width="fill", height="fill", align="top")
        conv_buttons_con_3.tk.config(pady=4)
        conv_buttons = PushButton(conv_buttons_con_1, text="Normal", align="left", command=lambda:self.change_img_conv("normal"), padx=b_set[3][0], pady=b_set[3][1])
        conv_buttons.bg = b_set[0]
        conv_buttons.text_color = b_set[1]
        conv_buttons.text_size = b_set[2]
        conv_buttons.tk.config(highlightthickness=2, highlightbackground=b_set[4], highlightcolor=b_set[4], cursor=_cursor)
        conv_buttons = PushButton(conv_buttons_con_1, text="Gray", align="left", command=lambda:self.change_img_conv("gray"), padx=b_set[3][0], pady=b_set[3][1])
        conv_buttons.bg = b_set[0]
        conv_buttons.text_color = b_set[1]
        conv_buttons.text_size = b_set[2]
        conv_buttons.tk.config(highlightthickness=2, highlightbackground=b_set[4], highlightcolor=b_set[4], cursor=_cursor)
        conv_buttons = PushButton(conv_buttons_con_1, text="Blur", align="left", command=lambda:self.change_img_conv("blur"), padx=b_set[3][0], pady=b_set[3][1])
        conv_buttons.bg = b_set[0]
        conv_buttons.text_color = b_set[1]
        conv_buttons.text_size = b_set[2]
        conv_buttons.tk.config(highlightthickness=2, highlightbackground=b_set[4], highlightcolor=b_set[4], cursor=_cursor)
        conv_buttons = PushButton(conv_buttons_con_1, text="Canny", align="left", command=lambda:self.change_img_conv("canny"), padx=b_set[3][0], pady=b_set[3][1])
        conv_buttons.bg = b_set[0]
        conv_buttons.text_color = b_set[1]
        conv_buttons.text_size = b_set[2]
        conv_buttons.tk.config(highlightthickness=2, highlightbackground=b_set[4], highlightcolor=b_set[4], cursor=_cursor)
        conv_buttons = PushButton(conv_buttons_con_2, text="Canny+", align="left", command=lambda:self.change_img_conv("canny+"), padx=b_set[3][0], pady=b_set[3][1])
        conv_buttons.bg = b_set[0]
        conv_buttons.text_color = b_set[1]
        conv_buttons.text_size = b_set[2]
        conv_buttons.tk.config(highlightthickness=2, highlightbackground=b_set[4], highlightcolor=b_set[4], cursor=_cursor)
        conv_buttons = PushButton(conv_buttons_con_2, text="Canny++", align="left", command=lambda:self.change_img_conv("canny++"), padx=b_set[3][0], pady=b_set[3][1])
        conv_buttons.bg = b_set[0]
        conv_buttons.text_color = b_set[1]
        conv_buttons.text_size = b_set[2]
        conv_buttons.tk.config(highlightthickness=2, highlightbackground=b_set[4], highlightcolor=b_set[4], cursor=_cursor)
        conv_buttons = PushButton(conv_buttons_con_3, text="Canny-", align="left", command=lambda:self.change_img_conv("canny-"), padx=b_set[3][0], pady=b_set[3][1])
        conv_buttons.bg = b_set[0]
        conv_buttons.text_color = b_set[1]
        conv_buttons.text_size = b_set[2]
        conv_buttons.tk.config(highlightthickness=2, highlightbackground=b_set[4], highlightcolor=b_set[4], cursor=_cursor)
        conv_buttons = PushButton(conv_buttons_con_3, text="Canny--", align="left", command=lambda:self.change_img_conv("canny--"), padx=b_set[3][0], pady=b_set[3][1])
        conv_buttons.bg = b_set[0]
        conv_buttons.text_color = b_set[1]
        conv_buttons.text_size = b_set[2]
        conv_buttons.tk.config(highlightthickness=2, highlightbackground=b_set[4], highlightcolor=b_set[4], cursor=_cursor)
        # User camera configurations.
        cam_buttons = Text(app_data_collect, text="Camera Feed: ", color="#FFE681", size=14, grid=[0,2], align="left")
        cam_buttons_con = Box(app_data_collect, grid=[1,2], layout="auto", width="fill", height="fill", align="right")
        cam_buttons_con.bg = "#364935"
        cam_buttons_con.tk.config(padx=15,pady=15)
        cam_buttons_con.set_border(3, "white")
        cam_buttons = PushButton(cam_buttons_con, text="Start", align="left", command=lambda:self.camera_stop(False), padx=b_set[3][0]+15, pady=b_set[3][1])
        cam_buttons.bg = b_set[0]
        cam_buttons.text_color = b_set[1]
        cam_buttons.text_size = b_set[2]
        cam_buttons.tk.config(highlightthickness=2, highlightbackground=b_set[4], highlightcolor=b_set[4], cursor=_cursor)
        cam_buttons = PushButton(cam_buttons_con, text="Stop", align="right", command=lambda:self.camera_stop(True), padx=b_set[3][0]+15, pady=b_set[3][1])
        cam_buttons.bg = b_set[0]
        cam_buttons.text_color = b_set[1]
        cam_buttons.text_size = b_set[2]
        cam_buttons.tk.config(highlightthickness=2, highlightbackground=b_set[4], highlightcolor=b_set[4], cursor=_cursor)
        # Data collection configurations.
        col_buttons = Text(app_data_collect, text="", color="black", size=10, grid=[0,3], align="left")
        col_buttons = Text(app_data_collect, text="Assign Label:", color="#FFE681", size=14, grid=[0,4], align="left")
        col_buttons_con = Box(app_data_collect, grid=[1,4], layout="grid", width="fill", height="fill", align="right")
        col_buttons_con.bg = "#364935"
        col_buttons_con.tk.config(padx=3, pady=10, highlightbackground="white", highlightcolor="white")
        col_buttons_con.set_border(5, "white")
        self.col_buttons_input = TextBox(col_buttons_con, grid=[0,0], width=20, height=3, multiline=True, text="Please enter a label...", scrollbar=False)
        self.col_buttons_input.bg = "#5C5B57"
        self.col_buttons_input.text_size = 15
        self.col_buttons_input.text_color = "white"
        self.col_buttons_input.tk.config(highlightthickness=3, highlightbackground="white", highlightcolor="white")
        col_buttons = Text(col_buttons_con, text="", color="black", size=5, grid=[0,1], align="left")
        col_buttons = PushButton(col_buttons_con, text="Save Sample", grid=[0,2], align="right", command=self.save_img_sample, padx=b_set[3][0]+15, pady=b_set[3][1])
        col_buttons.bg = b_set[1]
        col_buttons.text_color = b_set[0]
        col_buttons.text_size = b_set[2]
        col_buttons.tk.config(highlightthickness=3, highlightbackground=b_set[0], highlightcolor=b_set[0], cursor=_cursor)
        col_buttons = Text(col_buttons_con, text="", color="black", size=5, grid=[0,3], align="left")
        col_buttons = PushButton(col_buttons_con, text="Inspect Samples", grid=[0,4], align="right", command=lambda:self.menu_com("inspect_samples"), padx=b_set[3][0]+15, pady=b_set[3][1])
        col_buttons.bg = b_set[1]
        col_buttons.text_color = b_set[0]
        col_buttons.text_size = b_set[2]
        col_buttons.tk.config(highlightthickness=3, highlightbackground=b_set[0], highlightcolor=b_set[0], cursor=_cursor)
        self.col_save_notify = Text(col_buttons_con, text="Latest Label: Waiting...", color=b_set[0], size=10, grid=[0,6], align="right")
        self.col_save_notify.tk.config(pady=3)
        # Inquiry configurations.
        inquiry_buttons = Text(app_run_model, text="Inquiry:", color="orange", size=14, grid=[0,0], align="left")
        inquiry_buttons_con = Box(app_run_model, grid=[1,0], layout="auto", width=310, height=80, align="right")
        self.inquiry_input = Combo(inquiry_buttons_con, align="right", options=["How to use [...] in labs?", "Please list the lab experiments with [...]", "Tell me about recent research papers on [...]", "How to repair [...]?"])
        self.inquiry_input.bg = "#F9E5C9"
        self.inquiry_input.text_color = "#5C5B57"
        self.inquiry_input.text_size = 8
        self.inquiry_input.tk.config(cursor="question_arrow")
        # Audio player configurations.
        audio_buttons = Text(app_run_model, text="Audio Player: ", color="orange", size=14, grid=[0,1], align="left")
        self.audio_input = CheckBox(app_run_model, text="Activate Instant Play", grid=[1,1], align="right")
        self.audio_input.text_size = 13
        self.audio_input.text_color = "#F9E5C9"
        self.audio_input.tk.config(highlightthickness=0, pady=20, cursor="exchange")
        # Edge Impulse object detection model configurations.
        model_buttons = Text(app_run_model, text="", color="black", size=22, grid=[0,2], align="left")
        list_button = PushButton(app_run_model, grid=[0,3], align="left", command=lambda:self.menu_com("filters"), padx=b_set[3][0], pady=b_set[3][1])
        list_button.image = "assets/lab_icon.png"
        list_button.tk.config(highlightthickness=5, highlightbackground="#F9E5C9", highlightcolor="#F9E5C9", cursor=_cursor)
        model_buttons_con = Box(app_run_model, grid=[1,3], layout="grid", width="fill", height="fill", align="right")
        model_buttons_con.bg = "#9BB5CE"
        model_buttons_con.set_border(5, "#F9E5C9")
        model_buttons_con.tk.config(padx=20, pady=30)
        run_button = PushButton(model_buttons_con, text="Run Inference", grid=[0,0], align="right", command=lambda:self.run_inference(notify=True), padx=b_set[3][0], pady=b_set[3][1], width=19, height=3)
        run_button.bg = "orange"
        run_button.text_color = "#5C5B57"
        run_button.text_size = b_set[2]+2
        run_button.tk.config(highlightthickness=6, highlightbackground="#5C5B57", highlightcolor="#5C5B57", cursor=_cursor)
        self.model_notify = Text(model_buttons_con, text="Latest Detection: " + self.detected_class, color="#5C5B57", size=10, grid=[0,1], align="right")
        model_buttons = Text(model_buttons_con, text="", color="black", size=20, grid=[0,2], align="left")
        model_buttons = PushButton(model_buttons_con, text="Get ChatGPT Response", grid=[0,3], align="right", command=self.chatgpt_show_information, padx=b_set[3][0], pady=b_set[3][1], width=19, height=3)
        model_buttons.bg = "orange"
        model_buttons.text_color = "#5C5B57"
        model_buttons.text_size = b_set[2]+2
        model_buttons.tk.config(highlightthickness=6, highlightbackground="#5C5B57", highlightcolor="#5C5B57", cursor=_cursor)
        # Create the second window to display the information generated by ChatGPT.
        self.second_window = Window(self.app, title="Response from ChatGPT", bg="#5C5B57", width=appWidth, height=appHeight, layout= "grid")
        self.second_window.tk.resizable(False, False)
        self.second_window.hide()
        second_w_logo = Picture(self.second_window, image="assets/chatgpt_logo.png", width=200, height=200, grid=[0,0], align="top")
        second_w_logo.tk.config(padx=15, pady=15, highlightthickness=6, highlightbackground="white", highlightcolor="white")
        second_w_show_con = Box(self.second_window, width="fill", height="fill", grid=[1,0], layout="auto", align="left")
        second_w_show_con.tk.config(padx=25, pady=15)
        self.second_w_text = TextBox(second_w_show_con, width=62, height=29, multiline=True, text="Waiting ChatGPT Response...", scrollbar=False)
        self.second_w_text.bg = "#74AA9C"
        self.second_w_text.text_size = 12
        self.second_w_text.text_color = "white"
        self.second_w_text.tk.config(padx=10, pady=10, highlightthickness=6, highlightbackground="white", highlightcolor="white", cursor="target")
        
    def menu_com(self, com="help"):
        # Define the button commands.
        if(com == "inspect_samples"):
            subprocess.Popen(["xdg-open", "./samples"])
        if(com == "filters"):
            info("EI Model Label Filters", "spoon_spatula => Canny++\n\nforcep => Canny-\n\ndynamometer => Canny+\n\nbunsen_burner => Gray\n\nalcohol_burner => Blur\n\ntest_tube => Canny-\n\nskeleton_model => Canny\n\nmicroscope => Canny\n\nhatchery => Canny\n\nmicroscope_slide => Canny--")
        # Define the menubar features.
        if(com == "inspect_m"):
            webbrowser.open("https://edgeimpulse.com/")
        if(com == "inspect_e"):
            webbrowser.open("https://docs.edgeimpulse.com/docs/edge-impulse-studio/organizations/data-campaigns")
        if(com == "pre_lesson"):
            subprocess.Popen(["xdg-open", "./audio_lesson"])
        if(com == "o_tutorial"):
            webbrowser.open("https://www.hackster.io/kutluhan-aktar")
        if(com == "h_chatgpt"):
            webbrowser.open("https://platform.openai.com/docs/api-reference/introduction")
            
    def show_interface(self):
        # Show the designed user interface (GUI) on the screen.
        self.app.display()