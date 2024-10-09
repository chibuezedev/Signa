def video_loop(self):
    try:
        ok, frame = self.vs.read()
        if not ok:
            print("Failed to grab frame")
            return

        cv2image = cv2.flip(frame, 1)
        hands = self.hd.findHands(cv2image, draw=False, flipType=False)

        if hands:
            hand = hands[0]
            if 'bbox' in hand:
                x, y, w, h = hand['bbox']

                # Extract hand region
                image = cv2image[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

                # Process hand region (your existing logic here)
                white = cv2.imread("white.jpg")
                handz = self.hd2.findHands(image, draw=False, flipType=False)

                if handz:
                    hand = handz[0]
                    self.pts = hand['lmList']

                    # Drawing logic (kept from your original code)
                    os = ((400 - w) // 2) - 15
                    os1 = ((400 - h) // 2) - 15
                    for t in range(0, 4, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1),
                                 (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(5, 8, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(9, 12, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(13, 16, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(17, 20, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    cv2.line(white, (self.pts[5][0] + os, self.pts[5][1] + os1), (self.pts[9][0] + os, self.pts[9][1] + os1), (0, 255, 0),
                             3)
                    cv2.line(white, (self.pts[9][0] + os, self.pts[9][1] + os1), (self.pts[13][0] + os, self.pts[13][1] + os1), (0, 255, 0),
                             3)
                    cv2.line(white, (self.pts[13][0] + os, self.pts[13][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1),
                             (0, 255, 0), 3)
                    cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[5][0] + os, self.pts[5][1] + os1), (0, 255, 0),
                             3)
                    cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1), (0, 255, 0),
                             3)

                    for i in range(21):
                        cv2.circle(white, (self.pts[i][0] + os, self.pts[i][1] + os1), 2, (0, 0, 255), 1)


                    res = white
                    self.predict(res)

                    self.current_image2 = Image.fromarray(res)
                    imgtk = ImageTk.PhotoImage(image=self.current_image2)
                    self.panel2.imgtk = imgtk
                    self.panel2.config(image=imgtk)

        # Update main panel
        cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
        self.current_image = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=self.current_image)
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)

        # Update other UI elements
        self.panel3.config(text=self.current_symbol, font=("Courier", 30))
        self.b1.config(text=self.word1, font=("Courier", 20), wraplength=825, command=self.action1)
        self.b2.config(text=self.word2, font=("Courier", 20), wraplength=825, command=self.action2)
        self.b3.config(text=self.word3, font=("Courier", 20), wraplength=825, command=self.action3)
        self.b4.config(text=self.word4, font=("Courier", 20), wraplength=825, command=self.action4)
        self.panel5.config(text=self.str, font=("Courier", 30), wraplength=1025)

    except Exception as e:
        print(f"Error in video_loop: {str(e)}")
        print(traceback.format_exc())

    finally:
        self.root.after(1, self.video_loop)