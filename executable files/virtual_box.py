
import cv2
import numpy as np 
import math
from PIL import Image,ImageDraw,ImageFont
import pyaudio,struct,math
import sys

cascPath = 'haarcascade_frontalface_default.xml'
eyePath = 'haarcascade_eye.xml'
nosePath = 'haarcascade_mcs_nose.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eyePath)
noseCascade = cv2.CascadeClassifier(nosePath)
alpha=.5
first_time=True
RATE=16000
# audio parameters settings

def clip16(x):    
    # Clipping for 16 bits
    if x > 32767:
        x = 32767
    elif x < -32768:
        x = -32768
         
    return x

############am
f1=400
oma =2.0 * math.pi * f1 / RATE
theta=0
counter=0
############vibrato
f0 = 2
W = 0.2
buffer_MAX =  1024                          # Buffer length
buffer = [0.0 for i in range(buffer_MAX)]   # Initialize to zero
kr = 0  # read index
kw = int(0.5 * buffer_MAX)  # write index (initialize to middle of buffer)
kw = buffer_MAX/2


def my_callback(input_string, block_size, time_info, status):
	
	if effect == 'am':
		global theta 
		global oma
		output_sample=[0.0 for i in range(block_size)]

		in_sample = struct.unpack('h'*block_size, input_string)
		for i in range(block_size):
			output_sample[i]=in_sample[i]*math.cos(theta)
			theta=theta+oma
		if theta>math.pi:
			theta=theta - 2*math.pi
		output_string=struct.pack('h'*block_size,*output_sample)

	elif effect == 'vibrato':
		global counter,RATE
		global buffer
		global W,f0,kw,kr
		# Buffer (delay line) indices
		output_value=[0.0 for i in range(block_size)]
		input_value=struct.unpack('h'*block_size,input_string)
		for n in range(0, block_size):
			kr_prev = int(math.floor(kr))               
	        kr_next = kr_prev + 1
	        frac = kr - kr_prev    # 0 <= frac < 1
	        if kr_next >= buffer_MAX:
	            kr_next = kr_next - buffer_MAX

	        # Compute output value using interpolation
	        output_value[n] = (1-frac) * buffer[kr_prev] + frac * buffer[kr_next]
	        output_value[n]=int(output_value[n])

	        # Update buffer (pure delay)
	        buffer[int(kw)] = input_value[n]

	        # Increment read index
	        kr = kr + 1 + W * math.sin( 2 * math.pi * f0 * (n+(counter*block_size)) / RATE )
	            # Note: kr is fractional (not integer!)

	        # Ensure that 0 <= kr < buffer_MAX
	        if kr >= buffer_MAX:
	            # End of buffer. Circle back to front.
	            kr = 0

	        # Increment write index    
	        kw = kw + 1
	        if kw == buffer_MAX:
	            # End of buffer. Circle back to front.
	            kw = 0
		output_string = struct.pack('h'*block_size, *output_value)
	counter=counter+1
	return (output_string, pyaudio.paContinue)


def assign():
	global FRAME_YLIM,FRAME_XLIM
	global overlay_img_scaled
	FRAME_YLIM = frame.shape[0] - overlay_img_scaled.size[1] / 2
	FRAME_XLIM = frame.shape[1] - overlay_img_scaled.size[0] / 2 # cv2.shape and PIL.size give coordinates reversed

def coordinates(event,x,y,flags,param):
	global MOUSE_X,MOUSE_Y
	MOUSE_X = x
	MOUSE_Y = y


def eye_center_location(eyes,faces):#=([100,100,200,200],[400,100,600,200])):
	global x_center , y_center
	global file_name
	if file_name=='specs1.png':
		if(len(eyes) >= 2):
			x1=eyes[0][0]
			y1=eyes[0][1]
			e1c=[ x1+eyes[0][2]/2 , y1+eyes[0][3]/2 ]
			x2=eyes[1][0]
			y2=eyes[1][1]
			e2c=[ x2+eyes[1][2]/2 , y2+eyes[1][3]/2 ]
			eyes_center = [(e1c[0]+e2c[0])/2,e1c[1]]
			x_center =faces[0][0] + eyes_center[0]
			y_center =faces[0][1] + eyes_center[1]
			
	elif file_name == 'vader.png':
		if(len(eyes) >= 2):
			x1=eyes[0][0]
			y1=eyes[0][1]
			e1c=[ x1+eyes[0][2]/2 , y1+eyes[0][3]/2 ]
			x2=eyes[1][0]
			y2=eyes[1][1]
			e2c=[ x2+eyes[1][2]/2 , y2+eyes[1][3]/2 ]
			eyes_center = [(e1c[0]+e2c[0])/2,e1c[1]]
			x_center =faces[0][0] + eyes_center[0]
			y_center =faces[0][1] + eyes_center[1] + 20
			
	elif file_name=='fox.png':
		if(len(eyes) >= 2):
			x1=eyes[0][0]
			y1=eyes[0][1]
			e1c=[ x1+eyes[0][2]/2 , y1+eyes[0][3]/2 ]
			x2=eyes[1][0]
			y2=eyes[1][1]
			e2c=[ x2+eyes[1][2]/2 , y2+eyes[1][3]/2 ]
			eyes_center = [(e1c[0]+e2c[0])/2,e1c[1]]
			x_center =faces[0][0] + eyes_center[0] -20
			y_center =faces[0][1] + eyes_center[1] - 30
	elif file_name == 'dog.png':
		if(len(eyes) >= 2):
			x1=eyes[0][0]
			y1=eyes[0][1]
			e1c=[ x1+eyes[0][2]/2 , y1+eyes[0][3]/2 ]
			x2=eyes[1][0]
			y2=eyes[1][1]
			e2c=[ x2+eyes[1][2]/2 , y2+eyes[1][3]/2 ]
			eyes_center = [(e1c[0]+e2c[0])/2,e1c[1]]
			x_center =faces[0][0] + eyes_center[0] 
			y_center =faces[0][1] + eyes_center[1] +40
			



def reshape_overlay_img(eyes,faces):
	global overlay_img_scaled
	global file_name
	if file_name == 'specs1.png':
		if len(eyes) >=2:
			new_overlay_width = faces[0][2]
			new_overlay_height = eyes[0][3] 
			if new_overlay_height < 50 :
				new_overlay_height = 50
			if new_overlay_width < 50:
				new_overlay_width = 50
			overlay_img_scaled = overlay_img.resize((new_overlay_width, new_overlay_height), Image.ANTIALIAS)
	elif file_name == 'vader.png':
		if len(eyes) >=2:
			new_overlay_width = faces[0][2] +100
			new_overlay_height = faces[0][3] +175
			if new_overlay_height < 50 :
				new_overlay_height = 50
			if new_overlay_width < 50:
				new_overlay_width = 50
			overlay_img_scaled = overlay_img.resize((new_overlay_width, new_overlay_height), Image.ANTIALIAS)
	elif file_name == 'fox.png':
		if len(eyes) >=2:
			new_overlay_width = faces[0][2]
			new_overlay_height = faces[0][3] +100
			if new_overlay_height < 50 :
				new_overlay_height = 50
			if new_overlay_width < 50:
				new_overlay_width = 50
			overlay_img_scaled = overlay_img.resize((new_overlay_width, new_overlay_height), Image.ANTIALIAS)
	elif file_name == 'dog.png':
		if len(eyes) >=2:
			new_overlay_width = faces[0][2]
			new_overlay_height = faces[0][3] + 250
			if new_overlay_height < 50 :
				new_overlay_height = 50
			if new_overlay_width < 50:
				new_overlay_width = 50
			overlay_img_scaled = overlay_img.resize((new_overlay_width, new_overlay_height), Image.ANTIALIAS)


FRAME_XLIM = 0
FRAME_YLIM = 0

x_center = 500
y_center = 500 

cap=cv2.VideoCapture(0)
cv2.namedWindow('live_feed')

file_name=sys.argv[1]
effect = sys.argv[2]
overlay_img = Image.open(file_name)
overlay_img_scaled = overlay_img.resize((300,300), Image.ANTIALIAS) # initial size

WIDTH=2
CHANNELS=1
p = pyaudio.PyAudio()
stream = p.open(format = p.get_format_from_width(WIDTH),
                channels = CHANNELS,
                rate = RATE,
                input = True,
                output = True,
                frames_per_buffer = 1,
                stream_callback = my_callback)

while(True):
	
	ok,frame=cap.read()
	frame = cv2.flip(frame,1)
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	overlay=frame.copy()
	faces = faceCascade.detectMultiScale(
    	gray, # the variable name that holds the image
    	scaleFactor=1.1,
    	minNeighbors=10, # sdjusted this to detect the correct number of faces
    	minSize=(30, 30),
    	flags = cv2.CASCADE_SCALE_IMAGE
	)
	

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
	    # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # th ex and y's are the location of the rectangle top left corner
	    # cv2.putText()
	    roi_gray = gray[y : y+h, x : x+h]

	    eyes = eyeCascade.detectMultiScale(roi_gray,1.1,3) # change the extra parameters to get better results

	    
	    eye_center_location(eyes,faces)
	    reshape_overlay_img(eyes,faces)
	    assign()
	    
	    
	    for (ex,ey,ew,eh) in eyes:
	    	i=0
	    	# cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew ,y+ey+eh ),(255,0,0),2) # adding x and y because the ex and ey are just the coordinates within the facial rectangle
	    	
	    	i=i+1
	    	# So the eye rectangle have to be offset by that facial rectangle to draw around the eyes																	
	if y_center >= FRAME_YLIM:
		
		y_center = FRAME_YLIM
	elif y_center <= overlay_img_scaled.size[1]/2:
		
		y_center = overlay_img_scaled.size[1]/2

	if x_center >= FRAME_XLIM:
		x_center = FRAME_XLIM
	elif x_center <= overlay_img_scaled.size[0]/2:
		
		x_center = overlay_img_scaled.size[0]/2
	
	
	
	cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
	bg = Image.fromarray(cv2image)
	text_img = Image.new('RGBA', (frame.shape[1],frame.shape[0]), (0, 0, 0, 0))
	text_img.paste(bg, (0,0))
	text_img.paste(overlay_img_scaled, (int(x_center) - int(overlay_img_scaled.size[0]/2),int(y_center) - int(overlay_img_scaled.size[1]/2) ), mask=overlay_img_scaled)
	text_img=text_img.convert('RGB')
	img_f = np.array(text_img)
	img_f = img_f[:, :, ::-1].copy()
	
	cv2.imshow('live_feed',img_f)
	if cv2.waitKey(1) == ord('q'):
		break

stream.stop_stream()
stream.close()
p.terminate()
cap.release()
cv2.destroyAllWindows()