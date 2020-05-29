##how cloaking works in general: 1.create a bg image 2. input color to remove using
##hue on hsv format 3. create a mask specifically for those values using inRange
##function 4. bitwise_and mask with bg to get disappearing effect for only blue parts,
##5. bitwise_and video feed with the bitwise_not of mask 6.superimpose the results
##in step 4 and step 5.(final effect: only blue parts will show the bg, and other
##parts will show the video feed)

import cv2
import numpy as np

cap = cv2.VideoCapture(0)


ret, bg = cap.read()    #capture background image

cv2.imshow("bg",bg)
template = cv2.imread("my_face.jpg")

while (cap.isOpened()):
    #print out size of feed
##    width  = cap.get(3) # 640 width  
##    height = cap.get(4) # 480 height
##    print('width, height:', width, height) #note: alt 3 for multiline comment

    #read images from feed
    ret2, cap2 = cap.read()
    hsv = cv2.cvtColor(cap2,cv2.COLOR_BGR2HSV)  #use hsv for color segmentation,bgr not accurate
    #color to mask
    lower_blue = np.array([86,89,133])   #converting list to np.array for image manipulation
    upper_blue = np.array([127,255,255])#note:photo manipulation requires np.array, list won't work

    #using values above to create mask
    mask1 = cv2.inRange(hsv,lower_blue,upper_blue) #anything within this range will be converted to white
    
    #inverse and multiply by running feed to actively 'remove' blue parts
    mask_inv = cv2.bitwise_not(mask1)
    new_feed = cv2.bitwise_and(cap2,cap2,mask =mask_inv) #this step will show the cam feed without any blue color note: repeat src1 and src2 values if using mask
    blue_visible = cv2.bitwise_and(bg,bg,mask = mask1) #this step makes the background only visible when covered by blue
    #qn: why does error occur with mask in src2 slot? ans: because mask has different array config compared to video feed! 
    
    #superimposing blue_visible(bg shown where blue present) and new_feed(no blue color)
    final = cv2.addWeighted(blue_visible,1,new_feed,1,0)    #final: w=640,h=480


    #**TEMPLATE MATCHING!!**

    result = cv2.matchTemplate(cap2,template,cv2.TM_CCOEFF_NORMED) #this gives ghost template img(white part for most match)
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(result) #this gives coords of brightest part of template (2x2)

    #changing coordinates of max_loc(since its off centre? not sure why)
    max_loc = tuple([x+100 for x in list(max_loc)])
    
##    max_loc = list(max_loc)
##    max_loc = [x+100 for x in max_loc]  #iterates through values of max_loc and then adds 100
##    max_loc=tuple(max_loc) #conversion back to tuple
    
    zeros = np.zeros([480,640,3],'uint8')   #create a zeros background
    circle = cv2.circle(zeros,max_loc,150,(255,0,0),5) #drawing circle on black bg
    final2 = cv2.addWeighted(circle,1,final,1,0)    #superimposing circle img with cloak feed
    cv2.imshow("circle",circle)

    #viewing video results
    cv2.imshow("Result",result)    
    cv2.imshow("blue_visible",blue_visible)    
    cv2.imshow("Final2",final2)

    print(max_loc)
    print(final.shape)
    print(circle.shape)

    
    
    
    ch = cv2.waitKey(1)             #note that if these 3 lines are not present, imshow will stall
    if ch & 0xFF ==ord('q'):
        break
    

        
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
