import os
import threading 
th=[0]*2
import time
def func( i,idm):

   #os.system(")
   #screen -dmS  44 &&  screen -r 44 -xX stuff  time python3 /home/rahul/vegetable_grading/amazon_now/ch.py
   #screen -dmS j && screen -r j -xX stuff "time python3 /home/rahul/vegetable_grading/amazon_now/ch.py"
   #time.sleep(2)
   #temp=" time python3 /home/rahul/vegetable_grading/amazon_now/ch.py"
   #sos.system("
   #time python3 /home/deploy/vegetable_grading/amazon_now/mask_rcnn_grading1.py  splash --category=potato --client_name=amazon_now --response=mask_rcnn --grade_id=rh97  --image=/home/deploy/vegetable_grading/amazon_now/potato/potato__6th_copy_.jpg
   #os.system()
   #os.system("screen -dmS rahulha"+str(i))
   #fina="screen -r rahulha"+str(i)+"  -xX stuff 'time python3 /home/rahul/vegetable_grading/amazon_now/mask_rcnn_grading_rahultest.py  splash --category=onion --client_name=amazon_now --response=mask_rcnn --grade_id='"+(idm) +"' --image=/home/rahul/Downloads/potato__6th_copy_.jpg \n'"
   fina="time python3 /home/rahul/vegetable_grading/amazon_now/mask_rcnn_grading_rahultest.py  splash --category=onion --client_name=amazon_now --response=mask_rcnn --grade_id='"+(idm) +"' --image=/home/rahul/vegetable_grading/amazon_now/DCCBB200203O01.jpg \n"
   os.system(fina)
   

   #os.system(fina)
   #os.system(fina)
   # os.wait()


  # print(os.system()
   #print(fina))


   #print(scr)
   #os.system(scr)

# #print("eee")
tt=[]
for i in range(2):
 st=time.time()
 func(i,"rrr"+str(i))
 ft= time.time()
 tt.append(ft-st)
print(tt)
 

    
