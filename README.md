## Package
package cần để điều khiển tay máy
```bash
sudo apt-get install ros-noetic-effort-controllers
```

## ▶️ Chạy mô phỏng Gazebo  
Chạy lệnh sau để mở mô phỏng trong Gazebo:  
```bash
roslaunch slam_car gazebo.launch
```  
## 🖥️ Mở RViz  
Em đã rất cố gắng launch 2 cái này lại cùng lúc nhưng mà lực bất tòng tâm 😢  
Mở một terminal mới:  
```bash
roslaunch slam_car display.launch
```
  
## 🤖 Chạy tay máy  
Em chạy file `.bash` điều khiển hai tay máy quay trong vòng lặp cho dễ nhìn. Vẫn có thể gửi từng lệnh một để điều khiển từng tay máy tới vị trí mong muốn.  
Mở một terminal khác:  
```bash
cd catkin_ws/src/slam_car
./move_arm_loop.bash
```
Sẽ thấy tay máy quay trong vòng lặp 3 giây.  
![Tay máy quay](https://github.com/user-attachments/assets/c4d5dd5c-e602-4ae0-b9ff-58b739761214)  
  
## 🕹️ Điều khiển xe  
Mở một terminal mới, chạy lệnh:  
```bash
rosrun teleop_twist_keyboard teleop_twist_keyboard.py
```
![Điều khiển bằng bàn phím](https://github.com/user-attachments/assets/379a2d54-ca0f-48ff-bebe-7956305798ad)  
  
## 🛰️ Check /scan – vật cản trong map  
Có thể tắt IMU để quan sát rõ hơn. RViz sẽ hiển thị vật cản khi robot quét.  
![Vật cản](https://github.com/user-attachments/assets/f766193f-22f8-4faf-bb19-b18b39cf7609)  
  
## 🧭 IMU  
![IMU](https://github.com/user-attachments/assets/1f45c951-d6d4-4c83-8b83-15105de65590)  
  
## 🔁 TF động  
Khi robot di chuyển, TF xoay theo chuyển động bánh xe và tay máy. Tay máy lúc này vẫn đang chạy trong vòng lặp.  
![TF](https://github.com/user-attachments/assets/337cef32-7ac4-47c2-98d5-8f79e3b72bf9)

  
## 📍 Encoder  
Chạy lệnh sau để kiểm tra dữ liệu vị trí từ encoder:  
```bash
rostopic echo /odom
```
Vì tốc độ xuất dữ liệu rất nhanh nên anh hãy nhấn `Ctrl + C` sớm, sau đó xem kết quả in ra.  
![Odom](https://github.com/user-attachments/assets/ef73d93d-755c-41e1-9fab-8192c79e2894)  
Robot cách `fixed_frame` khoảng:  
x: 2.1352531397559322  
y: 0.19274187436668686  
z: -0.2249728070064274  
Khá là chuẩn. Z = -0.22 mà không phải 0 là vì khi em vẽ trong SolidWorks, em không đặt đúng vào gốc toạ độ mà vẽ lệch xuống dưới, nên mới bị -0.22. Mong anh bỏ qua :v



>>>>>>> Cập nhật README mô tả chạy mô phỏng và điều khiển robot
