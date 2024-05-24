'''
import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import time

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('1.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
tracker = Tracker()
vh_down = {}
counter = []
vh_up = {}
counter1 = []
vh_dtime = {}
vh_utime = {}
cy1 = 322
cy2 = 368
offset = 6

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1020, 500))

def display_speed(frame, cx, cy, vehicle_id, counter_list, elapsed_time, x, y):
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    cv2.putText(frame, str(vehicle_id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    if vehicle_id not in counter_list:
        counter_list.append(vehicle_id)

    distance = 10

    if elapsed_time > 0:
        speed_ms = distance / elapsed_time
        speed_km = speed_ms * 3.6
        cv2.putText(frame, str(int(speed_km)) + ' Km/h', (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(frame, 'Speed N/A', (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list = []

    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        c = class_list[d]

        if c in ['car', 'truck', 'bus']:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2

        if cy1 - offset < cy < cy1 + offset:
            vh_down[id] = cy
            vh_dtime[id] = time.time()

        if cy2 - offset < cy < cy2 + offset:
            vh_up[id] = cy
            vh_utime[id] = time.time()

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 1)
    

        if id in vh_down:
            elapsed_time = time.time() - vh_dtime[id]
            display_speed(frame, cx, cy, id, counter, elapsed_time, x4, y4)
        if id in vh_up:
            elapsed_time2 = time.time() - vh_utime[id]
            display_speed(frame, cx, cy, id, counter1, elapsed_time2, x4, y4)

    # Adjust cy1 and cy2 to increase the gap between the lines

    cv2.line(frame, (267, cy1), (829, cy1), (255, 255, 255), 1)
    cv2.putText(frame, '1line', (274, 318), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.line(frame, (167, cy2), (932, cy2), (255, 255, 255), 1)
    cv2.putText(frame, '2line', (181, 363), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)


   # cv2.putText(frame, 'GoingDown: ' + str(len(counter)), (60, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
   # cv2.putText(frame, 'GoingUp: ' + str(len(counter1)), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

    out.write(frame)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()    '''
#......................................................................................

import os
import sys

# Redirect stdout and stderr to /dev/null (Unix-like systems) or nul (Windows)
with open(os.devnull, 'w') as devnull:
    # Save the original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Redirect stdout and stderr to /dev/null (Unix-like systems) or nul (Windows)
    sys.stdout = devnull
    sys.stderr = devnull

    # Write your code here
    import cv2
    import pandas as pd
    from ultralytics import YOLO
    from tracker import Tracker
    import time

    model = YOLO('yolov8s.pt')

    def RGB(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            colorsBGR = [x, y]
            print(colorsBGR)

    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB', RGB)

    cap = cv2.VideoCapture('vehicles.mp4')

    my_file = open("coco.txt", "r")
    data = my_file.read()
    class_list = data.split("\n")

    count = 0
    tracker = Tracker()
    vh_down = {}
    counter = []
    vh_up = {}
    counter1 = []
    vh_dtime = {}
    vh_utime = {}
    cy1 = 322
    cy2 = 368
    offset = 6

    vehiclecount = 0  # Initialize vehicle count

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1020, 500))

    #dataFrame creation
    #vehicle_data = pd.DataFrame(columns=['Vehicle ID','Speed(km/h)'])
    vehicle_data = pd.DataFrame(columns=['Vehicle ID', 'Speed(km/h)', 'Vehicle Type'])

    def display_speed(frame, cx, cy, vehicle_id, counter_list, elapsed_time, x, y, vehicle_type):
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        

        if vehicle_id not in counter_list:
            counter_list.append(vehicle_id)
            global vehiclecount
            vehiclecount += 1  # Increase vehicle count for new vehicle ID
        cv2.putText(frame, str(vehiclecount), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        distance = 10

        if elapsed_time > 0:
            speed_ms = distance / elapsed_time
            speed_km = speed_ms * 3.6
            cv2.putText(frame, str(int(speed_km)) + ' Km/h', (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            #Add data to DataFrame
            #vehicle_data.loc[len(vehicle_data)]=[vehiclecount,speed_km]
            vehicle_data.loc[len(vehicle_data)] = [vehiclecount, speed_km, vehicle_type]
            
        else:
            cv2.putText(frame, 'Speed N/A', (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame, (1020, 500))

        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        list = []

        for index, row in px.iterrows():
            x1, y1, x2, y2, _, d = map(int, row)
            c = class_list[d]
            cv2.putText(frame, c, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            if c in ['car', 'truck', 'bus']:
                list.append([x1, y1, x2, y2])
                vehicle_type = c  # Store the vehicle type

        bbox_id = tracker.update(list)
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx = (x3 + x4) // 2
            cy = (y3 + y4) // 2

            if cy1 - offset < cy < cy1 + offset:
                vh_down[id] = cy
                vh_dtime[id] = time.time()

            if cy2 - offset < cy < cy2 + offset:
                vh_up[id] = cy
                vh_utime[id] = time.time()

            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 1)
        

            if id in vh_down:
                elapsed_time = time.time() - vh_dtime[id]
                #display_speed(frame, cx, cy, id, counter, elapsed_time, x4, y4)
                display_speed(frame, cx, cy, id, counter, elapsed_time, x4, y4, vehicle_type)
            if id in vh_up:
                elapsed_time2 = time.time() - vh_utime[id]
            #    display_speed(frame, cx, cy, id, counter1, elapsed_time2, x4, y4)

        cv2.putText(frame, f'Vehicle Count: {vehiclecount}', (60, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
    ## first line
        cv2.line(frame, (267, cy1), (829, cy1), (255, 255, 255), 1)
        cv2.putText(frame, '1line', (274, 318), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    ## second line
        cv2.line(frame, (167, cy2), (932, cy2), (255, 255, 255), 1)
        cv2.putText(frame, '2line', (181, 363), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)


    # cv2.putText(frame, 'GoingDown: ' + str(len(counter)), (60, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
    # cv2.putText(frame, 'GoingUp: ' + str(len(counter1)), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

        out.write(frame)

        cv2.imshow("RGB", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    vehicle_data.to_csv('vehicle_data.csv',index=False)



   # Read the CSV file into a DataFrame
    df = pd.read_csv('vehicle_data.csv')

    # Group the DataFrame by 'Vehicle ID' and calculate sum and count of speeds for each group
    grouped = df.groupby('Vehicle ID').agg({'Speed(km/h)': ['sum', 'count'], 'Vehicle Type': 'first'}).reset_index()

    # Calculate average speed for each group
    grouped['Average Speed'] = grouped[('Speed(km/h)', 'sum')] / grouped[('Speed(km/h)', 'count')]

    # Drop unnecessary columns
    grouped.drop(('Speed(km/h)', 'sum'), axis=1, inplace=True)
    grouped.drop(('Speed(km/h)', 'count'), axis=1, inplace=True)

    # Rename the columns
    grouped.columns = ['Vehicle ID', 'Vehicle Type', 'Average Speed']

    # Write the grouped DataFrame to a new CSV file
    grouped.to_csv('grouped_output.csv', index=False)

    # Display overspeed vehicles
    spd = pd.read_csv('grouped_output.csv')

    # Filter the DataFrame to include only rows where 'Average Speed' exceeds 50 km/h
    speed_limit_exceeded = spd[spd['Average Speed'] > 50]

    # Print the Vehicle ID and Speed for rows where the speed limit is exceeded
    

    cap.release()
    out.release()
    cv2.destroyAllWindows()


    # Restore the original stdout and stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
overspeed=0
print("Vehicle ID\tVehicle Type\tSpeed (km/h)")
for index, row in speed_limit_exceeded.iterrows():
    print(f"{int(row['Vehicle ID'])}\t\t{row['Vehicle Type']}\t\t{row['Average Speed']:.2f}")
    overspeed+=1
print("Total no of vehicles which crossed speed limit : ",overspeed)
