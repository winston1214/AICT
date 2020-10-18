# Geocoding output (Visualization Folium)

- Yellow Point 🟡 : Received Data(Facility Points)

- Red Point 🔴 : Received Data(Irregular in distance)

- Blue Point 🔵 : Received Data(divided by 10 meter)

- Green Point 🟢 : Red Points and Blue Points divided by 1 meter

- Black Point ⚫: Red Points divided by 1 meter (Over 200 meters)

**I wanted to show you the html file, but I couldn't put it in the readme because I lacked ability😥😥**

<p align="center"><img src="https://user-images.githubusercontent.com/47775179/96367042-6ef87a80-1186-11eb-9f41-57680a071636.png",height="100px",width="100px"></p>


# Convert From Absolute Points to Relative Points

- Absolute Points(Latitude,Longitude)

- Realative Points(TM planar coordinates)

**Why?** Because Latitude and Longitude are the coordinates of the earth (tripartite), we have to coordinate it in three dimensions, but we don't know the altitude.

**SO** Requires two-dimensionalization to be displayed on a plane map & Convert to TM plan coordinates to reduce distortion

**And** As shown in the picture below, the starting point was replaced with (0,0) and the points were replaced with the points beyond.

<p align='center'><img src = https://user-images.githubusercontent.com/47775179/96369164-6490ad80-1193-11eb-9ed4-90eee16c4ad8.png></p>

