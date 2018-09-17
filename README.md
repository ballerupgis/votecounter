# votecounter
Vote counter prototype for elections. 

## Use
In order to detect marks on ballots, regions of interest (ROI) needs to be drawn on an empty ballot. This is done
with `annotateMarkZones.py` where the ROIs are marked with left mouse button for selecting minimum two corners in a rectangle. 
Following keys can be used for in annotation: 
* "y" when ROI is finished, 
* "d" deleting currently selected corners
* "r" for deleting last ROI
* "q" for quit
* "o" for OCR scan of name

The following images shows the annotation on a ballot where the green rectangle is the one currently drawn whereas the red indicates finished rectangles.

![annotation](https://user-images.githubusercontent.com/7534153/45611300-3b0a7280-ba5f-11e8-9990-b8f7592f9df1.png)
