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
