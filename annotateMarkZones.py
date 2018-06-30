import cv2
import numpy as np
import pytesseract

vertices = []

def getCoordinates(event, x, y, flags, param):
    global vertices
    
    if event == cv2.EVENT_LBUTTONDOWN:
        vertices.append((x,y))

def ocr(image, scale=5):
    """
    OCR scanning image and returning string. Since OCR works best on DPI above
    Tesseract works best on images that have DPI of 300 dpi, or more. If you’re 
    working with ROI that have a DPI of less than 300 dpi, you might consider scaling.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.resize(gray, (int(w*scale), int(h*scale)))
    text = pytesseract.image_to_string(gray, lang='dan')
    return text

def selectROIs(image, scale=1):
    """
    Draw ROIs on ballot uning left mouse button for selecting minimum two corners in a rectangle. 
    Returns list of dictionaries containing name of politician and rectangle points for ROI.
    Press: 
    "y" when ROI is finished, 
    "d" deleting currently selected corners
    "r" for deleting last ROI
    "q" for quit
    "o" for OCR scan of name
    """
    global vertices
    points = []
    rois = []
    names = []
    data = []

    h, w = image.shape[:2]
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    copy = resized.copy()

    # Create window and prepare mouse click event
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", getCoordinates)

    # Draw and keep looping until the 'q' key is pressed
    while True:

        # Draw selected points and bounding rectangle
        if len(vertices) != 0:
            for vertice in vertices:
                cv2.circle(copy, vertice, 4, (0,255,0), 1)

            x,y,w,h = cv2.boundingRect(np.array([vertices], dtype=np.int32))
            cv2.rectangle(copy,(x,y),(x+w,y+h),(0,255,0),2)
            points = [(x, y+h),(x+w, y)]

        # Draw previous ROIs
        if len(rois) != 0:
            for p1, p2 in rois:
                cv2.rectangle(copy,p1,p2,(0,0,255),2)

        # display the image and wait for a keypress
        cv2.imshow("Select ROI", copy)      
        
        # Reset image to draw only current features
        copy = resized.copy()

        # Handle different keypress for editing
        key = cv2.waitKey(1) & 0xFF
        if key & 0xFF == ord("y"):
            rois.append(points)
            vertices = []
        elif key & 0xFF == ord("d"):
            vertices = []        
        elif key & 0xFF == ord("r"):
            rois = rois[:len(rois)-1]
        elif key & 0xFF == ord("o"):
            text = resized[y:y+h, x:x+w]
            name = ocr(text)
            print(name)
            names.append(name)
            vertices = []
        elif key == ord("q"):
            cv2.destroyAllWindows()
            break

    # Rescaling back to original coordinates
    if scale != 1:
        rescale = scale**-1
        rescaled_rois = [[tuple([int(x*rescale) for x in y]) for y in z] for z in rois]
    
    # Merging name and mark region    
    for name, rect in zip(names, rescaled_rois):
        d = {
            "name": name,
            "rect": rect 
        }
        
        data.append(d)

    return data

if __name__ == '__main__':
    backgr_path = "images/stemboks/stem_back.jpg"

    background = cv2.imread(backgr_path)
    # text = background[182:196, 6:101]
    # test = ocr(text)
    # print(test)

    data = selectROIs(background, scale=2)
    #np.save('data/stemboks.npy',data)
    print(data)

    # gray = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)
    # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # gray = cv2.resize(gray, (300, 50))
    # kernel = np.ones((3,3),np.uint8)
    # gray = cv2.dilate(gray,kernel,iterations = 1)

    
    # cv2.imshow("Background", gray)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
