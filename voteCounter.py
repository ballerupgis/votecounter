
def readNameAndLocations(path):
    pass

def pointWithinRectangle(point, rectangle):
    px = point[1]
    py = point[0]
    x_vals = [x for (y, x) in rectangle]
    y_vals = [y for (y, x) in rectangle]
    x_min = min(x_vals)
    x_max = max(x_vals)
    y_min = min(y_vals)
    y_max = max(y_vals)    
    if (x_min < px < x_max and y_min < py < y_max):
        return True
    else:
        return False
    
def getVote(point, data):
    for i in data:
        if pointWithinRectangle(point, i["rect"]):
            return i["name"]

if __name__ == '__main__':
    p1 = (2,2)
    p2 = (10,10)
    rectangle = [(1,1), (4,4)]
    rectangles = [
        {
            "name": "JENS OLE",
            "rect": [(1,1), (4,4)] 
        }, {
            "name": "FINN VONSYL",
            "rect": [(6,6), (12,12)] 
        }    
    ]

    print(getVote(p1, rectangles))