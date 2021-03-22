


def isDrawer(st):
    head = st[:2]
    tail = st[2:]
    if head == 'KS' and tail.isdecimal():
        return True
    return False


def findDrawer(data):
    temp = []
    left = []
    right = []
    drawer_index = None
    drawer_name = ''

    # check index start drawer and end drawer
    for i in range(len(data)):
        tmp = data[i]

        if isDrawer(tmp):
            drawer_index = i
            drawer_name = tmp
    
    if drawer_index != None:
        left = data[:drawer_index]
        right = data[drawer_index+1:]
    else:
        temp = data
        
    
    return drawer_name, left, right, temp

    
def formatResult(datas):
    result = []
    for key, value in datas.items():
        result.append({'drawer':key, 'books':value})
    return result


def filterDrawer(dr1, dr2):
    for book in dr1:
        if book in dr2:
            dr2.pop(dr2.index(book))
    return dr2    


def runFilterDrawer(datas):
    datas.reverse()
    for i in range(len(datas)):
        if i < len(datas) - 1:
            dr2 = filterDrawer(datas[i]['books'], datas[i+1]['books'])
            datas[i+1]['books'] = dr2
    datas.reverse()
    return datas


def putBookToDrawer(datas):
    drawers = {}
    cur_drawer = []
    pre_drawer = ''
    temp = []
    for data in datas:
        if not data:
            continue
        drawer_name, left, right, tmp = findDrawer(data)

        
        if  drawer_name != '' and not (drawer_name in cur_drawer):
            cur_drawer.append(drawer_name)
        
        if len(cur_drawer) == 2:
            pre_drawer = cur_drawer.pop(0)

        # temp
        if tmp:
            temp.extend(tmp)
            if pre_drawer != '':
                for i in tmp:
                    if i in drawers[pre_drawer]:
                        tmp.pop(tmp.index(i))

        # right
        if cur_drawer[0] in drawers:
            drawers[cur_drawer[0]].extend(right)
            drawers[cur_drawer[0]].extend(temp)
            temp = []
        else:
            drawers[cur_drawer[0]] = right

        # left
        if left:
            drawers[pre_drawer].extend(left)

    for key in drawers.keys():
        drawers[key] = list(set(drawers[key]))     
    
    return drawers


def returnResult(datas):
    drawers = putBookToDrawer(datas)
    drawers = formatResult(drawers)
    drawers = runFilterDrawer(drawers)
    return drawers




