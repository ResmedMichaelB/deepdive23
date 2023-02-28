def position_map(x):
    if x >= 30 and x<150:
        return 1 # Left
    elif x >= 150 and x<210:
        return 0 #Supine
    elif x >= 210 and x<330:
        return 2 # Right 
    else:
        return 3 #Prone


def fillius_map(x):
    # position_map={0:'left',1:'supine',2:'prone',3:'right'}
    if x==0:
        return 1
    elif x==1:
        return 0
    elif x==2:
        return 3
    elif x==3:
        return 2
    
    return 4