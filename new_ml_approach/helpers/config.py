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
        return 1 # 'left'
    elif x==1:
        return 0 # supine
    elif x==2:
        return 3 # prone
    elif x==3:
        return 2 # right
    
    return 4


def sleep_stage_str_map(x):

    if x==0:
        return 'REM'
    elif x==1:
        return 'Wake'
    elif x==-1:
        return 'Light'
    elif x==-2:
        return 'Light'
    elif x==-3:
        return 'Deep'
    elif x==-4:
        return 'Deep'

def sleep_stage_map(x):

    if x==0:
        return 0 # REM
    elif x==1:
        return 1 # Wake
    elif x==-1:
        return -1 # Light
    elif x==-2:
        return -1 # Light
    elif x==-3:
        return -3 # Deep
    elif x==-4:
        return -3 # Deep