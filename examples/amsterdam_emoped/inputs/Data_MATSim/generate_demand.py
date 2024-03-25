from lxml import etree
import pandas as pd
import os

fname = 'inputs/Data_MATSim/agent_plans_with_SAV.xml' # path of file
fname_out = 'inputs/Data_MATSim/demand_all.csv'
#path = os.getcwd()
# read xml file
tree = etree.parse(fname)
root = tree.getroot()
# Initialization
demand, id_nb, O_x, O_y, D_x, D_y, dep_time, mobility_service = ([] for i in range(8))

 # Iterate through the MATSim demand file and add each leg of the plan as a agent
# Iterate through the root node to get the child node
for user in tree.xpath("/population/person"):
    ID = user.get('id')
    plan = user.xpath('plan')
    activities = plan[0].xpath("activity")   # [0] is necessary
    legs = plan[0].xpath("leg")
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    time = []
    service = []
    for act in activities: # Get locations of activities
        x1.append(act.get("x"))
        y1.append(act.get('y'))
        x2.append(act.get("x"))
        y2.append(act.get('y'))
    for l in legs:   # Iterate through leg in plan
        service.append(l.get('mode'))  # get other travel modes
        time.append(l.get('dep_time')) # append departure time

    x1.pop() # Leg O/D are locations of before and after activities
    y1.pop()
    x2.pop(0)
    y2.pop(0)
    id_ = [ID+'-'+str(i+1) for i in range(len(x1))] # unique ID per trip
    id_nb+=id_ # id
    O_x+=x1   # x-coordinate of the origin
    O_y+=y1   # x-coordinate of the destination
    D_x+=x2   # y-coordinate of the origin
    D_y+=y2   # y-coordinate of the destination
    dep_time+=time  # departure time
    mobility_service+=service  # mobility service
# Create a list of combinasion of 4 columns
demand = list(zip(id_nb,dep_time,O_x,O_y,D_x,D_y,mobility_service))
#Create dataframe from listes
demandDf = pd.DataFrame(demand, columns = ['ID', 'DEPARTURE', 'O_x', 'O_y', 'D_x', 'D_y', 'SERVICE'])
# Concatenate and rename coordinates columns
demandDf['ORIGIN'] = demandDf.apply(lambda x : '%s %s' % (x['O_x'], x['O_y']), axis=1)
demandDf['DESTINATION'] = demandDf.apply(lambda x : '%s %s' % (x['D_x'], x['D_y']), axis=1)
# Delet the four origins coordinates columns
demandDf = demandDf.drop(labels=['O_x', 'O_y', 'D_x', 'D_y'], axis=1)
#Mobility = demandDf['SERVICE'].tolist()
#indices = [i for i, x in enumerate(Mobility) if x == 'PersonalVehicle']
#demandDf = demandDf.loc[indices]
demandDf = demandDf[['ID', 'DEPARTURE', 'ORIGIN', 'DESTINATION', 'SERVICE']]
 # switch to incremental order by dep_time
demandDf = demandDf.sort_values(by = 'DEPARTURE')
demandDf = demandDf.reset_index(drop=True)

# Save file
demandDf.to_csv(fname_out, sep = ';', index = False)