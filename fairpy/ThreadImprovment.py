import math
import random
from typing import List
import matplotlib.pyplot as plt
from goods_chores import Double_RoundRobin_Algorithm as round_robin_regular
from goods_chores import Generalized_Adjusted_Winner_Algorithm as Adjusted_Winner_regular
from goods_chores import  Generalized_Moving_knife_Algorithm as Moving_knife_regular
from goods_chores import  Generalized_Moving_knife_Algorithm_Recursive
from goods_chores import is_EF1 as isEF1
from fairpy.agentlist import AgentList
import time
import concurrent.futures as cfu
WORKERS = 6

def allocate_o_plus(agent_list :AgentList, o_plus:List, allocation:dict):
    while len(o_plus) != 0:
        for agent in reversed(agent_list):
            best_val = -math.inf
            allocate_chore = 0
            for chore in o_plus:
                curr_agent_val = agent.value(str(chore))
                if curr_agent_val > best_val:
                    best_val = curr_agent_val
                    allocate_chore = str(chore)

            allocation[agent.name()].append(allocate_chore)
            o_plus.remove(allocate_chore)

            if len(o_plus) == 0:
                break
    return allocation

def allocate_o_minus(agent_list :AgentList, o_minus:List, allocation:dict,k):

    # Allocate items in O- to agents in round-robin sequence

    while len(o_minus) != 0:
        for agent in agent_list:
            best_val = -math.inf
            allocate_chore = 0
            for chore in o_minus[0:len(o_minus) - k]:
                # if chore is None:
                #     allocation[agent.name()].append(None)
                #     allocate_chore = None
                #     break
                curr_agent_val = agent.value(str(chore))
                if curr_agent_val > best_val:
                    best_val = curr_agent_val
                    allocate_chore = chore
            if best_val < 0 and k > 0:
                allocate_chore = None
                k -= 1
            allocation[agent.name()].append(allocate_chore)
            o_minus.remove(allocate_chore)

            if len(o_minus) == 0:
                break

def partitionO(agent_list,O,o_plus,o_minus):
    flag = False
    for chore in O:
        for agent in agent_list:
            flag = False
            # if any agent values chore for more than 0
            if agent.value(str(chore)) > 0:
                o_plus.append(str(chore))
                flag = True
                break
        # if all agent values chore for less than or equal 0
        if flag is False:
            o_minus.append(str(chore))


def  Double_RoundRobin_Algorithm(agent_list :AgentList)->dict:
    """
    "Fair allocation of indivisible goods and chores" by  Ioannis Caragiannis ,
        Ayumi Igarashi, Toby Walsh and Haris Aziz.(2021) , link
        Algorithm 1: Finding an EF1 allocation
        Programmer: Yair Raviv , Rivka Strilitz

        >>> Double_RoundRobin_Algorithm(AgentList({"Agent1":{"1":-2,"2":1,"3":0,"4":1,"5":-1,"6":4},"Agent2":{"1":1,"2":-3,"3":-4,"4":3,"5":2,"6":-1},"Agent3":{"1":1,"2":0,"3":0,"4":6,"5":0,"6":0}}))
        {'Agent1': ['3', '6'], 'Agent2': ['5', '2'], 'Agent3': ['4', '1']}
        >>> Double_RoundRobin_Algorithm(AgentList({"Agent1":{"1":-2,"2":-2,"3":1,"4":0,"5":5,"6":3,"7":-2},"Agent2":{"1":3,"2":-1,"3":0,"4":0,"5":7,"6":2,"7":-1},"Agent3":{"1":4,"2":-3,"3":6,"4":-2,"5":4,"6":1,"7":0},"Agent4":{"1":3,"2":-4,"3":2,"4":0,"5":3,"6":-1,"7":-4}}))
        {'Agent1': ['4', '6'], 'Agent2': ['5'], 'Agent3': ['7', '3'], 'Agent4': ['2', '1']}
        >>> Double_RoundRobin_Algorithm(AgentList({"Agent1":{"1t":-2,"2d":-2,"3":1,"4":0,"5":5,"6":3,"7":-2},"Agent2":{"1t":3,"2d":-1,"3":0,"4":0,"5":7,"6":2,"7":-1},"Agent3":{"1t":4,"2d":-3,"3":6,"4":-2,"5":4,"6":1,"7":0},"Agent4":{"1t":3,"2d":-4,"3":2,"4":0,"5":3,"6":-1,"7":-4}}))
        {'Agent1': ['4', '6'], 'Agent2': ['5'], 'Agent3': ['7', '3'], 'Agent4': ['2d', '1t']}
    """

    N = agent_list.agent_names()
    O = agent_list.all_items()
    # Initialize the allocation for each agent
    allocation = {i: [] for i in N}

    # Partition the items into O+ and O-
    o_plus = []
    o_minus = []

    O1 = []
    O2 = []

    i = 0
    for item in O:
        if i < len(O)/2:
            O1.append(item)
        else:
            O2.append(item)
        i += 1


    with cfu.ThreadPoolExecutor(max_workers=WORKERS)as executor:
        executor.submit(partitionO, agent_list, O1, o_plus, o_minus)
        executor.submit(partitionO, agent_list, O2, o_plus, o_minus)


    # Add k dummy items to O- such that |O- | = an
    k = len(N) - (len(o_minus) % len(N))
    o_minus += [None] * k


    with cfu.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        # Allocate items in O- to agents in round-robin sequence
        executor.submit(allocate_o_minus,agent_list,o_minus,allocation,k)
        # Allocate items in O+ to agents in reverse round-robin sequence
        executor.submit(allocate_o_plus,agent_list,o_plus,allocation)

        # Remove dummy items from allocation
    for i in N:
        allocation[i] = [o for o in allocation[i] if o is not None]

    return allocation


#########################algo 2#######################
def devide_O_plus(winner,looser,all_items):
    o_plus = [x for x in all_items if winner.value(x) > 0 and looser.value(x) > 0]
    return o_plus
def devide_O_minus(winner,looser,all_items):
    o_minus = [x for x in all_items if winner.value(x) < 0 and looser.value(x) < 0]
    return o_minus
def devide_O_w(winner,looser,all_items):
    o_w = [x for x in all_items if winner.value(x) >= 0 and looser.value(x) <= 0]
    return o_w
def devide_O_l(winner,looser,all_items):
    o_l = [x for x in all_items if winner.value(x) <= 0 and looser.value(x) >= 0]
    return o_l


def  Generalized_Adjusted_Winner_Algorithm(agent_list :AgentList)->dict:

    if len(agent_list) != 2:
        raise "Invalid agents number"

    winner = agent_list[0]
    looser = agent_list[1]
    all_items = list(winner.all_items())




    with cfu.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        O_plus_future=executor.submit(devide_O_plus,winner,looser,all_items)
        O_minus_future = executor.submit(devide_O_minus, winner, looser, all_items)
        O_w_future = executor.submit(devide_O_w, winner, looser, all_items)
        O_l_future = executor.submit(devide_O_l, winner, looser, all_items)


    O_plus = O_plus_future.result()
    O_minus = O_minus_future.result()
    O_w = O_w_future.result()
    O_l = O_l_future.result()

    for x in O_l:
        if x in O_w:
            O_l.remove(x)

    Winner_bundle = [x for x in O_plus]
    for x in O_w:
        if x not in Winner_bundle:
            Winner_bundle.append(x)

    Looser_bundle = [x for x in O_minus]
    for x in O_l:
        if x not in Looser_bundle:
            Looser_bundle.append(x)

    O_plus_O_minus = sorted((O_plus + O_minus) , key=lambda x : (abs(looser.value(x)) / abs(winner.value(x))) , reverse=True)

    for t in O_plus_O_minus:
        if isEF1(winner , looser , Winner_bundle , Looser_bundle):
            return {"Agent1" : sorted(Winner_bundle , key=lambda x: int(x)) , "Agent2" : sorted(Looser_bundle , key=lambda x: int(x))}
        if t in O_plus:
            Winner_bundle.remove(t)
            Looser_bundle.append(t)
        else:
            Winner_bundle.append(t)
            Looser_bundle.remove(t)
    return {}
###################algo3####################

def calc_prop_value(agent_list,items,agents_num):
    result = {}
    prop_values = {}
    for agent in agent_list:
        prop_values[agent.name()] = (sum([agent.value(item) for item in items]) / agents_num)
        result[agent.name()] = []
    return prop_values,result

def Generalized_Moving_knife_Algorithm(agent_list :AgentList , items:list):
    """
        "Fair allocation of indivisible goods and chores" by  Ioannis Caragiannis ,
            Ayumi Igarashi, Toby Walsh and Haris Aziz.(2021) , link
            Algorithm 3:  Finding a Connected PROP1 Allocation
            Programmer: Yair Raviv , Rivka Strilitz
            Example 1: Non-Negative Proportional Utilities
            >>> Generalized_Moving_knife_Algorithm(AgentList({"Agent1":{"1":0,"2":-1,"3":2,"4":1},"Agent2":{"1":1,"2":3,"3":1,"4":-2},"Agent3":{"1":0,"2":2,"3":0,"4":-1}}) , ['1' , '2' , '3' , '4'])
            {'Agent1': ['3', '4'], 'Agent2': ['1'], 'Agent3': ['2']}
            >>> Generalized_Moving_knife_Algorithm(AgentList({"Agent1":{"1":0,"2":2,"3":0,"4":-4},"Agent2":{"1":1,"2":-2,"3":1,"4":-2},"Agent3":{"1":0,"2":-4,"3":1,"4":1}}),['1' , '2' , '3' , '4'])
            {'Agent1': ['1', '2', '3'], 'Agent2': [], 'Agent3': ['4']}
        """

    agents_num = len(agent_list)
    if agents_num <= 0:
        return {}

    with cfu.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        ans=executor.submit(calc_prop_value,agent_list,items,agents_num)

    prop_values =dict(ans.result()[0])
    result =dict(ans.result()[1])

    res = Generalized_Moving_knife_Algorithm_Recursive(agent_list= agent_list ,prop_values= prop_values , remain_items=items ,result= result)

    return res



if _name_ == '_main_':


    def ceate_agent_lists(all_agents_size,num_agents,num_items):
        all_agent_list =[]
        for i in range(all_agents_size):
            agent_lst = {}
            for agent in range(num_agents):
                lst = {}
                for item in range(0, num_items):
                    lst[str(item)] = random.randint(-10, 10)
                agent_lst["Agent{num}".format(num=str(agent))] = lst
            all_agent_list.append(AgentList(agent_lst))
        return all_agent_list

    def calc_time(algo,all_agent_list):
        run_time = []
        for agent_list in all_agent_list:
            start = time.perf_counter()
            algo(agent_list)
            run_time.append(time.perf_counter() - start)
        return run_time


    def calc_time_knife(algo, all_agent_list):
        run_time = []
        for agent_list in all_agent_list:
            items =list(agent_list.all_items())
            start = time.perf_counter()
            algo(agent_list,items)
            run_time.append(time.perf_counter() - start)
        return run_time


    def plot(xpoint,threadpoint,regularpoint,title):

        plt.subplots(1,1)
        plt.title(title)
        plt.plot(xpoint,threadpoint,'b-')
        plt.plot(xpoint, regularpoint, 'r-')
        plt.xlabel("size of input")
        plt.ylabel("run time")
        plt.show()


    agent_list_robin = ceate_agent_lists(50,30,50)
    thread_run_time = calc_time(Double_RoundRobin_Algorithm,agent_list_robin)
    regular_run_time = calc_time(round_robin_regular, agent_list_robin)
    print("tread run time round_robin",thread_run_time)
    print("regular run time round_robin", regular_run_time)
    plot(range(len(agent_list_robin)),thread_run_time,regular_run_time,"run time of round robin")


    agent_list_winner = ceate_agent_lists(30,2,30)
    print("agent list winner",agent_list_winner)
    thread_run_time = calc_time(Generalized_Adjusted_Winner_Algorithm,agent_list_winner)
    regular_run_time = calc_time(Adjusted_Winner_regular,agent_list_winner)
    print("tread run time Adjusted_Winner", thread_run_time)
    print("regular run time Adjusted_Winner", regular_run_time)
    plot(range(len(agent_list_winner)), thread_run_time, regular_run_time,"run time of  Adjusted_Winner")


    agent_list_knife = ceate_agent_lists(50,30,50)
    thread_run_time = calc_time_knife(Generalized_Moving_knife_Algorithm,agent_list_knife)
    regular_run_time = calc_time_knife(Moving_knife_regular, agent_list_knife)
    print("tread run time Moving_knife",thread_run_time)
    print("regular run time Moving_knife", regular_run_time)
    plot(range(len(agent_list_knife)),thread_run_time,regular_run_time,"run time of Moving_knife")


    import doctest
    (failures, tests) = doctest.testmod(report=True, optionflags=doctest.NORMALIZE_WHITESPACE + doctest.ELLIPSIS)
    print("{} failures, {} tests".format(failures, tests))
