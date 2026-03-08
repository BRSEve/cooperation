'''Class representing packet which stores the starting position, current position,
destination node, and time steps sent alive'''
class Packet(object):
    def __init__(self, startPos, endPos, curPos, index, weight, time = 0, flag=0, times=0,
                 service_level="normal", priority=1, resource_demand=1, max_retries=10):
        self._startPos = startPos
        self._endPos = endPos
        self._curPos = curPos
        self._index = index
        self._weight = weight
        self._time = time
        self._flag = flag
        self._times = times
        self._service_level = service_level
        self._priority = priority
        self._resource_demand = resource_demand
        self._max_retries = max_retries
    def get_startPos(self):
        return self._startPos

    def get_endPos(self):
        return self._endPos

    def get_curPos(self):
        return self._curPos
        
    def get_index(self):
        return self._index

    def get_weight(self):
        return self._weight
        
    def get_time(self):
        return self._time

    def get_flag(self):
        return self._flag

    def get_service_level(self):
        return self._service_level

    def get_priority(self):
        return self._priority

    def get_resource_demand(self):
        return self._resource_demand

    def get_max_retries(self):
        return self._max_retries
        
    def set_startPos(self, startNode):
        self._startPos = startNode

    def set_endPos(self, endNode):
        self._endPos = endNode

    def set_curPos(self, curNode):
        self._curPos = curNode
    
    def set_index(self, index):
        self._index = index

    def set_weight(self, weight):
        self._weight = weight
        
    def set_time(self, time):
        self._time = time 

    def set_flag(self, flag):
        self._flag = flag

    def set_service_level(self, service_level):
        self._service_level = service_level

    def set_priority(self, priority):
        self._priority = priority

    def set_resource_demand(self, resource_demand):
        self._resource_demand = resource_demand

    def set_max_retries(self, max_retries):
        self._max_retries = max_retries

    def congestion_times(self, times):
        self._times = times

    def get_congestion_times(self):
        return self._times
'''Class which stores all the packets in the network'''
class Packets(object):
    def __init__(self, packetList):
        self.packetList = packetList
        self.num_Packets = len(packetList)
