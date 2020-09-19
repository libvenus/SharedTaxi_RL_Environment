class Taxi():

    def __init__(self, location, no_of_passengers, max_capacity):
        self.location = location
        self.passengers_in_taxi = []
        self.passengers_dropped = []
        self.max_capacity = max_capacity
        self.no_of_passengers = no_of_passengers

    def get_passengers_in_taxi(self):

        return self.passengers_in_taxi

    def pickup_passenger(self, passenger):

        self.passengers_in_taxi.append(passenger)
        self.no_of_passengers = len(self.passengers_in_taxi)

        return True

    def drop_passenger(self, passenger):

        self._drop_passenger(passenger)
        self.passengers_dropped.append(passenger)
        self.no_of_passengers = len(self.passengers_in_taxi)

        return True

    def _drop_passenger(self, passenger):
        passenger_index = 0

        passenger_to_drop = passenger.get_passenger_id()

        for passenger in self.passengers_in_taxi:
            if passenger.get_passenger_id() != passenger_to_drop:
                passenger_index = passenger_index + 1
            else:
                break

        del self.passengers_in_taxi[passenger_index]

        return True

    def get_no_of_passengers_dropped(self):

        return len(self.passengers_dropped)

    def get_location(self):

        return self.location

    def set_location(self, location):

        self.location = location

        return True

    def get_max_capacity(self):

        return self.max_capacity

    def set_max_capacity(self, max_capacity):

        self.max_capacity = max_capacity

        return True

    def get_no_of_passengers(self):

        return self.no_of_passengers

    def set_no_of_passengers(self, no_of_passengers):

        self.no_of_passengers = no_of_passengers

        return True

    def get_passengers_dropped(self):

        return self.passengers_dropped

