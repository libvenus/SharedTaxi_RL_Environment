class Passenger():

    def __init__(self, passenger_id, location, destination):
        self.location = location
        self.destination = destination
        self.passenger_id = passenger_id

    def get_passenger_id(self):

        return self.passenger_id

    def get_location(self):

        return self.location

    def set_location(self, location):

        self.location = location

        return True

    def get_destination(self):

        return self.destination

    def set_destination(self, destination):

        self.destination = destination

        return True

