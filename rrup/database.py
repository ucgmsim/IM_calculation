from abc import abstractmethod, ABCMeta


class AbstractDatabase:
    """
    Interface for a Database object, this list of methods is extensive
    """
    __metaclass__ = ABCMeta

    """Returns void
        Saves the current state of the database to disk    
    """
    @abstractmethod
    def save(self):
        pass

    """Returns void
        Makes sure the data structures required to store the data exist
    """
    @abstractmethod
    def initialize(self):
        pass

    """Returns list
        SQLite returns a list of tuples. This method extracts the first value of the tuples 
        into a list form.
    """
    def convert_tuple_to_list(self, tuple_list):
        return [station[0] for station in tuple_list]

    """Returns bool
        If the event exists in the storage system returns true otherwise returns false
        If the event exists, it then updates the sim/obs event id in the database class
        so it can access the relevant data for that event (using the set_event_id method)       
        This method always returns false for 1 event storage systems
    """
    @abstractmethod
    def event_exists(self, run_name, is_simulation):
        pass

    """Returns void
        Inserts an event into the database with the specified parameters this function requires
    """
    @abstractmethod
    def insert_event(self, run_name, is_simulation, magnitude, rake, dip, Ztor, period):
        pass

    """Returns a list of tuples
        1: run_name, 2: event_id, 3: sim flag (bool if sim)
        Returns all the events present in the database
    """
    @abstractmethod
    def get_event_names(self):
        pass

    """Returns a list of measure names
        The measures that were calculated for a given event (obs or sim specified by flag)
    """
    @abstractmethod
    def get_event_ims(self, is_simulation):
        pass

    """returns list of floats
        for a sim/obs event what additional periods are loaded in the database.
        aka - the periods that are additionally specified in the config rather than the 
        autogenerated ones 
    """
    @abstractmethod
    def get_additional_periods(self, is_simulation):
        pass

    """Returns a list of floats
        the pSA values for a given station and component
        (the order of pSA values is given by 'get_allowable_periods')
    """
    @abstractmethod
    def get_pSA(self, is_simulation, station, component):
        pass

    """Returns a list of tuples.
        (1: lat, 2: lon, 3: group_concat(titles), 4: group_concat(values), 5: count(values))
        For every station location  list the requested measures (and pSA's for the given periods)
        
        Works for both intensity measures (obs/sim) and ratios (obs/sim)         
    """
    @abstractmethod
    def get_measures_for_spatial_plot(self, measures, periods, ratio, is_simulation):
        pass

    """Returns a list of tuples
        (1: Period, 2: group_concat(value), 3: group_concat(station_names))
        For each pSA period return the ratios for the stations and periods requested. 
        If there are no stations or periods set then it will return for all stations / periods
        There is a flag to switch to retrieving empirical ratios rather than obs/sim ratios    
    """
    @abstractmethod
    def get_pSA_ratio(self, stations, periods, component='geom', empirical=False):
        pass

    """Returns a list of tuples
        (1: values; 2 (optional): station_name)
        Fetches intensity measure values for a given component, measure (with period)
            This can fetch from sim, obs, obs/sim ratio, empirical, obs/emp ratio
        The include name flag adds the station_name as the second arguement in the tuple
    """
    @abstractmethod
    def get_IM(self, is_simulation, stations, component, measure, period, ratio, empirical, include_name):
        pass

    """Returns a list of tuples
        Just the rrups for the given stations (or all if no stations provided) for either simulation stations
        or observed stations
    """
    @abstractmethod
    def get_rrups(self, is_simulation, stations):
        pass

    """Returns a list of tuples
        (1 (optional): rrups, 2: intensity measure, (optional 3: NT, 4: DT, 5: time-offset)) 
        For the velocity seismogram retrieving the above data for the given
        station list, component and obs/sim
    """
    @abstractmethod
    def get_stations_with_IM(self, measure, component, is_simulation, stations, time_offsets, include_rrups):
        pass

    """Returns a list of tuples
        (1: 000 measure, 2: 090 measure, 3: ver measure, 4: geom measure, (optional 5: NT, 6: DT, 7: timeoffset), 8: station_name 
        Only used for velocity time series plots
    """
    @abstractmethod
    def get_stations_with_IM_for_all_components(self, measure, is_simulation, stations, time_offsets):
        pass

    """returns a float
       returns the maximum PGV over an entire event (either obs or sim)
       Used for scaling the velocity seismogram 
    """
    @abstractmethod
    def get_max_PGV(self, is_simulation):
        pass

    """Returns a list
        The list of periods that have been calculated for in a event (existing in the loaded obs/sim)    
    """
    @abstractmethod
    def get_allowable_periods(self):
        pass

    """Returns a float
        the highest pSA value for that event (obs or sim).
        Used to scale the pSA plots
    """
    @abstractmethod
    def get_max_pSA(self):
        pass

    """Returns void
        Adds a station to the database. Needs name and whether it belongs to a simulation or observation
    """
    @abstractmethod
    def insert_station(self, is_simulation, name, NT, DT, time_offset):
        pass

    """Returns a list of tuples
        1: station id, rrup, rjbs, vs30
        For the given stations return the values that are required to calculate the GMPE IMs for that location
        The station id is the key to recognise the station when inserting the GMPE values for it.
    """
    @abstractmethod
    def get_GMPE_stations(self, stations):
        pass

    """Returns Void
        Stores a GMPE value in the db. These are stored for a vs30 value over a range of rrups.
        Each value is inserted independently
    """
    @abstractmethod
    def insert_GMPE(self, im, period, vs30, rrup, im_rscaling, sigma_im_rscaling):
        pass

    """Returns void
        Based on the id given by get_GMPE_stations stores the GMPE value for each measure and period
        insert the empirical values for each measure, period
        This is called for each value inserted
    """
    @abstractmethod
    def add_empirical_to_station(self, s_id, measure, period, im_rscaling, sigma_im_rscaling):
        pass

    """Returns list of floats
        (it has a limit of returning two floats as the overlying code can only support reading up to 2 values)
        Called when plotting the GMPEs on the IM plots
    """
    @abstractmethod
    def get_GMPE_vs30s(self):
        pass

    """Returns a list of tuples
        1: rrup, 2: im_value, 3: sigma(total std)
        For the given im, vs30 and specified period (as necessary) return the values of the GMPE
    """
    @abstractmethod
    def get_GMPE(self, im, vs30, period):
        pass

    """Returns void
        adds a single rrup value to a station which has a given name for both obs / sim
    """
    @abstractmethod
    def add_rrups_to_station(self, name, lat, lon, r_rups, r_jbs):
        pass

    """Returns void
        For a station, adds the NT, DT and time_offset from the velocity file
    """
    @abstractmethod
    def add_times_to_station(self, name, is_simulation, NT, DT, time_offset):
        pass

    """Returns void
        For a station, adds the vs30 value to the station
    """
    @abstractmethod
    def add_vs30_to_station(self, station_name, value):
        pass

    """Returns void
        Stores an intensity measure, unique per measure and component
    """
    @abstractmethod
    def add_im_to_station(self, name, is_simulation, measure, component, value):
        pass

    """Returns void 
        Adds a list of psa (value, period pairs). unique per measure and component
    """
    @abstractmethod
    def add_pSA_to_station(self, name, is_simulation, measure, component, values, period):
        pass

    """Returns a list of tuples
        1:station_id, 2:measure_name, 3: component (always 'geom'), 4: ['000','090' im values], 5:period
        For stations that have both components, without a geometric calculated.
        This list is the measures that need to have the geometric calculated
    """
    @abstractmethod
    def get_geom(self):
        pass

    """Returns void
        For data already sorted in a list of tuples inserts them into the intensity measure storage
        Primarily used to insert geometric data
        1:station_id, 2:measure_name, 3: component 4: im_value, 5:period
    """
    @abstractmethod
    def insert_multiple_im(self, data):
        pass

    """Returns void
        For each intensity measure with a corresponding obs/sim pair. calculate ln(obs_im/sim_im)
        Ratios are only calculted on geometric values
        This function also stores these ratios. 
    """
    @abstractmethod
    def create_ratios(self, stations):
        pass

    """Returns void
        For each intensity measure with a corresponding obs/gmm pair. calculate ln(obs_im/gmm_im)
        Ratios are only calculted on geometric values
        This function also stores these ratios. 
    """
    @abstractmethod
    def create_obs_emp_ratios(self, stations):
        pass

    """Returns a list of station_names
        Returns the stations that exists in both the obs and sim sets
        Has an optional flag for rrups (true/false) to only returns stations that have an rrup value
        The argument stations is a list of stations to be checked, if the list is empty or None then it returns all stations
    """
    @abstractmethod
    def get_matched_station_names(self, stations, rrups):
        pass

    """Returns a list of station_names
        Returns the stations that exist in the selected obs/sim set
        Has an optional flag for rrups (true/false) to only returns stations that have an rrup value
    """
    @abstractmethod
    def get_station_names(self, is_simulation, rrups):
        pass

    """Returns void
        Sets the flag that the database contains RRUP values
    """
    @abstractmethod
    def set_rrups_calculated(self, value):
        pass

    """Returns void
        Sets the flag that the database contains vs30 values
    """
    @abstractmethod
    def set_vs30_calculated(self):
        pass

    """Returns void
        Sets the flag that the database contains GMPE values
    """
    @abstractmethod
    def set_GMPE_calculated(self, value):
        pass

    """Returns void
        Sets the flag that the database contains GMPE values for stations
    """
    @abstractmethod
    def set_stations_GMPE_calculated(self, value):
        pass

    """Returns void
        Sets the flag that the database contains GMPE obs/emp ratios 
    """
    @abstractmethod
    def set_GMPE_station_ratio_calculated(self, value):
        pass

    """Returns void
        Sets the flag that the database contains ratios values
    """
    @abstractmethod
    def set_ratios_calculated(self, value):
        pass

    """Returns void
        Sets the flag that the database contains intensity measures
    """
    @abstractmethod
    def set_IMs_calculated(self, value):
        pass

    """Returns a tuple
        1: bool if ratios calculated 2: bool if emp_ratios calculated
    """
    @abstractmethod
    def get_ratios_calculated(self):
        pass

    """Returns a bool
        if GMPE values are stored
    """
    @abstractmethod
    def is_GMPE_calculated(self):
        pass

    """Returns a tuple of bools
        if the following values are calculated it will be true
        1: IMs 2: rrups 3: stations GMM values 4: IM ratios
    """
    @abstractmethod
    def get_calculated(self):
        pass

    """Returns void
        Deletes all of the stored, IMs, ratios and empirical ratios
    """
    @abstractmethod
    def clear_IM(self):
        pass

    """Returns void
        Deletes all of the stored empirical values, station based empirical values and empirical ratios
    """
    @abstractmethod
    def clear_GMPE(self):
        pass

    """Returns void
        Closes the database. There should be no more calls to the db afterwards.
    """
    @abstractmethod
    def close(self):
        pass


