import os
import pandas as pd

from sqlalchemy import create_engine 

# Create the connection 
class PostgresClient:
    """ CLient for accessing data from Postgres database. """
    
    def __init__(self, address = None, port = None, dbname = None, username = None, password = None):       
        
        postgres_str = ('postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'.format(
            username=username or os.environ['POSTGRES_USERNAME'], 
            password=password or os.environ['POSTGRES_PASSWORD'], 
            ipaddress=address or os.environ['POSTGRES_ADDRESS'], 
            port=port or os.environ['POSTGRES_PORT'], 
            dbname=dbname or os.environ['POSTGRES_DBNAME'])) 
        
        self.conn = create_engine(postgres_str, connect_args={'sslmode':'require'})
        
    def connection(self):
        return self.conn

    def calendar(self, from_date, to_date):
        sql = "select date, weekday, day_type, statutory_holiday from cal_calendar where %(from)s <= date and date < %(to)s"
        df = pd.read_sql(sql, self.conn, params={'from': from_date, 'to': to_date}).set_index('date')
        return df
    
    def link_travel_time(self, link_ref, from_time, to_time):
        sql = "select time, link_travel_time_norm from link_travel_time where link_ref = %(link)s and %(from)s <= time and time < %(to)s"
        df = pd.read_sql(sql, self.conn, params={'link': link_ref, 'from': from_time, 'to': to_time}).set_index('time')
        return df
    
    def link_travel_time_n_preceding_normal_days(self, link_ref, time, n):
        sql = """
            with normal_days as
            (
                select
                    cal.date
                from
                    cal_calendar cal
                where
                    cal.date < %(time)s
                    and day_type = weekday
                order by cal.date desc
                limit %(n)s
            )
            select 
                dat.time, dat.link_travel_time_norm
            from
                link_travel_time dat
                join normal_days cal on cal.date = dat.time::date
            where
                dat.link_ref = %(link_ref)s
        """
        df = pd.read_sql(sql, self.conn, params={'link_ref': link_ref, 'time': time, 'n': n}).set_index('time')
        return df
    
    def link_travel_time_special_days(self, link_ref, from_time, to_time):
        sql = """
            select
                time, day_type, link_travel_time_norm
            from
                link_travel_time dat
                join cal_calendar cal on cal.date = dat.time::date
            where
                link_ref = %(link_ref)s
                and %(from_time)s <= time and time < %(to_time)s
                and day_type <> weekday
        """
        df = pd.read_sql(sql, self.conn, params={'link_ref': link_ref, 'from_time': from_time, 'to_time': to_time}).set_index('time')
        return df
