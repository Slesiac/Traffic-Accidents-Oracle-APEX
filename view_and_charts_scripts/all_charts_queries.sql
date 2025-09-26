-- DASHBOARD 1: Time Analysis _______________________________________________

-- KPIs
with total_crashes as (
     select count(*) as crash_count
       from a01_traffic_accidents_vw
),
avg_crashes_per_day as (
     select round(count(*) / count(distinct crash_date), 2) as avg_crashes
       from a01_traffic_accidents_vw
),
peak_hour as (
     select crash_hour,
            count(*) as cnt
       from a01_traffic_accidents_vw
   group by crash_hour
),
year_most_crashes as (
     select crash_year,
            count(*) as cnt
       from a01_traffic_accidents_vw
   group by crash_year
)
select 'fa fa-car' ICON_CLASS,
       'u-color-11' ICON_COLOR_CLASS,
       'Total Crashes' list_text,
       to_char(crash_count, 'FM999G999G999') list_title,
       1 sort_value
  from total_crashes
union all
select 'fa fa-calendar' ICON_CLASS,
       'u-color-1' ICON_COLOR_CLASS,
       'Avg Crashes per Day' list_text,
       to_char(avg_crashes, 'FM999G990D00') list_title,
       2 sort_value
  from avg_crashes_per_day
union all
select 'fa fa-clock-o' ICON_CLASS,
       'u-color-32' ICON_COLOR_CLASS,
       'Peak Crash Hour' list_text,
       max(crash_hour) keep (dense_rank last order by cnt) list_title,
       3 sort_value
  from peak_hour
union all
select 'fa fa-trophy' ICON_CLASS,
       'u-color-2' ICON_COLOR_CLASS,
       'Year with most Crashes' list_text,
       max(crash_year) keep (dense_rank last order by cnt) list_title,
       4 sort_value
  from year_most_crashes
order by sort_value;


-- Cumulative Crashes by Year
  select crash_year,
         count(*) as crash_count,
         sum(count(*)) over (order by crash_year) as crash_cumulative
    from a01_traffic_accidents_vw
   where crash_year between 2016 and 2023
group by crash_year
order by crash_year;


-- Crashes by Month and Year
  select crash_month_num,
         crash_year,
         count(*) as crash_count
    from a01_traffic_accidents_vw
   where crash_year between 2016 and 2023
group by crash_month_num,
         crash_year
order by crash_month_num,
         crash_year;


-- Crashes by Day of Week
select crash_day_num, -- 5
       crash_day_short, -- FRI
       count(*) as crash_count
  from a01_traffic_accidents_vw
group by crash_day_num,
         crash_day_short
order by crash_day_num;


-- Hourly Crash Distribution
  select crash_hour,
         crash_day_short_label,
         count(*) as crash_count
    from a01_traffic_accidents_vw
group by crash_hour, crash_day_short_label
order by crash_hour, crash_day_short_label


-- Year Over Year Crash Count Change
with crash_counts as (  
  select crash_year,
         count(*) as crash_count,
         coalesce(lag(count(*)) over ( order by crash_year ), 0) as prev_crash_count
    from a01_traffic_accidents_vw
   where crash_year between 2016 and 2023
group by crash_year
),
crash_deltas as (
  select crash_year,
         crash_count,
         prev_crash_count,
         crash_count - prev_crash_count as delta_crash_count,
         case
             when prev_crash_count = 0
             then 0
             else round(100 * (crash_count - prev_crash_count) / prev_crash_count, 2)
         end as delta_perc_crash_count
    from crash_counts
)
select crash_year,
       crash_count,
       prev_crash_count,
       delta_crash_count,
       delta_perc_crash_count,
       case 
           when delta_crash_count > 0 then '#bd4332' --rosso
           else '#4190AC' -- azzurro
       end as delta_series_color,
       case 
           when delta_perc_crash_count > 0 then '#CA4D3C' --rosso
           else '#577346' --verde
       end as delta_perc_series_color
  from crash_deltas
order by crash_year



-- DASHBOARD 2: Road & Environmental Conditions ________________________________

-- KPIs
with rain_crashes as (
     select count(*) as crash_count
       from a01_traffic_accidents_vw
      where weather_condition = 'RAIN'
),
night_crashes as (
     select count(*) as crash_count
       from a01_traffic_accidents_vw
      where lighting_condition in ('DARKNESS', 'DARKNESS, LIGHTED ROAD')
),
defective_road_crashes as (
     select count(*) as crash_count
       from a01_traffic_accidents_vw
      where road_defect not in ('NO DEFECTS','UNKNOWN')
),
malfunctioning_device_crashes as (
     select device_condition_group,
            count(*) as crash_count
       from a01_traffic_accidents_vw
      where device_condition_group = 'NOT FUNCTIONING'
   group by device_condition_group
)
select 'fa fa-cloud' ICON_CLASS,
       'u-color-1' ICON_COLOR_CLASS,
       'Crashes in Rain' list_text,
       to_char(crash_count, 'FM999G999G999') list_title,
       1 sort_value
  from rain_crashes
union all
select 'fa fa-moon-o' ICON_CLASS,
       'u-color-32' ICON_COLOR_CLASS,
       'Night Crashes' list_text,
       to_char(crash_count, 'FM999G999G999') list_title,
       2 sort_value
  from night_crashes
union all
select 'fa fa-road' ICON_CLASS,
       'u-color-5' ICON_COLOR_CLASS,
       'Crashes due to Defective Road Surface' list_text,
       to_char(crash_count, 'FM999G999G999') list_title,
       3 sort_value
  from defective_road_crashes
union all
  select 'fa fa-bolt' ICON_CLASS,
         'u-color-7' ICON_COLOR_CLASS,
         'Crashes with Malfunctioning Devices' list_text,
         to_char(crash_count, 'FM999G999G999') list_title,
         4 sort_value
    from malfunctioning_device_crashes
order by sort_value;


-- Crashes by Lighting Condition
    select lighting_group,
           count(*) as crash_count,
           case 
             when lighting_group in ('DARKNESS') then '#2c4752'
             when lighting_group in ('DAYLIGHT') then '#5f7d4f'
             when lighting_group in ('DAWN/DUSK') then '#437c94'           
             else '#6f757e'
           end as series_color
      from a01_traffic_accidents_vw
     where lighting_group != 'UNKNOWN'
  group by lighting_group


-- Crash Type by Critical Weather Conditions
with w as (
  select weather_group,
         case
             when nvl(injuries_fatal,0) > 0 or nvl(injuries_incapacitating,0) > 0 then 1
             else 0
         end as is_severe
    from a01_traffic_accidents_vw
   where weather_group not in ('UNKNOWN/OTHER', 'CLEAR')
)
  select weather_group,
         count(*) as total_crashes,
         round(100 * sum(is_severe) / count(*), 2) as severe_pct
    from w
group by weather_group
order by total_crashes desc;


-- Crash Severity by Road Surface Condition
  select roadway_surface_group,
         case
             when crash_type in ('INJURY AND / OR TOW DUE TO CRASH') then 'CRASH WITH INJURIES/TOW'
             else 'CRASH WITHOUT INJURIES'
         end as crash_severity,
         case 
             when crash_type in ('INJURY AND / OR TOW DUE TO CRASH') then '#2c4752'
             else '#437c94'
         end as series_color,
         count(*) as crash_count
    from a01_traffic_accidents_vw
   where roadway_surface_group not in ('UNKNOWN')
group by crash_type, roadway_surface_group
order by crash_type, roadway_surface_group


-- Crashes by Traffic Control Devices and their Conditions
  select traffic_control_device_group,
         device_condition_group,
         count(*) as crash_count
    from a01_traffic_accidents_vw
   where device_condition_group not in ('UNKNOWN', 'OTHER')
     and traffic_control_device_group not in ('NO CONTROL', 'UNKNOWN')
group by device_condition_group, traffic_control_device_group
order by device_condition_group, traffic_control_device_group


-- Crashes by Lane Count and Road Defect
with lanes as (
  select lane_bucket,
         case
             when nvl(injuries_fatal, 0) > 0 or nvl(injuries_incapacitating, 0) > 0 then 1 
             else 0 
         end as is_severe,
         road_defect_group
    from a01_traffic_accidents_vw
   where lane_bucket not in ('UNKNOWN/INVALID')
     and road_defect_group not in ('UNKNOWN/OTHER', 'NO DEFECTS')
)
  select lane_bucket,
         road_defect_group,
         count(*) as total_crashes,
         round(100 * sum(is_severe)/count(*), 2) as severe_pct
    from lanes
group by lane_bucket, road_defect_group
order by lane_bucket



-- DASHBOARD 3: Causes & Dynamics ____________________________________________

-- KPIs
with top_primary_cause as (
     select primary_cause_group,
            count(*) as crash_count
       from a01_traffic_accidents_vw
      where primary_cause_group not in ('UNKNOWN/NA')
   group by primary_cause_group
   order by crash_count desc
   fetch first 1 row only
),
top_first_crash_type as (
     select first_crash_type_group,
            count(*) as crash_count
       from a01_traffic_accidents_vw
   group by first_crash_type_group
   order by crash_count desc
   fetch first 1 row only
),
avg_num_units as (
     select round(avg(num_units), 0) as avg_num_units
       from a01_traffic_accidents_vw
),
avg_speed as (
     select round(avg(posted_speed_limit), 2) as avg_speed
       from a01_traffic_accidents_vw
)
   select 'fa fa-undo-arrow' ICON_CLASS,
          'u-color-5' ICON_COLOR_CLASS,
          'Crashes by Improper Maneuvers' list_text,
          to_char(crash_count, 'FM999G999G999') list_title
     from top_primary_cause
union all
   select 'fa fa-car' ICON_CLASS,
          'u-color-6' ICON_COLOR_CLASS,
          'Rear-end Crashes' list_text,
          to_char(crash_count, 'FM999G999G999') list_title
     from top_first_crash_type
union all
   select 'fa fa-tachometer' ICON_CLASS,
          'u-color-7' ICON_COLOR_CLASS,
          'Average Crash Speed' list_text,
          to_char(avg_speed, 'FM999G990D00') || ' km/h' list_title
     from avg_speed
union all
   select 'fa fa-users' ICON_CLASS,
          'u-color-8' ICON_COLOR_CLASS,
          'Average number of units involved' list_text,
          to_char(avg_num_units, 'FM999G999G999') list_title
     from avg_num_units


-- Improper Maneuvers
  select primary_cause_group,
          case
              when primary_cause in ('IMPROPER LANE USAGE') then 'LANE MISUSE'
              when primary_cause in ('IMPROPER BACKING') then 'IMPROPER BACKING'
              when primary_cause in ('IMPROPER TURNING/NO SIGNAL') then 'IMPROPER TURNING'
              else 'IMPROPER PASSING'
          end as prim_cause,
          count(*) as crash_count
     from a01_traffic_accidents_vw
   where primary_cause_group = 'IMPROPER MANEUVERS'
 group by primary_cause_group, primary_cause


-- Driver's Impairment Type
   select primary_cause_group,
          case
              when primary_cause in ('UNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED)') then 'ALCOHOL/DRUGS (DURING ARREST)'
              when primary_cause in ('PHYSICAL CONDITION OF DRIVER') then 'PHYSICAL CONDITION'
              else 'ALCOHOL (BEFORE ARREST)'
          end as prim_cause,
          count(*) as crash_count
     from a01_traffic_accidents_vw
    where primary_cause_group = 'IMPAIRMENT'
 group by primary_cause_group, primary_cause


-- Crash Contributory Cause by Years
  select crash_year,
         primary_cause_group,
         count(*) as crash_count
    from a01_traffic_accidents_vw
   where crash_year between 2016 and 2023
     and primary_cause_group in ('IMPROPER MANEUVERS', 'SPEED', 'SIGNALS/SIGNS', 'ENVIRONMENT/INFRASTRUCTURE', 'IMPAIRMENT', 'CELL PHONE USE')
group by primary_cause_group, crash_year


-- Crash Type
    select crash_type,
           first_crash_type_group,
           count(*) as crash_count,
           case 
             when crash_type in ('NO INJURY / DRIVE AWAY') then '#4C825C'          
             else '#B67745'
           end as series_color
      from a01_traffic_accidents_vw
  group by crash_type, first_crash_type_group
  order by crash_count desc


-- Crashes and Avg Injuries per Crash by Speed Bucket
with bucket as (
    select speed_bucket,
           count(*) as crash_count,
           round(avg(nvl(injuries_total,0)), 2) as avg_injuries
      from a01_traffic_accidents_vw
     where speed_bucket is not null
  group by speed_bucket
)
  select speed_bucket,
         crash_count,
         avg_injuries
    from bucket
order by speed_bucket;



-- DASHBOARD 4: Geographical Analysis ________________________________________

-- KPIs
with top_street_name as (
     select street_name,
            count(*) as crash_count
       from a01_traffic_accidents_vw
   group by street_name
   order by crash_count desc
   fetch first 1 row only
),
top_beat_of_occurence as (
     select beat_of_occurrence,
            count(*) as crash_count
       from a01_traffic_accidents_vw
   group by beat_of_occurrence
   order by crash_count desc
   fetch first 1 row only
),
intersection_crashes as (
     select trafficway_type_group,
            count(*) as crash_count
       from a01_traffic_accidents_vw
      where trafficway_type_group = 'INTERSECTION'
   group by trafficway_type_group
),
workzone_crashes as (
     select work_zone_type,
            count(*) as crash_count
       from a01_traffic_accidents_vw
      where work_zone_type is not null
   group by work_zone_type
)
   select 'fa fa-road' ICON_CLASS,
          'u-color-4' ICON_COLOR_CLASS,
          'Street with Most Crashes: Western Ave' list_text,
          to_char(crash_count, 'FM999G999G999') list_title
     from top_street_name

union all
   select 'fa fa-shield' ICON_CLASS,
          'u-color-15' ICON_COLOR_CLASS,
          'Top Beat of Occurrence' list_text,
          to_char(crash_count, 'FM999G999G999') list_title
     from top_beat_of_occurence

union all
select 'fa fa-wrench' ICON_CLASS,
       'u-color-14' ICON_COLOR_CLASS,
       'Crashes in Work Zones' list_text,
       to_char(sum(crash_count), 'FM999G999G999') list_title
  from workzone_crashes

union all
 select 'fa fa-random' ICON_CLASS,
        'u-color-13' ICON_COLOR_CLASS,
        'Crashes in Intersections' list_text,
        to_char(crash_count, 'FM999G999G999') list_title
   from intersection_crashes


-- Crashes on Work Zones
select work_zone_type,
       workers_present_group,
       count(*) as crash_count,
       case 
           when workers_present_group in ('WORKERS PRESENT') then '#4C825C'          
           when workers_present_group in ('NO WORKERS') then '#7A736E'          
           else '#7C85A4'
       end as series_color
  from a01_traffic_accidents_vw
 where work_zone_type is not null
   and work_zone_type not in ('UNKNOWN')
group by work_zone_type,
         workers_present_group
order by crash_count desc;


-- Trafficway Types
  select trafficway_type_group,
         count(*) as crash_count,
         case 
           when trafficway_type_group in ('NOT DIVIDED') then '#4C825C'
           when trafficway_type_group in ('DIVIDED') then '#7C85A4'
           when trafficway_type_group in ('ONE-WAY') then '#8B8580'
           when trafficway_type_group in ('INTERSECTION') then '#79B1C6'
           when trafficway_type_group in ('NON-ROADWAY') then '#B47282'           
           when trafficway_type_group in ('MINOR ROADS') then '#8B9D9E'
           else '#8B9D9E'
         end as series_color
    from a01_traffic_accidents_vw
group by trafficway_type_group


-- Top 10 Streets with most crashes
  select street_name,
         count(*) as crash_count,
         sum(injuries_total) as total_injuries,
         to_char(round(avg(posted_speed_limit))) || ' km/h' as average_speed
    from a01_traffic_accidents_vw
group by street_name
order by crash_count desc
 fetch first 10 row only


-- MAP ----------------------------------
-- Map: Work Zone Crashes
  select latitude,
         longitude
    from a01_traffic_accidents_vw
   where latitude is not null
     and longitude is not null
     and work_zone = 'Y'

-- Map: Dooring Crashes
  select latitude,
         longitude
    from a01_traffic_accidents_vw
   where latitude is not null
     and longitude is not null
     and dooring = 'Y'

-- Map: Fatal Crashes
  select latitude,
         longitude
    from a01_traffic_accidents_vw
   where latitude is not null
     and longitude is not null
     and nvl(injuries_fatal, 0) > 0
-------------------------------------------



-- DASHBOARD 5: Outcomes & Severity ___________________________________________

-- KPIs
with fatal_crashes as (
     select count(*) as crash_count
       from a01_traffic_accidents_vw
      where nvl(injuries_fatal,0) > 0
),
injury_crashes as (
     select injuries_total,
            count(*) as crash_count
       from a01_traffic_accidents_vw
      where nvl(injuries_total,0) > 0
   group by injuries_total
),
damage_crashes as (
     select damage,
            count(*) as crash_count
       from a01_traffic_accidents_vw
      where damage not in ('$500 OR LESS')
   group by damage
),
hit_run_crashes as (
     select hit_and_run,
            count(*) as crash_count
       from a01_traffic_accidents_vw
      where hit_and_run in ('Y')
   group by hit_and_run
)

select 'fa fa-ambulance' ICON_CLASS,
       'u-color-9' ICON_COLOR_CLASS,
       'Crashes with Injuries' list_text,
       to_char(sum(crash_count), 'FM999G999G999') list_title,
       1 sort_value
  from injury_crashes
union all

select 'fa fa-exclamation-triangle' ICON_CLASS,
       'u-color-11' ICON_COLOR_CLASS,
       'Fatal Crashes' list_text,
       to_char(crash_count, 'FM999G999G999') list_title,
       2 sort_value
  from fatal_crashes
union all

select 'fa fa-money' ICON_CLASS,
       'u-color-12' ICON_COLOR_CLASS,
       'Crashes with Damage over 500$' list_text,
       to_char(sum(crash_count), 'FM999G999G999') list_title,
       3 sort_value
  from damage_crashes
union all

select 'fa fa-person-running-fast' ICON_CLASS,
       'u-color-10' ICON_COLOR_CLASS,
       'Hit and Run Crashes' list_text,
       to_char(crash_count, 'FM999G999G999') list_title,
       4 sort_value
  from hit_run_crashes
order by sort_value;


-- Injuries by Police Report Type
select report_type_group,
       most_severe_injury_group,
       count(*) as crash_count,
       case 
           when report_type_group in ('DESK REPORT') then '#8F98B7'--'#c04f4f'          
           else '#6b7494'
       end as series_color
  from a01_traffic_accidents_vw
 where most_severe_injury_group not in ('UNKNOWN')
   and report_type_group not in ('UNKNOWN', 'AMENDED')
group by report_type_group,
         most_severe_injury_group
order by crash_count desc;


-- Year Over Year Change by Most Severe Injury (Current vs Prev Year)
-- Confronto Crash Count per Most Severe Injury rispetto all'anno precedente (YoY = Year Over Year)
with msi_crashes as (  
  select most_severe_injury_group,
         crash_year,
         count(*) as crash_count,
         coalesce(lag(count(*)) over ( partition by most_severe_injury_group
                                  order by crash_year ), 0) as prev_crash_count
    from a01_traffic_accidents_vw
  where most_severe_injury_group not in ('NO INJURY')
group by most_severe_injury_group,
         crash_year
)
  select most_severe_injury_group,
         crash_year,
         crash_count,
         prev_crash_count,
         crash_count - prev_crash_count as delta_crash_count,
         case
             when prev_crash_count = 0
             then 0
             else round(100 * (crash_count - prev_crash_count) / prev_crash_count, 2)
         end as delta_perc_crash_count
    from msi_crashes
   where crash_year = :P47_YEAR
order by most_severe_injury_group,
         crash_year;


-- Crashes Damage Levels
  select damage,
         count(*) as crash_count,
         case 
             when damage in ('OVER $1,500') then '#c04f4f'          
             when damage in ('$500 OR LESS') then '#6b7494'          
             else '#8F98B7'
         end as series_color
    from a01_traffic_accidents_vw
group by damage 


-- No Injury Crashes
select hit_and_run_group,
       count(*) as crash_count,
       case 
         when hit_and_run_group = 'HIT & RUN' then '#c04f4f'
         when hit_and_run_group = 'NOT REPORTED' then '#ADB6D2'
         else '#6b7494'
       end as series_color
  from a01_traffic_accidents_vw
 where (nvl(injuries_total,0) = 0)
   and hit_and_run_group not in ('NOT REPORTED')
group by hit_and_run_group;


-- Fatal Crashes
select hit_and_run_group,
       count(*) as crash_count,
       case 
         when hit_and_run_group = 'HIT & RUN' then '#c04f4f'
         when hit_and_run_group = 'NOT REPORTED' then '#ADB6D2'
         else '#6b7494'
       end as series_color
  from a01_traffic_accidents_vw
 where nvl(injuries_fatal,0) > 0
   and hit_and_run_group not in ('NOT REPORTED')
group by hit_and_run_group;


-- Injury Severity by Hour of the Day
  select crash_hour,
         sum(injuries_fatal) injuries_fatal_count,
         sum(injuries_total - injuries_fatal) injuries_non_fatal_count
    from a01_traffic_accidents_vw
group by crash_hour
order by crash_hour