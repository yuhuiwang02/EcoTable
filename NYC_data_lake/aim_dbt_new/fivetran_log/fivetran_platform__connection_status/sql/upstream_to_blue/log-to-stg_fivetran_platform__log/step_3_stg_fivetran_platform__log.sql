

with base as (

    select * 
    from "fivetran_log"."public_fivetran_platform"."stg_fivetran_platform__log_tmp"
),

fields as (
    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    connection_id
    
 as 
    
    connection_id
    
, 
    cast(null as TEXT) as 
    
    connector_id
    
 , 
    
    
    event
    
 as 
    
    event
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    message_data
    
 as 
    
    message_data
    
, 
    
    
    message_event
    
 as 
    
    message_event
    
, 
    
    
    sync_id
    
 as 
    
    sync_id
    
, 
    
    
    time_stamp
    
 as 
    
    time_stamp
    
, 
    
    
    transformation_id
    
 as 
    
    transformation_id
    



    from base
),

field_conversion as (
    select
        *,
        message_data as message_data_string
    from fields
),

final as (

    select
        id as log_id,
        sync_id,
        cast(time_stamp as timestamp) as created_at,
        
    coalesce(
        cast(connection_id as TEXT),
    
        cast(connector_id as TEXT)
    
    )
 as connection_id,
        case when transformation_id is not null and event is null then 'TRANSFORMATION'
        else event end as event_type,
        message_data_string as message_data,
        case
        when transformation_id is not null and message_data_string like '%has succeeded%' then 'transformation run success'
        when transformation_id is not null and message_data_string like '%has failed%' then 'transformation run failed'
        else message_event end as event_subtype,
        transformation_id
    from field_conversion
)

select * 
from final