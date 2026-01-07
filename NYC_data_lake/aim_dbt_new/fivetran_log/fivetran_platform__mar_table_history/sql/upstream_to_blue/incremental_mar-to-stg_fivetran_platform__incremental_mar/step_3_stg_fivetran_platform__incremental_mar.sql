with base as (

    select * 
    from "fivetran_log"."public_fivetran_platform"."stg_fivetran_platform__incremental_mar_tmp"
),

fields as (
    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    cast(null as TEXT) as 
    
    connection_id
    
 , 
    
    
    connection_name
    
 as 
    
    connection_name
    
, 
    
    
    connector_id
    
 as 
    
    connector_id
    
, 
    cast(null as TEXT) as 
    
    connector_name
    
 , 
    
    
    destination_id
    
 as 
    
    destination_id
    
, 
    
    
    free_type
    
 as 
    
    free_type
    
, 
    
    
    measured_date
    
 as 
    
    measured_date
    
, 
    
    
    schema_name
    
 as 
    
    schema_name
    
, 
    
    
    sync_type
    
 as 
    
    sync_type
    
, 
    
    
    table_name
    
 as 
    
    table_name
    
, 
    
    
    incremental_rows
    
 as 
    
    incremental_rows
    
, 
    
    
    updated_at
    
 as 
    
    updated_at
    



    from base
),

final as (

    select
        
    coalesce(
        cast(connection_name as TEXT),
    
        cast(connector_name as TEXT),
    
        cast(connector_id as TEXT)
    
    )
 as connection_name,
        destination_id,
        free_type,
        cast(measured_date as timestamp) as measured_date,
        schema_name,
        sync_type,
        table_name,
        updated_at,
        _fivetran_synced,
        incremental_rows
    from fields
)

select * 
from final