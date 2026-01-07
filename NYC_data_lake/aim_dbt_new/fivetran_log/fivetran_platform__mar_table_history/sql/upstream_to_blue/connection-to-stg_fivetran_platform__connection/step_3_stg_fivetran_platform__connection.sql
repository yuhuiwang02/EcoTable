with connection_base as (
        select * 
        from "fivetran_log"."public_fivetran_platform"."stg_fivetran_platform__connection_tmp"
    ),

    connection_fields as (
        select
            
    
    
    _fivetran_deleted
    
 as 
    
    _fivetran_deleted
    
, 
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    connecting_user_id
    
 as 
    
    connecting_user_id
    
, 
    
    
    connection_id
    
 as 
    
    connection_id
    
, 
    
    
    connection_name
    
 as 
    
    connection_name
    
, 
    
    
    connector_type
    
 as 
    
    connector_type
    
, 
    cast(null as TEXT) as 
    
    connector_type_id
    
 , 
    
    
    destination_id
    
 as 
    
    destination_id
    
, 
    
    
    paused
    
 as 
    
    paused
    
, 
    cast(null as integer) as 
    
    service_version
    
 , 
    
    
    signed_up
    
 as 
    
    signed_up
    



        from connection_base
    ),

    renamed_fields as (
        select
            connection_id as connection_id,
            connection_name as connection_name,
            connector_type_id,
            
    coalesce(
        cast(connector_type_id as TEXT),
    
        cast(connector_type as TEXT)
    
    )
 as connector_type,
            destination_id,
            connecting_user_id,
            paused as is_paused,
            signed_up as set_up_at,
            coalesce(_fivetran_deleted,  false) as is_deleted,
            _fivetran_synced
        from connection_fields
    ),

    sorted_rows as (
        select
            *,
            row_number() over (partition by connection_name, destination_id order by _fivetran_synced desc) as nth_last_record
        from renamed_fields
    ),

final as (
    select
        connection_id,
        connection_name,
        connector_type,
        destination_id,
        connecting_user_id,
        is_paused,
        set_up_at,
        is_deleted
    from sorted_rows
    where nth_last_record = 1
)

select * 
from final